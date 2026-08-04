"""
verifier/train_verifier.py

QwenVerifier(base Qwen2.5-VL + 새 LoRA + binary classification head)를
generate_verifier_data.py가 만든 (point_px, label) 데이터로 학습.

grounding LoRA(train.py)와 어댑터가 완전히 분리돼 있어서, 나중에 추론 시 base
모델은 하나만 로드해두고 필요에 따라 model.set_adapter()로 grounding <-> verifier를
스왑해서 쓸 수 있다 (RegionFocus의 Step1/3/4는 grounding adapter, Step2/5는 이
verifier adapter) - 그 스왑 통합은 이 학습이 끝난 뒤 region_focus.py 쪽에 별도로
붙일 예정, 이 스크립트는 학습까지만 담당한다.

[클래스 불균형]
generate_verifier_data.py가 만드는 데이터는 gt_center(항상 label=1)를 포함하지만
그래도 label=0(오답)이 label=1보다 많을 수 있다(모델이 잘 맞히는 샘플일수록 후보
전부가 label=1이라 오히려 반대로 쏠릴 수도 있음) - 그래서 자동으로 pos_weight를
계산해서 BCEWithLogitsLoss에 넣는다(--pos_weight로 수동 지정도 가능).

[2026-08 수정: VRAM OOM 버그 픽스]
processor를 min_pixels/max_pixels 상한 없이 로드하고 있었다 - qwen.py는 항상 이 상한
(DEFAULT_MAX_PIXELS=640*28*28)을 걸어서 이미지 한 장이 비주얼 토큰 몇천 개로 불어나는
걸 막는데, 여기는 AutoProcessor.from_pretrained(model_id)만 불러서 기본값(훨씬 큰
해상도 허용)을 그대로 쓰고 있었다 - VRAM 16GB+공유램까지 끌어 쓰다 죽는 원인이었을
가능성이 높음. qwen.py와 동일한 기본값으로 상한을 걸도록 고쳤고, 그래도 부족하면
--gradient_checkpointing을 켜거나 --batch_size를 낮출 것(1까지 낮춰도 됨 - 배치를
줄이는 대신 필요하면 나중에 gradient accumulation을 추가하는 것도 방법).

사용법 (verifier/가 vlm_agent/ 밑에 있든 나란히 있든 sys.path 자동탐지로 알아서 찾음 -
PYTHONPATH 수동 설정 불필요):
  cd vlm_agent/verifier
  python train_verifier.py \
      --jsonl verifier_train.jsonl --val_jsonl verifier_val.jsonl \
      --output_dir ./checkpoints/verifier-v1 \
      --epochs 2 --batch_size 1 --gradient_checkpointing
"""

import argparse
import os
import sys

# vlm_agent(coord_utils.py 등이 있는 폴더)를 sys.path에 넣는다 - generate_verifier_data.py
# 상단 주석 참고. verifier/가 vlm_agent/ 밑에 있든 나란히 있든 자동으로 찾는다.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = None  # coord_utils.py 등이 실제로 있는 폴더 (아래서 찾아서 채움)
for _candidate in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "vlm_agent")):
    _candidate = os.path.abspath(_candidate)
    if os.path.isfile(os.path.join(_candidate, "coord_utils.py")):
        _BASE_DIR = _candidate
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from coord_utils import load_jsonl
from data import build_verifier_messages
from model import MODEL_ID, QwenVerifier
from qwen import DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS


def _resolve_image_path(image_path):
    """
    generate_verifier_data.py는 image_path를 절대경로가 아니라 자신의 _BASE_DIR
    (vlm_agent 루트) 기준 상대경로로 저장한다 - 로컬(Windows)에서 만든 jsonl을
    서버(Linux)에서 학습에 쓸 때 절대경로(드라이브 문자/마운트 경로)가 서로 달라서
    깨지는 걸 피하기 위함. 여기서는 그 상대경로를 "이 머신의" _BASE_DIR 기준으로
    다시 풀어준다 - 두 머신 모두 vlm_agent 밑 폴더 구조(data/processed/images/...)는
    같다는 전제. generate_verifier_data.py의 동명 함수와 동일한 폴백 순서
    (그대로/存재하면 사용 -> _BASE_DIR 기준 -> 앞의 "../" 벗겨가며 재시도)를 쓴다.
    """
    if os.path.isabs(image_path) or os.path.exists(image_path):
        return image_path

    if _BASE_DIR:
        candidate = os.path.join(_BASE_DIR, image_path)
        if os.path.exists(candidate):
            return candidate

        stripped = image_path
        while stripped.startswith("../") or stripped.startswith("..\\"):
            stripped = stripped[3:]
            candidate = os.path.join(_BASE_DIR, stripped)
            if os.path.exists(candidate):
                return candidate

    return image_path


class VerifierDataset(Dataset):
    def __init__(self, jsonl_path):
        self.records = load_jsonl(jsonl_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def make_collate_fn(processor):
    def collate(batch):
        all_messages = [
            build_verifier_messages(
                _resolve_image_path(rec["image_path"]), rec["instruction"], rec["point_px"]
            )
            for rec in batch
        ]
        labels = [float(rec["label"]) for rec in batch]

        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in all_messages
        ]
        image_inputs = []
        for m in all_messages:
            imgs, _ = process_vision_info(m)
            image_inputs.extend(imgs)

        inputs = processor(
            text=texts, images=image_inputs, padding=True, return_tensors="pt",
        )
        inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
        return inputs

    return collate


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
            )
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="generate_verifier_data.py 출력 (TRAIN 데이터 기반)")
    ap.add_argument("--val_jsonl", default=None,
                     help="선택: 학습 중 모니터링용 검증셋. 반드시 train 쪽에서 별도로 떼어낸 "
                          "데이터로 만들 것 - test/val.jsonl을 여기 넣으면 결국 최종 채점용 "
                          "held-out을 학습 모니터링에 써버리는 셈이라 의미가 훼손됨.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--pos_weight", type=float, default=None,
                     help="positive class 가중치(불균형 보정). 기본값은 n_neg/n_pos로 자동 계산")
    ap.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS,
                     help="qwen.py와 동일한 기본 상한을 씀 - 이걸 안 걸면 processor가 "
                          "기본값(훨씬 큰 해상도)을 그대로 써서 이미지 한 장이 비주얼 토큰 "
                          "수천 개로 불어나 VRAM이 크게 튈 수 있음.")
    ap.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--gradient_checkpointing", action="store_true",
                     help="VRAM이 부족할 때 켤 것 - 역전파에 필요한 중간 activation을 저장해두지 "
                          "않고 필요할 때 다시 계산해서 메모리를 크게 아끼는 대신 조금 느려짐.")
    ap.add_argument("--grad_accum", type=int, default=1,
                     help="train.py와 동일한 목적 - --batch_size를 낮춰서 VRAM을 아끼는 대신, "
                          "이 값만큼 스텝을 모아서 한 번에 optimizer.step()을 밟아 실질적인 "
                          "배치 크기(batch_size * grad_accum)를 흉내낸다. 예: --batch_size 1 "
                          "--grad_accum 4 면 메모리는 배치 1개만큼만 쓰면서 배치 4개를 모은 것과 "
                          "같은 효과의 업데이트를 함.")
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    model = QwenVerifier(
        model_id=args.model_id, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    ).to(device)
    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        # gradient checkpointing은 KV 캐시(디코딩 가속용)와 같이 못 씀 - 우리는 generate()를
        # 안 쓰고 forward 한 번만 하는 분류기라 use_cache를 꺼도 아무 영향 없음.
        model.backbone.config.use_cache = False
    model.backbone.print_trainable_parameters()

    train_ds = VerifierDataset(args.jsonl)
    labels = [r["label"] for r in train_ds.records]
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    pos_weight = args.pos_weight if args.pos_weight is not None else (n_neg / max(n_pos, 1))
    print(f"[train_verifier] n={len(labels)} pos={n_pos} neg={n_neg} pos_weight={pos_weight:.3f}")

    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = None
    if args.val_jsonl:
        val_ds = VerifierDataset(args.val_jsonl)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        )

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            labels_t = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
            )
            # grad_accum>1이면 grad_accum번의 loss 평균을 낸 것과 같은 크기가 되도록
            # 미리 나눠준다 (안 그러면 accum 스텝 수만큼 그래디언트가 커짐).
            loss = criterion(logits, labels_t) / args.grad_accum
            loss.backward()
            # optimizer.step()은 grad_accum 스텝마다 한 번만 - 그 사이엔 그래디언트가
            # 계속 누적(accumulate)된다 (zero_grad를 안 부르니까). 마지막 배치가
            # grad_accum으로 안 나눠떨어지고 끝나도 아래 for문 밖에서 한 번 더 처리한다.
            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * args.grad_accum
            if step % args.log_every == 0:
                print(f"[epoch {epoch}] step {step}/{len(train_loader)} loss={loss.item() * args.grad_accum:.4f}")
        # 이 epoch이 grad_accum으로 안 나눠떨어지는 스텝 수로 끝났으면 남은 그래디언트를 마저 반영.
        if len(train_loader) % args.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / max(len(train_loader), 1)
        msg = f"[epoch {epoch}] 평균 loss={avg_loss:.4f}"
        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            msg += f" val_acc={val_acc:.4f}"
        print(msg)

    model.save_pretrained(args.output_dir)
    print(f"[train_verifier] 저장 완료: {args.output_dir}")


if __name__ == "__main__":
    main()
