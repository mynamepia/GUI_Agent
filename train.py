"""
train.py

Qwen2.5-VL-3B-Instruct GUI grounding SFT (LoRA)

- train/val: data/processed/train.jsonl, val.jsonl (prepare_dataset.py로 생성)
- CPU-only 환경(예: 미니PC)에서도 일단 돌아가도록 기본값은 매우 보수적으로 설정.
  GPU 있으면 --bf16, --batch_size 등을 올려서 사용.
- 모델 로딩은 qwen.py의 load_model_and_processor()를 그대로 가져다 쓴다.
  (이전 버전은 Qwen2VLForConditionalGeneration을 import했는데 실제 체크포인트인
  Qwen2.5-VL과 클래스가 어긋나서 로딩/학습이 정상 동작하지 않는 버그가 있었음)

필요 패키지:
    pip install torch torchvision transformers accelerate peft qwen-vl-utils pillow

[좌표 포맷 - qwen.py/test.py/evaluation.py와 반드시 동일해야 함]
모델 출력은 이미지 내 클릭 좌표를 0~1000으로 정규화한 "(x,y)" 텍스트로 생성한다.
  예: 이미지가 1920x1080이고 실제 클릭 지점이 (960,540)이면 -> 정규화 좌표 (500,500)
좌표 변환 로직(PROMPT_TEMPLATE, build_target_text, load_jsonl 등)은 coord_utils.py 공용
모듈에 있고 test.py/evaluation.py도 같이 import해서 씀 - 포맷 바꿀 땐 coord_utils.py만
고치면 됨. 학습용 Dataset 클래스(GUIGroundingDataset)는 train.py 여기서만 쓰므로 별도
data_utils.py 없이 이 파일에 바로 정의한다.

[label masking]
collate_fn에서 assistant 답변(좌표) 토큰만 loss 대상으로 남기고, 프롬프트(이미지+지시문+
챗템플릿 boilerplate) 부분은 -100으로 마스킹한다. 프롬프트까지 label에 포함시키면
teacher-forcing 특성상 "이미 준 입력을 그대로 재생성"하는 쉬운 loss가 섞여 들어가서
실제 좌표 예측 성능을 반영하지 못하는 loss 숫자가 나오게 됨.
패딩 마스킹은 pad_token_id와의 값 비교가 아니라 attention_mask로 한다 - Qwen 토크나이저는
보통 pad_token == eos_token이라, id로 비교하면 답변 맨 끝의 진짜 eos(정답의 일부로 학습해야
하는 토큰)까지 -100으로 지워버리는 문제가 있었음.

[학습 후 평가]
Trainer.evaluate()는 teacher-forcing loss만 계산하고 실제 좌표 정확도는 알려주지 않기
때문에, 학습이 끝나면 evaluation.py의 run_generation_eval()로 val set에 대해 진짜
generate() 기반 click accuracy(+ Wave-UI/ScreenSpot-v2 breakdown)를 한 번 더 돌린다.
매 epoch마다 돌리기엔 generation이 느려서(특히 CPU) 기본은 학습 종료 후 1회만 실행.
--eval_limit으로 스모크 테스트 시 표본 수를 줄일 수 있고, --skip_gen_eval로 아예 끌 수도 있다.

사용법 (스모크 테스트, 몇 step만 돌려서 파이프라인 검증):
  python train.py --max_steps 5 --batch_size 1 --eval_limit 5

실제 학습:
  python train.py --num_train_epochs 1 --batch_size 2 --grad_accum 8
"""

import argparse
import json
import os

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from coord_utils import PROMPT_TEMPLATE, build_target_text, load_jsonl
from evaluation import format_report, run_generation_eval
from qwen import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, MODEL_ID, load_model_and_processor


class GUIGroundingDataset(Dataset):
    """SFT 학습용 데이터셋. train.py 전용."""

    def __init__(self, jsonl_path: str):
        self.records = load_jsonl(jsonl_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        prompt = PROMPT_TEMPLATE.format(instruction=rec["instruction"])
        target = build_target_text(rec["point"], rec["resolution"])
        return {
            "id": rec["id"],
            "image": image,
            "prompt": prompt,
            "target": target,
            "raw": rec,
        }


def build_collate_fn(processor):
    def collate_fn(batch):
        texts = []
        prompt_texts = []
        image_inputs_all = []

        for item in batch:
            user_content = [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": item["prompt"]},
            ]
            full_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item["target"]},
            ]
            prompt_messages = [{"role": "user", "content": user_content}]

            # 정답까지 포함한 전체 시퀀스
            full_text = processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            # 프롬프트만 (assistant 생성 시작 지점까지) - label 마스킹 길이 계산용
            prompt_text = processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            texts.append(full_text)
            prompt_texts.append(prompt_text)

            image_inputs, _ = process_vision_info(full_messages)
            # 주의: processor(images=...)는 배치 전체에 대해 "펼쳐진(flat)" 이미지 리스트를
            # 기대한다. 샘플별 리스트를 그대로 append하면 (list of list가 되어) batch_size가
            # 1보다 클 때 이미지 개수와 <image> 플레이스홀더 토큰 개수가 어긋나서 깨진다.
            image_inputs_all.extend(image_inputs)

        inputs = processor(
            text=texts,
            images=image_inputs_all,
            return_tensors="pt",
            padding=True,
        )

        # 같은 이미지로 프롬프트만 다시 토큰화해서, 실제 프롬프트 토큰 길이를 구함.
        # (padding_side="right" 가정 -> full 시퀀스의 앞쪽 prompt_len개 토큰이 프롬프트 구간)
        prompt_inputs = processor(
            text=prompt_texts,
            images=image_inputs_all,
            return_tensors="pt",
            padding=True,
        )

        labels = inputs["input_ids"].clone()
        for i in range(len(batch)):
            # attention_mask 기반으로 길이를 재야 함. pad_token_id로 비교하면 Qwen 토크나이저
            # 특성상 pad_token == eos_token인 경우가 많아서, 정답 끝의 진짜 eos까지 프롬프트로
            # 오인해서 잘라내 버리는 문제가 생긴다.
            prompt_len = int(prompt_inputs["attention_mask"][i].sum())
            labels[i, :prompt_len] = -100

        # 오른쪽 패딩 구간(attention_mask == 0)도 loss에서 제외.
        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return inputs

    return collate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="./data/processed/train.jsonl")
    ap.add_argument("--val_jsonl", default="./data/processed/val.jsonl")
    ap.add_argument("--output_dir", default="./checkpoints/qwen2.5vl-3b-gui-lora")
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1,
                     help="스모크 테스트용. -1이면 epoch 기준으로 끝까지 학습")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--bf16", action="store_true", help="GPU에서만 사용 권장")
    ap.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS,
                     help="이미지당 최소 비주얼 토큰 관련 픽셀 수 (CPU 메모리 보호용)")
    ap.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS,
                     help="이미지당 최대 비주얼 토큰 관련 픽셀 수 (CPU 메모리 보호용)")
    ap.add_argument("--eval_max_new_tokens", type=int, default=32,
                     help="학습 후 generation eval에서 좌표 생성 시 max_new_tokens")
    ap.add_argument("--eval_limit", type=int, default=None,
                     help="학습 후 generation eval에 쓸 val 샘플 수 제한 (스모크 테스트용)")
    ap.add_argument("--skip_gen_eval", action="store_true",
                     help="학습 후 generation 기반 click accuracy 평가를 건너뜀")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    device_map = None
    dtype = torch.bfloat16 if (args.bf16 and use_cuda) else torch.float32

    model, processor = load_model_and_processor(
        model_id=args.model_id,
        device_map=device_map,
        dtype=dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    # label masking에서 "앞쪽 prompt_len개 토큰 = 프롬프트"를 가정하므로 오른쪽 패딩 고정.
    processor.tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    # PEFT(LoRA) + gradient_checkpointing 조합에서는 입력 임베딩의 requires_grad가 꺼져 있어서
    # 이 호출 없이는 backward에서 "element 0 of tensors does not require grad" 에러가 난다.
    model.enable_input_require_grads()
    model.config.use_cache = False  # gradient checkpointing과 캐시는 같이 못 씀
    model.print_trainable_parameters()

    train_ds = GUIGroundingDataset(args.train_jsonl)
    val_ds = GUIGroundingDataset(args.val_jsonl)
    print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=1,
        eval_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=max(1, args.max_steps // 2) if args.max_steps > 0 else None,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=args.bf16 and use_cuda,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=build_collate_fn(processor),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[done] LoRA adapter saved to {args.output_dir}")

    if not args.skip_gen_eval:
        print("[train.py] running post-training generation eval on val set ...")
        val_metrics = run_generation_eval(
            model,
            processor,
            args.val_jsonl,
            max_new_tokens=args.eval_max_new_tokens,
            limit=args.eval_limit,
            predictions_out_path=os.path.join(args.output_dir, "val_predictions.jsonl"),
        )
        print(format_report(val_metrics, title="val (post-training)"))
        with open(os.path.join(args.output_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()