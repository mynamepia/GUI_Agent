"""
train_planner.py

Qwen2.5-VL-3B-Instruct planner SFT (LoRA) - agent/planner.py의 ReAct 스타일 출력
스키마(reasoning/action/target_description/text/status/answer JSON)를 생성하도록
파인튜닝한다.

vlm_agent/train.py(grounding LoRA 학습)의 구조를 그대로 재사용한다 - LoRA 설정,
label masking(프롬프트는 -100, assistant JSON 답변만 loss 대상), --resume_from_checkpoint,
--init_adapter_dir, TrainingArguments 값들이 거의 동일하다. 다른 점은 딱 두 가지:
  1) 타깃이 좌표 "(x,y)" 텍스트가 아니라 planner의 JSON 액션 객체 문자열이다.
  2) 시스템 프롬프트가 필요하다 - grounding 학습은 system 메시지 없이 user 메시지만
     썼지만, planner는 agent/planner.py의 _SYSTEM_PROMPT를 그대로 가져다 쓴다(추론
     시점과 동일한 프롬프트여야 학습/추론 분포가 일치함 - 그래서 텍스트를 복제하지 않고
     import해서 씀).

[데이터]
prepare_planner_dataset.py가 만든 data/processed/planner_train.jsonl, planner_val.jsonl을
읽는다. 각 행: {task_id, platform, image_path, instruction, history_text, target(dict)}.
target dict를 json.dumps해서 assistant 답변으로 쓴다(순서/생략 필드는 이미
prepare_planner_dataset.py에서 planner.py 출력 관례에 맞춰 정리해둠).

[eval]
grounding 학습(train.py)의 run_generation_eval()에 해당하는 "학습 후 실제 generate() 기반
정확도 측정"은 아직 만들지 않았다 - planner 출력은 좌표 hit/miss처럼 단순 채점이 안 되고
(자연어 target_description을 무엇과 비교해 "정답"으로 칠지가 불명확함), 사용자도 이번
요청에서 "eval이 필요하지 않으면 넘어가도 된다"고 명시함. 대신 훨씬 가벼운 스모크
체크(--gen_check)를 넣었다 - 학습 후 val 표본 일부에 대해 실제로 generate()를 돌려서
agent.planner._parse_planner_action()이 파싱에 성공하는 비율(JSON 포맷 준수율)만 본다.
이건 "맞는 답을 내는지"가 아니라 "포맷이 무너지지 않았는지"만 보는 최소한의 안전장치다.

사용법 (스모크 테스트):
  python train_planner.py --max_steps 5 --batch_size 1

실제 학습:
  python train_planner.py --num_train_epochs 2 --batch_size 2 --grad_accum 8

이어서 학습 재개:
  python train_planner.py --resume_from_checkpoint auto --num_train_epochs 3 ...(동일 output_dir)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

_HERE = os.path.dirname(os.path.abspath(__file__))
_VLM_AGENT_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_AGENT_DIR = os.path.join(_VLM_AGENT_DIR, "agent")
for _candidate in (_VLM_AGENT_DIR, _AGENT_DIR):
    if _candidate not in sys.path and os.path.isdir(_candidate):
        sys.path.insert(0, _candidate)

from planner import _SYSTEM_PROMPT, _parse_planner_action  # noqa: E402
from qwen import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, MODEL_ID, load_model_and_processor  # noqa: E402


def _build_user_text(instruction: str, history_text: str) -> str:
    # agent/planner.py plan_next_action()의 user_text 조립과 동일하게 맞춘다 - 학습/추론
    # 프롬프트가 어긋나면 안 됨.
    return (
        f'Task: "{instruction}"\n\n'
        f"History:\n{history_text}\n\n"
        "The attached image is the current screenshot. What is the next action?"
    )


class PlannerDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        user_text = _build_user_text(rec.get("instruction", ""), rec.get("history_text", ""))
        target_text = json.dumps(rec["target"], ensure_ascii=False)
        return {
            "image": image,
            "user_text": user_text,
            "target_text": target_text,
            "raw": rec,
        }


def build_collate_fn(processor):
    def collate_fn(batch):
        texts, prompt_texts, image_inputs_all = [], [], []

        for item in batch:
            user_content = [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": item["user_text"]},
            ]
            full_messages = [
                {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": item["target_text"]},
            ]
            prompt_messages = [
                {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]

            full_text = processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            texts.append(full_text)
            prompt_texts.append(prompt_text)

            image_inputs, _ = process_vision_info(full_messages)
            # train.py와 동일한 이유: processor(images=...)는 배치 전체에 대해 flat list를
            # 기대하므로 샘플별 리스트를 extend해야 함(append하면 batch_size>1에서 깨짐).
            image_inputs_all.extend(image_inputs)

        inputs = processor(text=texts, images=image_inputs_all, return_tensors="pt", padding=True)
        prompt_inputs = processor(
            text=prompt_texts, images=image_inputs_all, return_tensors="pt", padding=True
        )

        labels = inputs["input_ids"].clone()
        for i in range(len(batch)):
            # train.py와 동일: attention_mask 기반 길이로 마스킹(pad_token_id==eos_token_id인
            # 경우가 많아서 id 비교로 하면 정답 끝의 진짜 eos까지 지워버리는 문제가 있음).
            prompt_len = int(prompt_inputs["attention_mask"][i].sum())
            labels[i, :prompt_len] = -100
        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return inputs

    return collate_fn


class EmptyCacheCallback(TrainerCallback):
    """
    (2026-08 추가) 실측으로 확인된 문제: 학습 스텝을 도는 동안 GPU 메모리(할당된 실제
    사용량이 아니라 "reserved"로 캐싱 할당자가 쥐고 있는 양)가 점점 불어나서, 42% 지점에서
    이미 27GB까지 올라가는 걸 확인함. 데이터셋 안의 샘플마다 이미지 해상도(700k pixel
    상한 안에서도 원본 스크린샷 크기에 따라 실제 비주얼 토큰 수가 다름)/history_text
    길이가 제각각이라, PyTorch의 caching allocator가 스텝마다 다른 크기의 블록을 요청하며
    파편화되기 쉽다 - gen_check 루프에는 이미 매 반복 torch.cuda.empty_cache()를 넣어뒀는데
    정작 메인 학습 루프에는 없었던 게 이 문제의 원인일 가능성이 높다. N스텝마다 캐시를
    비워서 파편화가 누적되는 걸 막는다(빈도가 너무 잦으면 캐시 재할당 오버헤드로 오히려
    느려질 수 있어서 기본은 20스텝마다).
    """

    def __init__(self, every_n_steps: int = 20):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps > 0 and state.global_step % self.every_n_steps == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return control


def run_gen_check(model, processor, val_jsonl: str, limit: int, max_new_tokens: int):
    """
    학습 후 최소한의 안전장치: 실제 generate()를 돌려서 JSON 포맷을 지키는지(파싱 성공률)만
    확인한다. "정답인지"는 안 본다(자연어 target_description을 채점할 기준이 없음) - 이건
    run_generation_eval()의 대체재가 아니라 훨씬 가벼운 스모크 체크.
    """
    from qwen_vl_utils import process_vision_info as _pvi

    ds = PlannerDataset(val_jsonl)
    n = min(limit, len(ds)) if limit else len(ds)
    n_ok = 0
    was_training = model.training
    # (2026-08 버그 수정) 학습 중엔 gradient checkpointing과의 호환을 위해
    # model.config.use_cache = False로 꺼뒀었는데(main()의 학습 세팅 부분), gen_check가
    # 그 상태를 그대로 물려받아서 generate()가 KV 캐시 없이 매 새 토큰마다 전체 시퀀스를
    # 다시 계산했다 - 이미지 토큰까지 낀 300토큰 생성을 이 상태로 반복하면 연산량/메모리가
    # 크게 뛰어서 실측으로 RAM이 넘치는 원인이 됐다. generate() 전에 다시 켜고, 끝나면
    # 원래 상태(False)로 복원한다(원래 state를 기억해뒀다가 되돌리는 방식 - 이 함수를
    # 학습 도중에도 재사용할 가능성을 감안).
    original_use_cache = model.config.use_cache
    model.config.use_cache = True
    model.eval()
    with torch.no_grad():
        for i in range(n):
            item = ds[i]
            messages = [
                {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": item["user_text"]},
                    ],
                },
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = _pvi(messages)
            inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
            response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            parsed = _parse_planner_action(response)
            if not parsed.get("_parse_failed"):
                n_ok += 1
            # 이미지+300토큰 생성마다 남는 중간 텐서를 즉시 회수 - 표본이 많으면(--gen_check_limit)
            # 누적 파편화로 뒤로 갈수록 메모리 압박이 커질 수 있어서 매 반복 정리.
            del inputs, gen_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    model.config.use_cache = original_use_cache
    if was_training:
        model.train()
    rate = n_ok / n if n else 0.0
    print(f"[train_planner] gen_check: JSON 파싱 성공 {n_ok}/{n} ({rate:.1%})")
    return {"n": n, "n_ok": n_ok, "parse_success_rate": rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/processed/planner_train.jsonl")
    ap.add_argument("--val_jsonl", default="data/processed/planner_val.jsonl")
    ap.add_argument("--output_dir", default="checkpoints/qwen2.5vl-3b-planner-lora")
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=-1, help="스모크 테스트용. -1이면 epoch 기준")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--bf16", action="store_true", help="GPU에서만 사용 권장")
    ap.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--init_adapter_dir", default=None,
                     help="지정하면 새 LoRA 대신 이 경로의 기존 어댑터를 이어서 학습")
    ap.add_argument("--load_in_4bit", action="store_true",
                     help="(2026-08 추가) QLoRA 스타일 - base 모델을 4bit로 양자화해서 로드하고 "
                          "그 위에 LoRA만 학습한다(bitsandbytes 필요: pip install bitsandbytes). "
                          "max_pixels=700000(그라운딩 체크포인트와 맞춘 해상도)을 유지한 채로 "
                          "OOM이 나는 경우를 위한 옵션 - base 가중치 메모리를 bf16 대비 약 1/4로 "
                          "줄여서 그만큼을 이미지 토큰(활성화 메모리) 쪽에 더 쓸 수 있게 해준다. "
                          "--load_in_8bit와 동시에 켜지 말 것(둘 다 켜면 4bit가 우선 적용됨, "
                          "qwen.py의 BitsAndBytesConfig 참고).")
    ap.add_argument("--load_in_8bit", action="store_true",
                     help="4bit보다는 덜 aggressive한 양자화. 4bit로 정확도 저하가 걱정되면 "
                          "이쪽을 먼저 시도해볼 것.")
    ap.add_argument("--optim", default="adamw_torch",
                     help="옵티마이저 상태(Adam은 파라미터당 2개)도 VRAM을 꽤 먹는다 - OOM이면 "
                          "'adamw_bnb_8bit'(bitsandbytes 필요)로 바꿔서 옵티마이저 메모리도 줄일 것.")
    ap.add_argument("--resume_from_checkpoint", default=None,
                     help="train.py와 동일: 'auto'/'true'면 output_dir의 최신 checkpoint에서 "
                          "optimizer/step까지 이어서 재개. 특정 경로를 줄 수도 있음.")
    ap.add_argument("--save_total_limit", type=int, default=None,
                     help="(2026-08 수정) 예전 기본값 5 -> None(무제한)으로 변경. save_total_limit이 "
                          "걸려있으면 Trainer가 오래된 checkpoint-XXX를 자동 삭제하는데, 그 폴더 "
                          "안의 trainer_state.json(=학습/eval loss 로그 히스토리)도 같이 날아가서 "
                          "'로그가 없어진다'는 문제로 이어짐 - 기본은 아무것도 안 지우도록 바꿈. "
                          "디스크 용량이 실제로 부족해지면 이 값을 명시적으로 지정할 것.")
    ap.add_argument("--gen_check", action="store_true",
                     help="학습 후 val 표본 일부로 JSON 포맷 준수율만 가볍게 확인 (정답 채점 아님)")
    ap.add_argument("--gen_check_limit", type=int, default=30)
    ap.add_argument("--gen_check_max_new_tokens", type=int, default=300)
    ap.add_argument("--empty_cache_every", type=int, default=0,
                     help="(2026-08 추가, 2026-08-09 기본값 20->0 변경) N스텝마다 "
                          "torch.cuda.empty_cache() 호출. 원래는 파편화 완화를 노리고 넣었는데, "
                          "epoch1(이 옵션 없음, paged_adamw_8bit)은 3.64s/it로 안정적이었던 반면 "
                          "epoch2 재개 때 이 콜백 + optim 교체를 동시에 넣고 나서 24~28s/it까지 "
                          "느려지고 메모리도 더 불어났다 - 잦은 empty_cache()가 캐싱 할당자를 "
                          "비웠다가 다음 스텝에서 다시 cudaMalloc을 호출하게 만들어서, Windows "
                          "WDDM 드라이버 상에서는 오히려 재할당 오버헤드/파편화가 커질 수 있다는 "
                          "가설(사용자 지적). 원인이 optim 교체가 아니라 이 콜백일 가능성이 있어서 "
                          "일단 기본값을 꺼짐(0)으로 되돌림 - epoch1과 완전히 같은 조건으로 재현"
                          "테스트한 뒤에 필요하면 다시 켤 것.")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (args.bf16 and use_cuda) else torch.float32
    quantized = args.load_in_4bit or args.load_in_8bit
    if quantized and args.init_adapter_dir:
        # PeftModel.from_pretrained로 기존 어댑터를 이어받는 경로는 bitsandbytes 양자화
        # 모델 위에서도 동작은 하지만 prepare_model_for_kbit_training과의 상호작용을 이
        # 세션에서 검증하지 못했다 - 처음부터 QLoRA로 학습을 시작하는 경로만 우선 지원.
        sys.exit("[train_planner] --load_in_4bit/--load_in_8bit는 --init_adapter_dir와 "
                 "함께 쓰는 경로를 아직 검증하지 못했습니다. 새 LoRA로 시작하거나(권장), "
                 "필요하면 양자화 없이 이어서 학습하세요.")

    model, processor = load_model_and_processor(
        model_id=args.model_id,
        # (2026-08 추가) bitsandbytes 양자화는 GPU에서만 동작하고 device_map이 None이면
        # 레이어를 GPU에 자동 배치하지 않아 양자화 로딩 자체가 실패한다 - 양자화를 켰을 때만
        # device_map="auto"로 바꾼다(기존 비양자화 경로는 그대로 None 유지, accelerate의
        # 자동 배치가 오히려 단일 GPU 환경에서 불필요한 오버헤드/분산을 유발할 수 있어서).
        device_map="auto" if (quantized and use_cuda) else None,
        dtype=dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit and not args.load_in_4bit,
    )
    processor.tokenizer.padding_side = "right"

    # (2026-08 추가) device_map="auto"는 VRAM이 부족하면 accelerate가 일부 레이어를 조용히
    # CPU(또는 디스크)로 오프로드할 수 있다 - 이러면 매 forward/backward마다 그 레이어만큼
    # CPU<->GPU 텐서 복사가 끼어서 step당 소요 시간이 크게 늘어난다(속도 저하의 흔한 원인).
    # hf_device_map이 있으면 찍어서 전부 "cuda:*"인지, 일부가 "cpu"/"disk"로 빠졌는지 바로
    # 확인할 수 있게 함.
    if hasattr(model, "hf_device_map"):
        devices = set(model.hf_device_map.values())
        print(f"[train_planner] hf_device_map에 등장하는 디바이스 종류: {devices}")
        if any(d in ("cpu", "disk") for d in devices):
            print("[train_planner] 경고: 일부 레이어가 cpu/disk로 오프로드됨 - 이게 step당 "
                  "시간이 느린 원인일 수 있음. VRAM을 더 확보하거나(다른 프로세스 종료 등) "
                  "--max_pixels를 낮춰서 재시도해볼 것.")

    if quantized:
        # QLoRA 표준 절차: 양자화된 base를 그대로 LoRA로 학습하면 layernorm 등이 낮은 정밀도로
        # 남아 학습이 불안정해질 수 있어서, peft의 prepare_model_for_kbit_training이 이런
        # 레이어를 fp32로 캐스팅하고 gradient checkpointing 관련 설정까지 맞춰준다 -
        # get_peft_model()보다 먼저 호출해야 함(base 모델 상태에 적용하는 전처리이므로).
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if args.init_adapter_dir:
        print(f"[train_planner] 기존 LoRA 어댑터 이어서 학습: {args.init_adapter_dir}")
        model = PeftModel.from_pretrained(model, args.init_adapter_dir, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_ds = PlannerDataset(args.train_jsonl)
    val_ds = PlannerDataset(args.val_jsonl)
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
        save_total_limit=args.save_total_limit,
        bf16=args.bf16 and use_cuda,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
        optim=args.optim,
    )

    callbacks = []
    if args.empty_cache_every > 0:
        callbacks.append(EmptyCacheCallback(every_n_steps=args.empty_cache_every))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=build_collate_fn(processor),
        callbacks=callbacks,
    )

    resume = args.resume_from_checkpoint
    if resume is not None and resume.lower() in ("auto", "true"):
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[done] planner LoRA adapter saved to {args.output_dir}")

    # (2026-08 추가) eval_loss(에폭 끝날 때 200개 val 전체에 대해 자동으로 계산되는
    # teacher-forcing loss)는 계산은 되는데 trainer_state.json 안 로그 히스토리로만
    # 묻혀 들어가고, grounding 학습(train.py의 val_metrics.json)처럼 따로 읽기 쉬운 파일이
    # 없었다 - "val 결과를 어디서 보냐"는 질문에 대응해서 train/eval loss 로그를 따로 뽑아
    # 저장하고 마지막 eval_loss를 콘솔에도 명확히 찍어준다. (주의: 이건 gen_check처럼 실제
    # generate() 기반 채점이 아니라 teacher-forcing loss라는 점은 여전함 - "정답률"이
    # 아니라 "학습이 잘 수렴하고 있는지/val에서 overfitting 안 하는지"를 보는 지표.)
    log_history = trainer.state.log_history
    eval_entries = [e for e in log_history if "eval_loss" in e]
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"train_loss_log": train_entries, "eval_loss_log": eval_entries}, f, ensure_ascii=False, indent=2)
    if eval_entries:
        last_eval = eval_entries[-1]
        print(f"[train_planner] 마지막 eval_loss: {last_eval.get('eval_loss')} "
              f"(epoch {last_eval.get('epoch')}) -> 전체 로그: {log_path}")
    else:
        print(f"[train_planner] 경고: eval_loss 기록이 없음(비정상) - train loss만 {log_path}에 저장됨")

    if args.gen_check:
        step_tag = trainer.state.global_step
        # (2026-08 추가) Trainer의 optimizer(Adam state 등)가 아직 메모리에 남아있는 채로
        # gen_check을 돌리면 그만큼 여유가 줄어든다 - 학습은 끝났으니 옵티마이저는 더 필요
        # 없어서 명시적으로 비운다. 학습 자체를 계속 이어갈 게 아니라 여기서 스크립트가
        # 끝나는 경로라 안전함(--resume_from_checkpoint는 디스크에 저장된 optimizer.pt를
        # 쓰지 이 트레이너 인스턴스를 재사용하지 않음).
        del trainer.optimizer, trainer.lr_scheduler
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = run_gen_check(
            model, processor, args.val_jsonl,
            limit=args.gen_check_limit, max_new_tokens=args.gen_check_max_new_tokens,
        )
        out_path = os.path.join(args.output_dir, f"gen_check_step{step_tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[train_planner] gen_check 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
