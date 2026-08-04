"""
verifier/model.py

Verifier(judge) 전용 모델 - grounding LoRA(train.py, checkpoint-4130)와는 완전히
분리된 새 LoRA 어댑터 + explicit binary classification head를 base Qwen2.5-VL 위에
얹은 wrapper.

[문헌 근거] (memory: reference_separate_verifier_literature 참고)
"Why is Your Language Model a Poor Implicit Reward Model?"이 지적하는 바에 따르면
implicit reward model(="YES"/"NO" 텍스트 토큰의 로짓을 그대로 판단 신호로 쓰는 방식,
지금까지의 judge_inference()가 이 방식)은 표면적 토큰 단서에 의존해서 distribution
shift(예: 학습 안 해본 플랫폼/해상도)에 일반화가 약하다. explicit reward model(=
별도 스칼라 classification head)이 같은 데이터/loss로도 더 잘 일반화한다고 보고됨.
가설2에서 관찰한 "judge logit margin 양극단분포"(project_hypo2_judge_logitmargin_bimodal
메모리)도 이 implicit reward 한계와 부합하는 증거였음 - 그래서 여기서는 텍스트 생성
대신 마지막 hidden state를 풀링해서 linear head 하나로 binary logit을 직접 뽑는다.

[grounding LoRA와의 관계]
base 모델은 공유하고 LoRA 어댑터만 분리했기 때문에, 추론 시 하나의 3B 백본만 로드해둔
채로 peft의 멀티 어댑터 스왑(model.set_adapter("grounding") <-> ("verifier"))으로
RegionFocus의 Step1/3/4(grounding)와 Step2/5(verifier) 사이를 오갈 수 있다 - 3B를
두 번 로드할 필요가 없음. (이 스왑 통합은 verifier 학습이 끝난 뒤 region_focus.py
쪽에 별도로 붙일 예정 - 이 파일은 학습/단독 추론까지만 담당.)

[입력 표현] verifier/data.py의 build_verifier_messages() 참고 - judge_inference()와
동일하게 후보점에 별을 찍은 이미지 + instruction 텍스트를 입력으로 받는다.
"""

import os

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _hidden_size(base_model):
    cfg = base_model.config
    # Qwen2.5-VL은 config.text_config.hidden_size에 LLM 쪽 hidden size가 들어있다
    # (transformers 버전에 따라 최상위 config.hidden_size로도 노출될 수 있어 폴백을 둠).
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size
    return cfg.hidden_size


class QwenVerifier(nn.Module):
    """
    base Qwen2.5-VL(+ 새 LoRA) + binary classification head.
    forward()는 (input_ids, attention_mask, pixel_values, image_grid_thw) 배치를
    받아서 "이 후보점이 정답일 확률"의 logit(sigmoid 적용 전)을 반환한다.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules=DEFAULT_LORA_TARGET_MODULES,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
            # 이 wrapper는 언어모델 head(다음 토큰 생성)를 쓰지 않고 hidden state만
            # 뽑아서 별도 classification head에 넣으므로 CAUSAL_LM이 아니라
            # FEATURE_EXTRACTION으로 지정 (peft가 generate() 관련 유틸을 강제하지 않게).
            task_type="FEATURE_EXTRACTION",
        )
        self.backbone = get_peft_model(base, lora_config)
        self.hidden_size = _hidden_size(base)
        # classification head는 안정성을 위해 항상 fp32로 유지 (LoRA 쪽은 bf16이어도 무방).
        self.head = nn.Linear(self.hidden_size, 1).to(dtype=torch.float32)

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (B, T, H)

        # 마지막 non-pad 토큰 위치를 풀링한다. processor의 기본 padding은 오른쪽
        # (right-padding)이라 attention_mask.sum(dim=1)-1이 곧 그 시퀀스의 마지막
        # 실제 토큰 인덱스가 된다 - 왼쪽 패딩으로 바뀌면 이 계산도 같이 고쳐야 함.
        seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_idx, seq_lens]  # (B, H)

        logit = self.head(pooled.float()).squeeze(-1)  # (B,)
        return logit

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.backbone.save_pretrained(save_dir)
        torch.save(self.head.state_dict(), os.path.join(save_dir, "head.pt"))

    @classmethod
    def from_pretrained(cls, save_dir, model_id: str = MODEL_ID, dtype=torch.bfloat16):
        """
        train_verifier.py가 save_pretrained()로 저장한 (LoRA 어댑터 + head.pt)를
        읽어서 추론 전용 인스턴스를 만든다. is_trainable=False로 로드 (학습 재개용이
        아니라 judge_inference() 등에서 스코어링만 할 때 사용).
        """
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.backbone = PeftModel.from_pretrained(base, save_dir, is_trainable=False)
        obj.hidden_size = _hidden_size(base)
        obj.head = nn.Linear(obj.hidden_size, 1).to(dtype=torch.float32)
        obj.head.load_state_dict(torch.load(os.path.join(save_dir, "head.pt"), map_location="cpu"))
        return obj

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        """추론 전용 헬퍼: sigmoid까지 적용한 "정답일 확률"을 반환."""
        self.eval()
        logit = self.forward(input_ids, attention_mask, pixel_values, image_grid_thw)
        return torch.sigmoid(logit)
