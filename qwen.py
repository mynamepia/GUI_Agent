"""
qwen.py

Qwen2.5-VL-3B-Instruct 모델 로더 (+ GUI grounding 에이전트용 공용 모델 초기화 모듈).

작은 VLM(Vision-Language Model)을 백본으로 사용하는 에이전트 프로젝트의
모델 초기화/추론 모듈로 사용하기 위한 코드. train.py(LoRA SFT)도
load_model_and_processor()를 그대로 가져다 써서, 두 파일이 항상 같은
모델 클래스/설정으로 모델을 로드하도록 맞춰 놨다.

필요 패키지:
    pip install torch torchvision transformers accelerate qwen-vl-utils pillow
    (load_in_8bit/load_in_4bit를 쓰려면 추가로: pip install bitsandbytes)
"""

import torch
torch.set_num_threads(12)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Qwen2-VL 계열은 이미지 한 장이 patch(28x28) 단위로 쪼개져서 비주얼 토큰이 되는데,
# 상한을 안 걸어두면 고해상도 스크린샷 한 장이 수천 토큰이 될 수 있음.
# CPU/저사양 환경(미니PC 등)에서는 min_pixels/max_pixels로 반드시 상한을 걸어두는 게 안전하다.
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 384 * 28 * 28 # 640 * 28 * 28


def load_model_and_processor(
    model_id: str = MODEL_ID,
    device_map: str | None = None,
    dtype=torch.bfloat16,
    min_pixels: int | None = DEFAULT_MIN_PIXELS,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
    low_cpu_mem_usage: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    offload_folder: str | None = None,
):
    """
    모델 + 프로세서를 로드해서 반환하는 공용 함수.
    QwenVLModel(추론용)과 train.py(LoRA 학습용)이 동일한 로딩 로직/모델 클래스를
    공유하도록 여기에 분리해뒀다.
    (이전 버전 train.py는 Qwen2VLForConditionalGeneration을 import했는데, 실제
    체크포인트인 Qwen2.5-VL과 클래스가 어긋나서 로딩이 깨지는 버그가 있었음 - 여기서 통일)

    [미니 PC(CPU-only) 등 저사양 환경용 메모리 절약 옵션]
    - low_cpu_mem_usage: from_pretrained가 가중치를 임시로 두 배 복사하지 않고 바로 목표
      위치에 로드하게 함. 부작용 없이 항상 켜둬도 되는 옵션이라 기본값을 True로 둠.
    - load_in_8bit / load_in_4bit: bitsandbytes 양자화로 모델 크기를 1/2~1/4로 줄임.
      다만 bitsandbytes는 원래 GPU 최적화 라이브러리라 CPU 전용 환경에서는 효과가 없거나
      아예 동작하지 않을 수 있음 - GPU가 있는 환경에서 VRAM을 더 아끼고 싶을 때 위주로 사용.
    - offload_folder: device_map에 "disk"가 포함될 때, 다 못 올린 레이어를 디스크로 내려서
      최후 수단으로 메모리 부족을 회피함(대신 디스크 I/O 때문에 많이 느려짐).
    """
    use_cuda = torch.cuda.is_available()
    actual_dtype = dtype if use_cuda else torch.float32

    quantization_config = None
    if load_in_8bit or load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

    from_pretrained_kwargs = dict(
        torch_dtype=actual_dtype,
        device_map=device_map if use_cuda else None,
        # SDPA(Scaled Dot Product Attention)를 명시적으로 지정 - torch 내장이라 별도 설치 없이
        # attention 연산의 VRAM 사용량을 줄여준다 (Qwen-GUI-3B 논문도 FlashAttention/SDPA로
        # 메모리 효율화를 함). 지정 안 하면 transformers 버전에 따라 더 무거운 eager로 돌 수 있음.
        attn_implementation="sdpa" if use_cuda else None,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    if quantization_config is not None:
        from_pretrained_kwargs["quantization_config"] = quantization_config
    if offload_folder is not None:
        from_pretrained_kwargs["offload_folder"] = offload_folder

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        **from_pretrained_kwargs,
    )

    processor_kwargs = {}
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels

    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    # Qwen 계열 토크나이저는 보통 pad_token이 이미 있지만, 혹시 없는 체크포인트를 쓸 경우 대비.
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


def generate_text(model, processor, messages: list, max_new_tokens: int = 512) -> str:
    """
    이미 로드된 model/processor로 messages(Qwen 챗 템플릿 포맷)에 대한 답변 텍스트 하나를 생성.
    QwenVLModel.generate()와 evaluation.py의 배치 추론(test.py, train.py 사후 평가)에서
    똑같은 로직을 공유하려고 분리해뒀다. model은 base 모델이든 PeftModel(LoRA 어댑터를 얹은
    상태)이든 상관없이 그대로 동작한다 (PeftModel도 .generate()를 그대로 지원).
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


class QwenVLModel:
    """
    Qwen2.5-VL-3B 모델을 감싸는 래퍼 클래스.
    에이전트에서 "이 모델로 이미지/텍스트를 보고 답을 생성한다"는 인터페이스로 쓰기 위함.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        dtype=torch.bfloat16,
        min_pixels: int | None = DEFAULT_MIN_PIXELS,
        max_pixels: int | None = DEFAULT_MAX_PIXELS,
        low_cpu_mem_usage: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        offload_folder: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[qwen.py] Loading {model_id} on {self.device} ...")

        self.model, self.processor = load_model_and_processor(
            model_id=model_id,
            device_map="auto" if self.device == "cuda" else None,
            dtype=dtype,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            low_cpu_mem_usage=low_cpu_mem_usage,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            offload_folder=offload_folder,
        )
        if self.device != "cuda":
            self.model.to(self.device)

        print("[qwen.py] Model loaded.")

    def generate(self, messages: list, max_new_tokens: int = 512) -> str:
        """
        messages: Qwen 채팅 템플릿 형식의 메시지 리스트.
        예시:
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "path/to/image.jpg"},
                        {"type": "text", "text": "이 이미지에 뭐가 있어?"},
                    ],
                }
            ]
        """
        return generate_text(self.model, self.processor, messages, max_new_tokens=max_new_tokens)

    def chat(self, text: str, image_path: str | None = None, max_new_tokens: int = 512) -> str:
        """텍스트(+선택적 이미지)를 받아 답변을 반환하는 간단한 헬퍼."""
        content = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]
        return self.generate(messages, max_new_tokens=max_new_tokens)


if __name__ == "__main__":
    # 간단한 동작 확인용 (텍스트만 입력하는 예시)
    model = QwenVLModel(load_in_8bit=True)
    response = model.chat("너는 어떤 역할을 하는 에이전트야? 한 문장으로 소개해줘.")
    print("응답:", response)
