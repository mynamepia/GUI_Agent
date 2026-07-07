"""
model.py

Qwen2.5-VL-3B-Instruct 모델 로더.
작은 VLM(Vision-Language Model)을 백본으로 사용하는 에이전트 프로젝트의
모델 초기화/추론 모듈로 사용하기 위한 코드.

필요 패키지:
    pip install torch torchvision transformers accelerate qwen-vl-utils pillow
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class QwenVLModel:
    """
    Qwen2.5-VL-3B 모델을 감싸는 래퍼 클래스.
    에이전트에서 "이 모델로 이미지/텍스트를 보고 답을 생성한다"는 인터페이스로 쓰기 위함.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None, dtype=torch.bfloat16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[model.py] Loading {model_id} on {self.device} ...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        print("[model.py] Model loaded.")

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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

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
    model = QwenVLModel()
    response = model.chat("너는 어떤 역할을 하는 에이전트야? 한 문장으로 소개해줘.")
    print("응답:", response)