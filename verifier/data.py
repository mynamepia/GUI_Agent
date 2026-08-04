"""
verifier/data.py

verifier 학습/추론 공용: (image, instruction, point_px) -> "별 찍은 이미지 + 프롬프트"
messages를 빌드하는 헬퍼. region_focus.judge_inference()가 후보점을 판정할 때 쓰는
것과 동일한 입력 표현(핑크 별 오버레이)을 그대로 재사용한다 - 학습 때 본 입력 분포와
실제 judge_inference() 추론 때 들어오는 입력 분포가 다르면 안 되기 때문.

plot_points_on_image()는 region_focus.py에 이미 있는 걸 그대로 가져다 쓴다
(PYTHONPATH=..로 상위 모듈 재사용, hypo1과 동일한 패턴 - 중복 구현 금지).
"""

from PIL import Image

from region_focus import plot_points_on_image

# region_focus.judge_inference()의 별 마커 스타일과 동일하게 맞춤 (색/사이즈).
_STAR_COLOR = (255, 0, 255, 128)
_STAR_SIZE = 12

VERIFIER_PROMPT_TEMPLATE = (
    'Instruction: "{instruction}"\n'
    "A pink star marks a candidate click point on the screenshot. Determine whether "
    "the star precisely marks the correct UI element for the instruction. The star "
    "may only partially cover the target, and that still counts as correct."
)


def build_verifier_image(image, point_px):
    """image(경로 str 또는 PIL.Image) 위에 point_px 위치에 별을 찍은 이미지를 반환."""
    pil_image = Image.open(image) if isinstance(image, str) else image
    pil_image = pil_image.convert("RGB")
    return plot_points_on_image(
        pil_image, [point_px], colors=[_STAR_COLOR], markers=["star"], sizes=[_STAR_SIZE],
    )


def build_verifier_messages(image, instruction, point_px):
    """QwenVerifier.forward()에 바로 넣을 수 있는 messages(챗 템플릿 포맷)를 빌드."""
    highlighted = build_verifier_image(image, point_px)
    prompt_text = VERIFIER_PROMPT_TEMPLATE.format(instruction=instruction)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": highlighted},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
