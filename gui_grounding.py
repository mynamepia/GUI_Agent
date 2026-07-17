"""
gui_grounding.py

qwen_agent 라이브러리 없이, GUI grounding에 필요한 function-calling 파이프라인을
직접 구현한 모듈.

베이스라인(Qwen25VLModel.ground())이 qwen_agent에 의존하던 세 부분을 대체한다:

    1) ComputerUse 액션 스키마       -> ComputerUseTool
    2) NousFnCallPrompt(시스템 프롬프트) -> build_fncall_system_prompt()
    3) <tool_call> 파싱               -> parse_tool_call()

그리고 이 세 조각을 qwen.py의 QwenVLModel과 엮어서, 로컬 모델로 좌표를
얻어내는 ground() 함수까지 제공한다.

주의:
    ComputerUseTool의 액션/파라미터 이름은 공개된 Qwen-Agent computer_use 툴 스펙을
    참고해 재구성한 것이라, 원본 qwen_agent 소스나 실제로 학습에 쓰인 액션 스페이스와
    필드명이 100% 동일하다는 보장은 없다. grounding에서 실제로 쓰이는 것은
    "coordinate" 인자 하나뿐이라 그 부분은 정확히 맞춰뒀고, 나머지 액션(type/key/scroll 등)은
    이후 full agent loop를 만들 때 네가 쓰는 벤치마크(OSWorld, ScreenSpot 등) 포맷에
    맞춰 필드명을 검증/수정하는 걸 추천한다.

필요 패키지: qwen.py와 동일 (torch, transformers, qwen-vl-utils, pillow)
"""

import json
import re
import time

from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

from qwen import QwenVLModel, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS


# ---------------------------------------------------------------------------
# 1) GUI 액션 스키마 (qwen_agent.ComputerUse 대체)
# ---------------------------------------------------------------------------
class ComputerUseTool:
    """
    GUI 에이전트가 쓸 수 있는 '컴퓨터 조작' 액션들을 function-calling 스키마(dict)로
    정의하는 클래스. 모델은 이 스키마를 시스템 프롬프트로 받고, 그 스키마 안의
    함수 하나를 호출하는 형태(JSON)로 응답하도록 유도된다.

    display_width_px / display_height_px는 "모델이 실제로 보는 이미지"의 픽셀
    크기여야 한다 (원본 스크린샷 크기가 아니라, smart_resize를 거친 크기).
    좌표는 항상 이 크기를 기준으로 해석된다.
    """

    name = "computer_use"

    def __init__(self, display_width_px: int, display_height_px: int):
        self.display_width_px = display_width_px
        self.display_height_px = display_height_px

    @property
    def function(self) -> dict:
        """function-calling 스키마 (JSON Schema 형식의 dict)."""
        return {
            "name": self.name,
            "description": (
                "Use a mouse and keyboard to interact with a GUI screenshot.\n"
                f"* The screenshot's resolution is {self.display_width_px}x{self.display_height_px} pixels.\n"
                "* Coordinates are given in pixels, measured from the top-left corner "
                "of the screenshot (0,0)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "left_click",
                            "double_click",
                            "right_click",
                            "left_click_drag",
                            "mouse_move",
                            "type",
                            "key",
                            "scroll",
                            "wait",
                            "terminate",
                        ],
                        "description": "수행할 GUI 액션.",
                    },
                    "coordinate": {
                        "type": "array",
                        "description": (
                            "[x, y] 픽셀 좌표. left_click, double_click, right_click, "
                            "left_click_drag, mouse_move, scroll 액션에 필요."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": "입력할 텍스트, 또는 누를 키 이름. `type`/`key` 액션에 필요.",
                    },
                    "time": {
                        "type": "number",
                        "description": "대기할 시간(초). `wait` 액션에 필요.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["success", "failure"],
                        "description": "작업 종료 상태. `terminate` 액션에 필요.",
                    },
                },
                "required": ["action"],
            },
        }


# ---------------------------------------------------------------------------
# 2) function-calling 시스템 프롬프트 빌더 (qwen_agent.NousFnCallPrompt 대체)
# ---------------------------------------------------------------------------
def build_fncall_system_prompt(
    functions: list, base_system_text: str = "You are a helpful assistant."
) -> str:
    """
    Qwen2.5 계열이 파인튜닝 때 학습한 Hermes/Nous 스타일 function-calling 포맷으로
    시스템 프롬프트 텍스트를 만든다.

    functions: [tool.function, ...] 형태의 dict 리스트 (ComputerUseTool.function 등).

    모델은 이 포맷을 보면 아래와 같은 형태로 응답하도록 학습되어 있다:

        <tool_call>
        {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [123, 45]}}
        </tool_call>
    """
    tools_json = "\n".join(
        json.dumps({"type": "function", "function": fn}, ensure_ascii=False)
        for fn in functions
    )
    return (
        f"{base_system_text}\n\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        f"{tools_json}\n"
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def build_grounding_messages(
    instruction: str, image, tool: ComputerUseTool, prompt_text: str | None = None
) -> list:
    """
    ground()에서 쓰는 messages(Qwen 챗 템플릿 포맷)를 조립.
    image는 QwenVLModel.generate()가 받아들이는 형태 그대로(경로 str 또는 PIL.Image)면 된다.

    prompt_text: user 턴에 넣을 지시문. None이면 기본 grounding 문구를 쓰고,
    RegionFocus처럼 다른 문구가 필요한 곳(region_focus, next_action_regionfocus 등)에서는
    직접 넘겨서 재사용한다.
    """
    system_text = build_fncall_system_prompt(functions=[tool.function])

    if prompt_text is None:
        prompt_text = (
            f'Output the most relevant point in the image corresponding to '
            f'the instruction "{instruction}" with grounding, by calling the '
            f'computer_use function with a left_click action.'
        )

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 3) <tool_call> 파서 (베이스라인의 fragile split() 방식 대체)
# ---------------------------------------------------------------------------
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(response_text: str) -> dict | None:
    """
    모델 응답 텍스트에서 <tool_call>...</tool_call> 안의 JSON을 뽑아 dict로 반환.
    - 여러 개면 마지막 것을 사용 (베이스라인과 동일한 동작).
    - 태그가 없거나 JSON 파싱에 실패하면 None.
    """
    matches = _TOOL_CALL_RE.findall(response_text)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# 위 세 조각을 엮은 로컬 grounding 함수
# (베이스라인 Qwen25VLModel.ground()의 로컬 모델 버전)
# ---------------------------------------------------------------------------
def ground(
    qwen_model: QwenVLModel,
    instruction: str,
    image,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_new_tokens: int = 128,
) -> dict:
    """
    GUI 스크린샷 위에서 instruction에 해당하는 지점을 찾는다.

    Returns:
        {
            "result": "positive" | "wrong_format",
            "point": [x_norm, y_norm] | None,  # 원본 이미지 기준 0~1 정규화 좌표
            "raw_response": str,
        }
    """
    pil_image = Image.open(image) if isinstance(image, str) else image

    # (a) smart_resize로 "모델이 실제로 보게 될 크기"를 우리가 직접 고정한다.
    #     이 크기를 알아야 모델이 뱉는 픽셀 좌표를 원본 이미지 좌표로 되돌릴 수 있다.
    resized_height, resized_width = smart_resize(
        pil_image.height, pil_image.width,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    resized_image = pil_image.resize((resized_width, resized_height))

    # (b) 액션 스키마 + function-calling 시스템 프롬프트로 messages 구성
    tool = ComputerUseTool(display_width_px=resized_width, display_height_px=resized_height)
    messages = build_grounding_messages(instruction, resized_image, tool)

    # (c) processor에 그대로 넘긴다. processor 내부에서도 (min_pixels, max_pixels)
    #     기준으로 smart_resize를 다시 돌리는데, smart_resize는 이미 28의 배수이고
    #     면적이 [min_pixels, max_pixels] 안에 있는 크기를 넣으면 그대로 반환하는
    #     멱등 함수라서 여기서 이미지가 또 리사이즈되지 않는다.
    #     (구버전 코드는 processor.image_processor.min_pixels/max_pixels를 직접
    #     덮어쓰려 했는데, transformers 버전에 따라 그 속성이 없어서 AttributeError가
    #     났음 - 애초에 불필요한 작업이라 제거함)
    #
    #     주의: 이 함수를 호출할 때 넘기는 min_pixels/max_pixels는 QwenVLModel을
    #     만들 때 썼던 값과 반드시 같아야 한다. 다르면 processor가 우리 계산과
    #     다른 크기로 다시 리사이즈해버려서 좌표가 어긋난다.
    _t0 = time.time()
    raw_response = qwen_model.generate(messages, max_new_tokens=max_new_tokens)
    print(f"[ground] generate() 완료 - {time.time() - _t0:.1f}초")

    # (d) tool_call 파싱 + 좌표 정규화
    tool_call = parse_tool_call(raw_response)
    if tool_call is None:
        return {"result": "wrong_format", "point": None, "raw_response": raw_response}

    try:
        x, y = tool_call["arguments"]["coordinate"]
        point_norm = [x / resized_width, y / resized_height]
        return {"result": "positive", "point": point_norm, "raw_response": raw_response}
    except (KeyError, TypeError, ValueError):
        return {"result": "wrong_format", "point": None, "raw_response": raw_response}


if __name__ == "__main__":
    # 간단한 동작 확인용
    model = QwenVLModel(load_in_8bit=True)
    result = ground(model, "로그인 버튼을 클릭", "path/to/screenshot.png")
    print(result)