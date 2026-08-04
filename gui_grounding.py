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

[2026-08 수정] ComputerUseTool/build_grounding_messages/parse_tool_call(Hermes
tool-calling 포맷)은 이 LoRA(checkpoint-4130, train.py 학습)가 실제로 학습받은
포맷이 아니라는 게 확인됐다. train.py/evaluation.py는 system 메시지 없이
coord_utils.PROMPT_TEMPLATE 하나로 묻고 "(x,y)"(0~1000 정규화) 텍스트로 답을
받도록 학습/평가하는데, 여기서는 그와 다른 tool-call JSON 포맷으로 물어보고
있었다 - 즉 LoRA가 한 번도 본 적 없는 입력 구조로 grounding을 시킨 셈이었다.
그래서 실제로 이 LoRA에게 좌표를 묻는 ground()는 build_point_prompt_messages()
+ parse_point_from_text() 조합(학습 포맷 그대로)으로 바꿨다. ComputerUseTool 등
기존 tool-calling 유틸리티는 지우지 않고 남겨뒀다 - 나중에 tool-calling을 실제로
지원하는 다른 백본으로 바꿀 일이 생기면 그대로 쓸 수 있지만, 지금 이 LoRA 기반
grounding 파이프라인에서는 쓰지 않는다.
"""

import json
import os
import re
import time

from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

from qwen import QwenVLModel, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS
from coord_utils import PROMPT_TEMPLATE, parse_point_from_text


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

    def __init__(self, display_width_px: int, display_height_px: int, coord_mode: str = "pixel"):
        """
        coord_mode: "pixel"(기본, 원본 tool-call 방식) | "norm1000"(Step4 ablation
        옵션C용 - tool-call schema/파싱 구조는 그대로 두고 coordinate 값의 스케일만
        0~1000 정규화로 바꾼 하이브리드. region_focus.py의
        next_action_regionfocus_toolcall_norm1000()에서 씀.
        """
        self.display_width_px = display_width_px
        self.display_height_px = display_height_px
        assert coord_mode in ("pixel", "norm1000"), f"unknown coord_mode: {coord_mode}"
        self.coord_mode = coord_mode

    @property
    def function(self) -> dict:
        """function-calling 스키마 (JSON Schema 형식의 dict)."""
        if self.coord_mode == "norm1000":
            description = (
                "Use a mouse and keyboard to interact with a GUI screenshot.\n"
                "* Coordinates are given as integers from 0 to 1000, representing the "
                "relative position in the screenshot (0,0 = top-left corner, 1000,1000 = "
                "bottom-right corner), independent of the screenshot's actual pixel resolution."
            )
            coordinate_desc = (
                "[x, y], where x and y are integers from 0 to 1000 representing the relative "
                "position in the screenshot (0,0 = top-left, 1000,1000 = bottom-right), NOT "
                "pixels. Required for left_click, double_click, right_click, left_click_drag, "
                "mouse_move, scroll."
            )
        else:
            description = (
                "Use a mouse and keyboard to interact with a GUI screenshot.\n"
                f"* The screenshot's resolution is {self.display_width_px}x{self.display_height_px} pixels.\n"
                "* Coordinates are given in pixels, measured from the top-left corner "
                "of the screenshot (0,0)."
            )
            coordinate_desc = (
                "[x, y] 픽셀 좌표. left_click, double_click, right_click, "
                "left_click_drag, mouse_move, scroll 액션에 필요."
            )
        return {
            "name": self.name,
            "description": description,
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
                        "description": coordinate_desc,
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
# 2-b) 학습 포맷 그대로 묻는 프롬프트 빌더
#      (이 LoRA에게 실제로 좌표를 물어보는 모든 호출은 이걸 써야 한다)
# ---------------------------------------------------------------------------
def build_point_prompt_messages(instruction: str, image) -> list:
    """
    train.py가 LoRA를 학습시킬 때 쓴 것과 완전히 동일한 messages를 만든다
    (evaluation.py의 run_generation_eval과 100% 동일한 구조):
        - system 메시지 없음
        - user 메시지 하나: 이미지 + PROMPT_TEMPLATE.format(instruction=instruction)
    모델은 "(x,y)"(0~1000 정규화 정수) 텍스트로 답하도록 학습되어 있다
    (coord_utils.parse_point_from_text로 파싱).

    ground(), region_focus_WJ.py의 region_focus()/next_action_regionfocus() 등
    이 LoRA에게 실제로 좌표를 물어보는 모든 호출은 build_grounding_messages()
    (Hermes tool-calling 포맷) 대신 반드시 이 함수를 써야 한다 - LoRA는 tool-call
    포맷을 학습에서 한 번도 보지 못했다.
    """
    prompt_text = PROMPT_TEMPLATE.format(instruction=instruction)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
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
# --debug_text 공용 헬퍼 (gui_grounding.py, region_focus.py가 같이 씀)
# ---------------------------------------------------------------------------
def dump_prompt_debug(messages, response, task_id=None, step_name="generate", index=None, extra=""):
    """
    --debug_text 전용: 실제로 모델에 들어간 프롬프트(messages의 텍스트 파트, 시스템+유저)와
    모델 응답 원문을 ./debug/<task_id>/prompt_<step_name>[_index].txt 파일로 남긴다.

    실제 프롬프팅이 의도대로 들어갔는지(예: judge_inference가 어떤 문구로 물어봤고
    모델이 정확히 뭐라고 답했는지) 눈으로 직접 확인하기 위한 용도.
    이미지 파트는 여기서 남기지 않음 - 이미지 저장은 --debug_image 쪽 책임
    (region_focus.py의 calculate_crop_region/crop_and_upsample/judge_inference 등).

    extra: 그 호출의 "해석 결과"(예: judge의 최종 YES/NO 판정, aggregation의 선택 번호)를
    같이 남기고 싶을 때 쓰는 선택적 문자열.
    """
    debug_dir = f"./debug/{task_id}" if task_id else "./debug"
    os.makedirs(debug_dir, exist_ok=True)
    safe_name = re.sub(r"[^0-9a-zA-Z_.=-]", "_", str(step_name))
    fname = f"prompt_{safe_name}_{index}.txt" if index is not None else f"prompt_{safe_name}.txt"

    lines = []
    for m in messages:
        role = m.get("role", "?")
        texts = [
            c.get("text", "")
            for c in m.get("content", [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        for t in texts:
            lines.append(f"--- [{role}] ---\n{t}")

    with open(os.path.join(debug_dir, fname), "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
        f.write(f"\n\n--- [response] ---\n{response}\n")
        if extra:
            f.write(f"\n--- [extra] ---\n{extra}\n")


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
    temperature: float = 0.0,
    debug_text: bool = False,
    task_id=None,
) -> dict:
    """
    GUI 스크린샷 위에서 instruction에 해당하는 지점을 찾는다.

    [2026-08 수정] 프롬프트/파싱/좌표변환을 train.py가 LoRA를 학습시킬 때 쓴 포맷
    (evaluation.py + coord_utils.py)과 완전히 동일하게 맞췄다: system 메시지 없이
    build_point_prompt_messages()(PROMPT_TEMPLATE)로 묻고, parse_point_from_text()로
    "(x,y)"(0~1000 정규화) 텍스트를 파싱한다. 예전 버전은 Hermes 스타일 tool-calling
    시스템 프롬프트(ComputerUseTool/build_grounding_messages/parse_tool_call)를 썼는데,
    LoRA는 이 포맷을 학습에서 한 번도 보지 못했다 - grounding 정확도 저하의 원인 중
    하나로 지목되어 여기서 통일했다.

    Returns:
        {
            "result": "positive" | "wrong_format",
            "point": [x_norm, y_norm] | None,  # 원본 이미지 기준 0~1 정규화 좌표
            "raw_response": str,
        }
    """
    pil_image = Image.open(image) if isinstance(image, str) else image

    # smart_resize는 여전히 필요하다 - 모델에 넣을 이미지의 토큰 예산(min/max_pixels)을
    # 우리가 직접 통제하기 위함일 뿐, 아래 좌표 변환에는 이제 이 크기가 전혀 쓰이지
    # 않는다. 모델이 답하는 "(x,y)"는 0~1000 상대좌표라 해상도 무관이라서, nx/1000,
    # ny/1000이 곧바로 원본 이미지 기준 0~1 정규화 좌표가 된다
    # (coord_utils.norm1000_to_point가 하는 일과 동치 - w,h가 그대로 나눠지고 곱해져서
    # 사라짐).
    resized_height, resized_width = smart_resize(
        pil_image.height, pil_image.width,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    resized_image = pil_image.resize((resized_width, resized_height))

    # 학습 포맷 그대로: system 메시지 없이 PROMPT_TEMPLATE 하나로 질의
    messages = build_point_prompt_messages(instruction, resized_image)

    # processor에 그대로 넘긴다. processor 내부에서도 (min_pixels, max_pixels) 기준으로
    # smart_resize를 다시 돌리는데, smart_resize는 이미 28의 배수이고 면적이
    # [min_pixels, max_pixels] 안에 있는 크기를 넣으면 그대로 반환하는 멱등 함수라서
    # 여기서 이미지가 또 리사이즈되지 않는다.
    _t0 = time.time()
    raw_response = qwen_model.generate(
        messages, max_new_tokens=max_new_tokens, temperature=temperature,
    )
    print(f"[ground] generate() 완료 - {time.time() - _t0:.1f}초")

    if debug_text:
        dump_prompt_debug(messages, raw_response, task_id=task_id, step_name="ground")

    # "(x,y)"(0~1000 정규화) 파싱
    norm_point = parse_point_from_text(raw_response)
    if norm_point is None:
        return {"result": "wrong_format", "point": None, "raw_response": raw_response}

    nx, ny = norm_point
    # coord_utils.parse_point_from_text/norm1000_to_point는 범위를 클램프하지 않는다
    # (evaluation.py도 그대로 둬서, 범위를 벗어나면 bbox hit-test에서 자연히 miss
    # 처리됨). 다만 이 point는 RegionFocus의 crop 계산 등 하류 로직에서 그대로 픽셀
    # 연산에 쓰이므로, 여기서만 방어적으로 [0,1]로 클램프해둔다 - 학습/평가 스펙의
    # 일부가 아니라 이 함수를 쓰는 하류 코드를 위한 안전장치일 뿐이다.
    x_norm = max(0.0, min(1.0, nx / 1000))
    y_norm = max(0.0, min(1.0, ny / 1000))
    return {"result": "positive", "point": [x_norm, y_norm], "raw_response": raw_response}


def ground_toolcall_norm1000(
    qwen_model: QwenVLModel,
    instruction: str,
    image,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    debug_text: bool = False,
    task_id=None,
) -> dict:
    """
    [Step1 ablation] ground()와 역할/반환 스키마는 완전히 동일(원본 이미지 기준 0~1
    정규화 좌표를 돌려줌)하지만, 좌표를 묻는 방식만 old RegionFocus가 쓰던
    tool-call(computer_use function-calling) 스키마로 바꾸고, coordinate 값 자체는
    old(raw pixel)가 아니라 0~1000 정규화로 받는다.

    region_focus.py의 next_action_regionfocus_toolcall_norm1000()(Step4 ablation
    옵션C)과 짝을 이루는 Step1용 버전 - Step4만 old 스키마로 바꿨을 때는 오히려 결과가
    baseline보다 나빠졌는데(28.07%(B) 대비 25.71%), Step1까지 같이 old 스키마로
    바꾸면 뭔가 달라지는지 보기 위한 실험용 함수. (단, old의 Step1은 do_sample을
    명시 안 해서 비결정적 샘플링이었는데 이 함수는 여전히 temperature<=0이면
    do_sample=False로 고정이라, "스키마+좌표 스케일"만 old를 재현하고 "샘플링
    비결정성"까지 재현하진 않는다 - 완전한 old 재현은 아님.)

    Returns: ground()와 동일한 스키마.
        {
            "result": "positive" | "wrong_format",
            "point": [x_norm, y_norm] | None,
            "raw_response": str,
        }
    """
    pil_image = Image.open(image) if isinstance(image, str) else image

    resized_height, resized_width = smart_resize(
        pil_image.height, pil_image.width,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    resized_image = pil_image.resize((resized_width, resized_height))

    tool = ComputerUseTool(resized_width, resized_height, coord_mode="norm1000")
    prompt_text = (
        f'Output the most relevant point in the image corresponding to '
        f'the instruction "{instruction}" with grounding, by calling the '
        f'computer_use function with a left_click action.'
    )
    messages = build_grounding_messages(instruction, resized_image, tool, prompt_text=prompt_text)

    _t0 = time.time()
    raw_response = qwen_model.generate(
        messages, max_new_tokens=max_new_tokens, temperature=temperature,
    )
    print(f"[ground_toolcall_norm1000] generate() 완료 - {time.time() - _t0:.1f}초")

    if debug_text:
        dump_prompt_debug(messages, raw_response, task_id=task_id, step_name="ground_toolcall_norm1000")

    tool_call = parse_tool_call(raw_response)
    if not tool_call:
        return {"result": "wrong_format", "point": None, "raw_response": raw_response}
    coord = (tool_call.get("arguments") or {}).get("coordinate")
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return {"result": "wrong_format", "point": None, "raw_response": raw_response}
    try:
        nx, ny = float(coord[0]), float(coord[1])
    except (TypeError, ValueError):
        return {"result": "wrong_format", "point": None, "raw_response": raw_response}

    x_norm = max(0.0, min(1.0, nx / 1000))
    y_norm = max(0.0, min(1.0, ny / 1000))
    return {"result": "positive", "point": [x_norm, y_norm], "raw_response": raw_response}


def ground_toolcall_pixel(
    qwen_model: QwenVLModel,
    instruction: str,
    image,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    debug_text: bool = False,
    task_id=None,
) -> dict:
    """
    [Step1 - old 프롬프트 그대로] bak/gui_grounding.py의 원래(수정 전) ground()를
    그대로 이식한 버전. tool-call 스키마도, coordinate description("픽셀 좌표")도,
    파싱 후 좌표 변환(0~1000 변환 없이 raw pixel을 그대로 resized_width/height로
    나눔)도 old와 100% 동일하다 - 유일한 차이는 qwen_model.generate()가 이제
    qwen.py의 고정된 do_sample=False를 타서 결정적이라는 점뿐(old는 이 부분이
    버그로 비결정적이었음).

    "old 스키마+old 좌표표현" 조합을 재현해서, do_sample 비결정성만 뺀 순수 old
    프롬프트가 37.81%에 가까운 RegionFocus 우위를 재현하는지 보기 위한 컨트롤 실험용.

    Returns: ground()와 동일한 스키마.
    """
    pil_image = Image.open(image) if isinstance(image, str) else image

    resized_height, resized_width = smart_resize(
        pil_image.height, pil_image.width,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    resized_image = pil_image.resize((resized_width, resized_height))

    tool = ComputerUseTool(display_width_px=resized_width, display_height_px=resized_height)
    messages = build_grounding_messages(instruction, resized_image, tool)

    _t0 = time.time()
    raw_response = qwen_model.generate(
        messages, max_new_tokens=max_new_tokens, temperature=temperature,
    )
    print(f"[ground_toolcall_pixel] generate() 완료 - {time.time() - _t0:.1f}초")

    if debug_text:
        dump_prompt_debug(messages, raw_response, task_id=task_id, step_name="ground_toolcall_pixel")

    tool_call = parse_tool_call(raw_response)
    if tool_call is None:
        return {"result": "wrong_format", "point": None, "raw_response": raw_response}

    try:
        x, y = tool_call["arguments"]["coordinate"]
        point_norm = [x / resized_width, y / resized_height]
        return {"result": "positive", "point": point_norm, "raw_response": raw_response}
    except (KeyError, TypeError, ValueError):
        return {"result": "wrong_format", "point": None, "raw_response": raw_response}


def _cli():
    """
    로컬 실행/디버깅용 CLI.
    --adapter_dir을 지정하면 base 모델 위에 LoRA 어댑터(train.py 체크포인트)를 얹어서
    돌린다 - 안 주면 파인튜닝 안 된 base Qwen2.5-VL로 동작하니 주의.
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="스크린샷 이미지 경로")
    ap.add_argument("--instruction", required=True, help="grounding할 지시문")
    ap.add_argument("--model_id", default=None, help="베이스 모델 id (기본값: qwen.py의 MODEL_ID)")
    ap.add_argument("--adapter_dir", default=None,
                    help="LoRA 어댑터 디렉토리 (train.py --output_dir로 저장된 checkpoint-XXX 폴더)")
    ap.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="샘플링 온도 (기본값 0.0 = greedy decoding)")
    ap.add_argument("--debug_text", action="store_true",
                    help="실제 프롬프트+응답을 ./debug/<task_id>/에 텍스트로 저장")
    ap.add_argument("--task_id", default="demo")
    args = ap.parse_args()

    model_kwargs = dict(
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        adapter_dir=args.adapter_dir,
        load_in_8bit=args.load_in_8bit,
    )
    if args.model_id:
        model_kwargs["model_id"] = args.model_id

    model = QwenVLModel(**model_kwargs)
    result = ground(
        model, args.instruction, args.image,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        temperature=args.temperature,
        debug_text=args.debug_text, task_id=args.task_id,
    )
    print(result)


if __name__ == "__main__":
    _cli()