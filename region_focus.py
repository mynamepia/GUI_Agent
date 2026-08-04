"""
region_focus.py

베이스라인 논문(Qwen25VLModel)의 RegionFocus 파이프라인을 로컬 Qwen2.5-VL 모델
(qwen.py의 QwenVLModel)로 재현한 모듈. (순수 RegionFocus - ZoomClick 전처리 없음.
zoom 버전은 hypo1/region_focus_WJ.py 참고.)

원본은 OpenAI 호환 서빙 엔드포인트(_call_endpoint, vLLM 등)를 통해 추론했지만,
여기서는 gui_grounding.py에서 만든 로컬 파이프라인으로 같은 알고리즘(초기 grounding ->
판단 -> crop/zoom 반복 -> 후보 종합)을 로컬 모델로 돌린다.

원본과 다르게 의도적으로 바꾼 부분 2가지:
    1) judge_inference / next_action_regionfocus_aggregation에는 원본이 실수로(혹은
       습관적으로) computer_use 툴 스키마가 담긴 system 메시지를 끼워 넣었는데, 이 두
       작업은 좌표가 아니라 자유 텍스트(YES/NO, "Selected point: #")로 답해야 하는
       작업이라 오히려 tool_call 포맷을 유도해서 방해가 될 수 있음 - 여기서는 system
       메시지 없이 순수 텍스트 질의로 처리한다.
    2) next_action_regionfocus에서 crop_and_upsample이 만든 "확대된" 이미지를 모델에
       넣기 전에 smart_resize로 한번 더 정렬한다. 원본은 서빙 엔드포인트에 아주 넓은
       min/max_pixels(3136~12845056)를 고정으로 넘겨서 이 문제가 거의 안 드러났지만,
       qwen.py의 기본 min/max_pixels(200704~501760)는 훨씬 좁아서 확대 이미지가 이
       범위를 벗어나기 쉽고, 그러면 processor가 내부에서 우리가 모르는 크기로 또
       리사이즈해버려 좌표가 조용히 어긋나는 버그가 생긴다. smart_resize로 미리
       맞추고 그 배율만큼 zoom_x/zoom_y를 보정해서 이 문제를 없앴다.

[2026-08 수정] initial grounding(local_ground)과 별개로, region_focus()/
next_action_regionfocus() 역시 좌표를 직접 이 LoRA(checkpoint-4130)에게 묻는
호출인데, 지금까지 gui_grounding.ComputerUseTool/build_grounding_messages/
parse_tool_call(Hermes tool-calling 포맷)을 썼다. 이 LoRA는 train.py가 학습시킬 때
쓴 "system 메시지 없음 + PROMPT_TEMPLATE + (x,y) 텍스트" 포맷만 알고 tool-call
포맷은 학습에서 본 적이 없어서, 이 두 함수도 gui_grounding.build_point_prompt_messages()
+ coord_utils.parse_point_from_text()로 바꿨다. judge_inference/
next_action_regionfocus_aggregation은 원래도 system 메시지 없는 자유 텍스트 질의라
이번 수정 대상이 아니다(위 1번 항목 그대로 유지). (hypo1/region_focus_WJ.py에
적용한 것과 동일한 수정 - 두 파일 다 이 픽스가 들어가 있어야 함.)

[디버그 로깅: --debug_image / --debug_text]
둘을 분리해뒀다:
    - debug_image: crop/zoom/판단 과정에서 생기는 중간 "이미지"들을
      ./debug/<task_id>/*.png 로 저장 (기존 --debug와 동일한 이미지들).
    - debug_text : 각 단계(judge_inference/region_focus/next_action_regionfocus/
      aggregation)에서 모델에 실제로 들어간 프롬프트 원문 + 응답 원문을
      ./debug/<task_id>/prompt_<step>.txt 로 저장 (gui_grounding.dump_prompt_debug 사용).
      judge_inference처럼 프롬프팅/파싱 로직 자체가 의심될 때, 실제로 모델한테 뭐가
      들어갔고 뭐가 나왔는지 눈으로 직접 확인하기 위한 용도.
둘 다 독립적으로 켤 수 있다 (이미지만 보고 싶으면 --debug_image만, 프롬프트만 보고
싶으면 --debug_text만).

필요 패키지: qwen.py, gui_grounding.py와 동일 (torch, transformers, qwen-vl-utils, pillow)
(주: 아래 numpy는 plot_points_on_image의 ndarray 입력 지원에 실제로 쓰이고 있음.
opencv-python은 원래 임포트돼 있었으나 본 파일 어디서도 실제로 쓰이지 않아 제거함.)
"""

import os
import re
import io
import json
import math
import time
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from qwen_vl_utils import process_vision_info

from qwen import QwenVLModel, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS
from gui_grounding import (
    build_point_prompt_messages,
    ground as local_ground,
    ground_toolcall_norm1000 as local_ground_toolcall_norm1000,
    ground_toolcall_pixel as local_ground_toolcall_pixel,
    dump_prompt_debug,
    ComputerUseTool,
    build_grounding_messages,
    parse_tool_call,
)
from coord_utils import parse_point_from_text

# [2026-08 Step1/Step4 ablation] ComputerUseTool/build_grounding_messages/parse_tool_call은
# 원래 이 LoRA가 학습받지 못한 old tool-call 포맷이라 gui_grounding.ground()/
# region_focus()/next_action_regionfocus()에서는 안 쓰기로 했지만(파일 상단
# docstring 참고), old 포맷이 new 포맷보다 훨씬 컸던 RegionFocus uplift
# (27.99%->37.81% vs 27.91%->28.07%)가 "tool-call schema 구조" 때문인지
# "raw pixel 좌표" 때문인지 분리해서 보기 위해, 아래 실험용 함수들에서만 예외적으로
# 다시 끌어다 쓴다.
# [결과1] Step4만 old 스키마+0~1000으로 바꾼 조합("toolcall_norm1000")은 오히려
# baseline보다 나빠짐 (28.07%(B) -> 25.71%) - "스키마 자체"가 old 우위의 원인은
# 아니라는 뜻.
# [결과2] Step1까지 같이 "toolcall_norm1000"으로 바꾸면 더 나빠짐 (-> 22~24%대) -
# 초기 grounding 자체가 tool-call+0~1000 조합에서 더 못 찾음.
# [진행중] "toolcall_pixel" = old 프롬프트를 좌표 description/변환까지 100% 그대로
# 재현(0~1000 변환 없음, description도 "픽셀"로 원복)한 컨트롤 실험 - do_sample만
# 고정(결정적)이고 나머지는 old와 완전히 동일. old의 37.81%가 raw pixel 표현 때문인지,
# 아니면 old의 비결정적 샘플링 노이즈였는지 마지막으로 가려보기 위함.

# ---------------------------------------------------------------------------
# 순수 유틸 (모델 호출 없음) - 베이스라인에서 거의 그대로 포팅
# ---------------------------------------------------------------------------
def draw_point(image: Image.Image, point: list, color=None):
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)
        except ValueError:
            color = (255, 0, 0, 128)
    else:
        color = (255, 0, 0, 128)
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    x, y = point
    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)], fill=color
    )
    center_radius = radius * 0.1
    overlay_draw.ellipse(
        [
            (x - center_radius, y - center_radius),
            (x + center_radius, y + center_radius),
        ],
        fill=(0, 255, 0, 255),
    )
    image = image.convert("RGBA")
    combined = Image.alpha_composite(image, overlay)
    return combined.convert("RGB")


def bbox_2_point(bbox, dig=2):
    point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    point = [f"{item:.2f}" for item in point]
    return "({},{})".format(point[0], point[1])


def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    return "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])


def pred_2_point(s):
    floats = re.findall(r"-?\d+\.?\d*", s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0] + floats[2]) / 2, (floats[1] + floats[3]) / 2]
    return None


def extract_bbox(s):
    pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    matches = re.findall(pattern, s)
    if matches:
        last_match = matches[-1]
        return (int(last_match[0]), int(last_match[1])), (
            int(last_match[2]),
            int(last_match[3]),
        )
    return None


def plot_points_on_image(
    image, points, colors=None, sizes=None, markers=None, labels=None, save_path=None
):
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image.copy()

    draw = ImageDraw.Draw(image_pil)

    if colors is None:
        colors = [(255, 0, 255) for _ in range(len(points))]
    elif isinstance(colors, tuple) and len(colors) == 3:
        colors = [colors for _ in range(len(points))]

    if sizes is None:
        sizes = [10 for _ in range(len(points))]
    elif isinstance(sizes, int):
        sizes = [sizes for _ in range(len(points))]

    if markers is None:
        markers = ["star" for _ in range(len(points))]
    elif isinstance(markers, str):
        markers = [markers for _ in range(len(points))]

    for i, (x, y) in enumerate(points):
        x, y = int(x), int(y)
        color = colors[i] if i < len(colors) else (255, 0, 255)
        size = sizes[i] if i < len(sizes) else 10
        marker = markers[i] if i < len(markers) else "star"

        if marker == "star":
            pts = []
            for j in range(5):
                angle_outer = math.pi / 2 + j * 2 * math.pi / 5
                px_outer = x + size * math.cos(angle_outer)
                py_outer = y + size * math.sin(angle_outer)
                pts.append((px_outer, py_outer))

                angle_inner = math.pi / 2 + (j + 0.5) * 2 * math.pi / 5
                px_inner = x + size / 2 * math.cos(angle_inner)
                py_inner = y + size / 2 * math.sin(angle_inner)
                pts.append((px_inner, py_inner))

            draw.polygon(pts, fill=color)

        elif marker == "circle":
            draw.ellipse((x - size, y - size, x + size, y + size), fill=color)

        elif marker == "square":
            draw.rectangle((x - size, y - size, x + size, y + size), fill=color)

        elif marker == "cross":
            draw.line((x - size, y - size, x + size, y + size), fill=color, width=2)
            draw.line((x - size, y + size, x + size, y - size), fill=color, width=2)

        elif marker == "diamond":
            draw.polygon(
                [(x, y - size), (x + size, y), (x, y + size), (x - size, y)],
                fill=color,
            )

        if labels and i < len(labels):
            label = labels[i]
            draw.text((x + size + 2, y - size - 2), str(label), fill=color)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image_pil.save(save_path)

    return image_pil


def calculate_crop_region(
    coords,
    img,
    viewport_width=1280,
    viewport_height=720,
    ratio_x=0.5,
    ratio_y=0.5,
    debug_image=False,
    task_id=None,
    index=None,
):
    x_center, y_center = coords
    viewport_width, viewport_height = img.size

    if x_center > viewport_width or y_center > viewport_height:
        x_center = min(x_center, viewport_width)
        y_center = min(y_center, viewport_height)

    crop_w = float(viewport_width * ratio_x)
    crop_h = float(viewport_height * ratio_y)

    left = x_center - crop_w / 2
    top = y_center - crop_h / 2
    right = left + crop_w
    bottom = top + crop_h

    if left < 0:
        shift = -left
        left += shift
        right += shift
    if right > viewport_width:
        shift = right - viewport_width
        left -= shift
        right -= shift

    if top < 0:
        shift = -top
        top += shift
        bottom += shift
    if bottom > viewport_height:
        shift = bottom - viewport_height
        top -= shift
        bottom -= shift

    left = max(0, left)
    top = max(0, top)
    right = min(viewport_width, right)
    bottom = min(viewport_height, bottom)

    if debug_image:
        debug_dir = f"./debug/{task_id}" if task_id else "./debug"
        os.makedirs(debug_dir, exist_ok=True)
        debug_img = img.copy()
        draw = ImageDraw.Draw(debug_img)
        point_radius = 5
        draw.ellipse(
            (
                x_center - point_radius,
                y_center - point_radius,
                x_center + point_radius,
                y_center + point_radius,
            ),
            fill=(255, 0, 0),
        )
        rect_coords = [
            (left, top),
            (left + crop_w, top),
            (left + crop_w, top + crop_h),
            (left, top + crop_h),
        ]
        draw.line(rect_coords + [rect_coords[0]], fill=(0, 255, 0), width=2)
        crop_debug_filename = (
            f"crop_region_{index}.png" if index is not None else "crop_region.png"
        )
        debug_img.save(os.path.join(debug_dir, crop_debug_filename))

    return left, top, right - left, bottom - top


def crop_and_upsample(bbox, image, debug_image=False, task_id=None, index=None, keep_aspect_ratio=True):
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    img_width, img_height = img.size

    left, top, w, h = bbox
    left = max(0, left)
    top = max(0, top)
    w = min(w, img_width - left)
    h = min(h, img_height - top)

    cropped = img.crop((left, top, left + w, top + h))

    if debug_image:
        debug_dir = f"./debug/{task_id}" if task_id else "./debug"
        os.makedirs(debug_dir, exist_ok=True)
        crop_filename = f"crop_{index}.png" if index is not None else "crop.png"
        cropped.save(os.path.join(debug_dir, crop_filename))

    viewport_width = img_width
    viewport_height = img_height

    # 주의(vestigial): keep_aspect_ratio=False 분기는 현재 코드베이스 어디서도 호출되지
    # 않는다 (ground_with_regionfocus는 항상 keep_aspect_ratio=True로 호출) - 사실상
    # 도달 불가능한 죽은 분기. 삭제하진 않고 표시만 해둠 - 나중에 letterbox 없이 강제로
    # viewport 크기에 맞추는 모드가 필요해지면 그대로 쓸 수 있음.
    if not keep_aspect_ratio:
        upsampled = cropped.resize((viewport_width, viewport_height), Image.Resampling.LANCZOS)
        zoom_x = viewport_width / w
        zoom_y = viewport_height / h
        offset_w = 0
        offset_h = 0
    else:
        zoom_x = viewport_width / w
        zoom_y = viewport_height / h
        zoom_factor = min(zoom_x, zoom_y)

        new_w = round(w * zoom_factor)
        new_h = round(h * zoom_factor)
        upsampled = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 주의(vestigial): offset_w/offset_h는 "letterbox 패딩된 캔버스 중앙에 배치했을 때의
        # 여백"을 가정하고 계산한 값인데, 실제로 반환하는 upsampled 이미지는 패딩 없이
        # new_w x new_h 그대로라 이 여백 자체가 존재하지 않는다. next_action_regionfocus()가
        # 이 두 값을 파라미터로 받긴 하지만 실제로는 안 씀 (좌표 역투영은 zoom_x/zoom_y와
        # left/top만으로 충분히 정확함) - 계산/전달 자체는 안전하니 굳이 지우지 않고 남겨둠.
        offset_w = float(viewport_width - new_w) / 2
        offset_h = float(viewport_height - new_h) / 2

        zoom_x = zoom_factor
        zoom_y = zoom_factor

    if debug_image:
        upsampled_filename = f"upsampled_{index}.png" if index is not None else "upsampled.png"
        upsampled.save(os.path.join(debug_dir, upsampled_filename))

    output_buffer = io.BytesIO()
    upsampled.save(output_buffer, format="PNG")
    screenshot_bytes = output_buffer.getvalue()

    return screenshot_bytes, zoom_x, zoom_y, offset_w, offset_h


# ---------------------------------------------------------------------------
# 로컬 모델 추론 헬퍼 (qwen.py의 generate_text를 temperature/top_p까지 지원하도록 확장)
# ---------------------------------------------------------------------------
def _generate_with_sampling(
    qwen_model: QwenVLModel,
    messages: list,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    step_name: str = "",
) -> str:
    """
    qwen.py의 generate_text()와 동일한 로직이되, temperature/top_p로 샘플링을 제어할
    수 있게 확장한 버전. RegionFocus가 여러 temperature로 후보를 뽑아야 해서 필요함.
    temperature<=0이면 원본과 동일하게 greedy decoding(do_sample=False).

    step_name을 넘기면 이 호출 하나가 끝나는 데 걸린 시간을 찍어준다 - RegionFocus는
    generate()를 여러 번 순차 호출하는 구조라, 어느 단계에서 오래 걸리는지 눈으로
    보려고 넣었다. (프롬프트/응답 원문을 파일로 남기는 --debug_text 로깅은 이 함수를
    호출하는 쪽(judge_inference 등)에서 dump_prompt_debug()로 별도 처리한다 - 이
    함수는 순수 생성만 담당하도록 분리해뒀다.)
    """
    model, processor = qwen_model.model, qwen_model.processor

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

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    _t0 = time.time()
    generated_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - _t0

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    n_new_tokens = len(generated_ids_trimmed[0])
    label = f"[{step_name}] " if step_name else "[generate] "
    print(f"{label}완료 - {elapsed:.1f}초 (토큰 {n_new_tokens}개, {elapsed / max(n_new_tokens,1):.2f}초/토큰)")

    return output_text[0]


# ---------------------------------------------------------------------------
# RegionFocus 알고리즘 본체
# ---------------------------------------------------------------------------
def _parse_judge_verdict(response: str):
    """
    judge_inference 응답에서 {"reason": "...", "ans": "YES/NO"} JSON을 파싱한다.
    모델이 JSON 앞뒤에 군더더기 텍스트를 붙이는 경우까지 커버하려고, 응답 전체에서
    {...} 블록만 정규식으로 뽑아 json.loads를 시도한다.

    JSON 파싱에 실패하면(모델이 포맷을 안 지켰을 때) 예전 방식(대문자 변환 후
    YES/NO 부분 문자열 탐색)으로 폴백한다 - 폴백은 임시 안전망일 뿐이고, 정상적으로는
    위 JSON 포맷 강제로 거의 항상 성공해야 한다.

    Returns: (ans: "YES"|"NO"|None, reason: str|None) - 완전히 파싱 실패하면 (None, None).
    """
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            ans = str(obj.get("ans", "")).strip().upper()
            if ans in ("YES", "NO"):
                return ans, obj.get("reason")
        except (json.JSONDecodeError, AttributeError):
            pass

    # 폴백: JSON 강제가 실패했을 때만 쓰는 예전 substring 방식.
    upper = response.upper()
    has_yes = "YES" in upper or "CORRECT" in upper
    has_no = "NO" in upper or "INCORRECT" in upper
    if has_yes and not has_no:
        return "YES", None
    if has_no:
        return "NO", None
    return None, None


def judge_inference(
    qwen_model, instruction, image, point,
    debug_image=False, debug_text=False, debug_mode="always", task_id=None,
):
    """순수 RegionFocus judge - crop/zoom 전처리 없이 원본 이미지 그대로 별을 찍어서 판정한다."""
    pil_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image).copy()

    highlighted_image = plot_points_on_image(
        pil_image, [point], colors=[(255, 0, 255, 128)], markers=["star"], sizes=[12]
    )

    # (2026-07 업데이트) 기존엔 자유 텍스트로 설명부터 시키고 "YES"/"NO" 부분 문자열을
    # 응답 전체에서 찾는 방식이었는데, --debug_text로 실제 응답을 까보니 진짜 문제는
    # (idx1/idx11 사례) judge 자신의 시각 인식/의미 해석 오류였지 파싱 버그는 아니었음 -
    # 그래도 파싱 자체를 견고하게 만들어두는 게 안전해서, JSON 강제 포맷으로 교체함.
    # reason을 ans보다 먼저 쓰게 해서 결론부터 내리고 사후 정당화하는 대신 판단 근거를
    # 먼저 풀어놓게 유도한다(다만 idx1처럼 순수 시각 오인식은 이 순서 변경으로도 안 고쳐질 수 있음).
    # NEUTRAL은 일단 제외하고 YES/NO 이분법만 유지 (요청에 따름).
    judge_prompt = (
        f'Instruction: "{instruction}"\n'
        f"A pink star marks a candidate click point. The star may only partially cover the "
        f"target, and that still counts as correct.\n\n"
        f'Reply with ONLY this JSON: {{"reason": "<short reason>", "ans": "YES/NO"}}\n'
        f"Think through the reason first, then decide. Be strict: the star must precisely "
        f"match the correct element. If there is any real doubt, or the star seems close but "
        f"not exactly on the target, answer NO."
    )

    # judge는 좌표가 아니라 자유 텍스트 판단이라 tool 스키마(system 메시지) 없이 질의한다.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": highlighted_image},
                {"type": "text", "text": judge_prompt},
            ],
        }
    ]

    response = _generate_with_sampling(
        qwen_model, messages, max_new_tokens=256, temperature=0.0, step_name="judge_inference"
    )

    ans, parsed_reason = _parse_judge_verdict(response)
    if ans is None:
        # JSON도 실패하고 폴백(substring)도 YES/NO를 못 찾은 완전 파싱 실패 케이스 -
        # 안전하게 "오답"으로 처리해서 RegionFocus가 재탐색하도록 한다 (조용히 넘어가지 않음).
        is_correct = False
    else:
        is_correct = ans == "YES"

    # debug_mode="incorrect"면 이 시점(판정 완료 후)에 판단해서 오답 샘플만 저장한다.
    should_dump = debug_mode == "always" or (debug_mode == "incorrect" and not is_correct)

    if debug_image and should_dump:
        debug_dir = f"./debug/{task_id}" if task_id else "./debug"
        os.makedirs(debug_dir, exist_ok=True)
        highlighted_image.save(os.path.join(debug_dir, "initial_point_highlighted.png"))

    if debug_text and should_dump:
        extra = f"Point: {point}\nParsed ans: {ans}\nJudgment: {'CORRECT' if is_correct else 'INCORRECT'}"
        if parsed_reason:
            extra += f"\nParsed reason: {parsed_reason}"
        dump_prompt_debug(messages, response, task_id=task_id, step_name="judge_inference", extra=extra)

    return is_correct, response


def region_focus(
    qwen_model,
    instruction,
    image,
    temperature=0.0,
    top_p=1.0,
    debug_text=False,
    task_id=None,
    min_pixels=DEFAULT_MIN_PIXELS,
    max_pixels=DEFAULT_MAX_PIXELS,
):
    """
    initial grounding이 틀렸다고 판단됐을 때, 다른 temperature로 다시 좌표 후보를 뽑는다.
    image는 이미 smart_resize로 정렬된 상태(= ground_with_regionfocus의 original_image)라고 가정.

    [2026-08 수정] Hermes tool-calling 포맷(ComputerUseTool/build_grounding_messages/
    parse_tool_call) 대신, 이 LoRA가 실제로 학습받은 포맷
    (gui_grounding.build_point_prompt_messages + coord_utils.parse_point_from_text)으로
    묻는다. 자세한 이유는 파일 상단 docstring 참고.
    """
    pil_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image).copy()

    messages = build_point_prompt_messages(instruction, pil_image)

    response = _generate_with_sampling(
        qwen_model,
        messages,
        max_new_tokens=128,
        temperature=temperature,
        top_p=top_p,
        step_name=f"region_focus(temp={temperature})",
    )

    if debug_text:
        dump_prompt_debug(
            messages, response, task_id=task_id, step_name=f"region_focus_temp{temperature}",
        )

    norm_point = parse_point_from_text(response)
    if norm_point is None:
        return None, response

    nx, ny = norm_point
    # 0~1000 상대좌표는 해상도 무관이라 nx/1000, ny/1000이 곧바로 pil_image 기준 0~1
    # 정규화 좌표다 (gui_grounding.ground()와 동일한 근거). 모델이 가끔 범위를 살짝
    # 벗어난 값을 내놓는 경우가 있어(특히 gt target이 화면 가장자리에 붙어있을 때),
    # 하류의 calculate_crop_region 강제 edge-clamping 등으로 예측 불가능하게 전파되지
    # 않도록 여기서 [0,1]로 방어적으로 클램프한다(원본 모델 오차 자체를 고치는 건 아님).
    x_norm = max(0.0, min(1.0, nx / 1000))
    y_norm = max(0.0, min(1.0, ny / 1000))
    return [x_norm, y_norm], response


def next_action_regionfocus(
    qwen_model,
    instruction,
    zoomed_img_bytes,
    left,
    top,
    zoom_x,
    zoom_y,
    offset_w,  # vestigial: crop_and_upsample() 참고 - 실제로 아래 로직에서 안 씀
    offset_h,  # vestigial: 위와 동일
    w,
    h,
    original_image,
    debug_image=False,
    debug_text=False,
    task_id=None,
    index=None,
    temperature=0.0,
    top_p=1.0,
    min_pixels=DEFAULT_MIN_PIXELS,
    max_pixels=DEFAULT_MAX_PIXELS,
):
    """
    확대(crop+upsample)된 영역 위에서 다시 좌표를 찍고, 원본 이미지 좌표로 역투영한다.

    offset_w/offset_h 파라미터는 받기만 하고 실제로 안 쓴다 - crop_and_upsample()의
    같은 이름 변수 주석 참고 (letterbox 패딩이 없는 구조라 애초에 불필요한 값).
    """
    raw_zoomed_img = Image.open(io.BytesIO(zoomed_img_bytes))

    # crop_and_upsample이 만든 이미지는 28의 배수/설정된 pixel 범위에 안 맞을 수 있다.
    # smart_resize로 모델이 실제로 보게 될 크기를 우리가 직접 고정하고, 그만큼 추가로
    # 늘어나거나 줄어든 비율을 zoom_x/zoom_y에 반영해서 좌표 역투영이 어긋나지 않게 한다.
    resized_h, resized_w = smart_resize(
        raw_zoomed_img.height, raw_zoomed_img.width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    zoomed_img = raw_zoomed_img.resize((resized_w, resized_h))
    extra_zoom_x = resized_w / raw_zoomed_img.width
    extra_zoom_y = resized_h / raw_zoomed_img.height
    zoom_x = zoom_x * extra_zoom_x
    zoom_y = zoom_y * extra_zoom_y

    # [2026-08 수정] Hermes tool-calling 포맷 대신 이 LoRA가 학습받은 포맷
    # (build_point_prompt_messages + parse_point_from_text)으로 묻는다.
    # 자세한 이유는 파일 상단 docstring 참고.
    messages = build_point_prompt_messages(instruction, zoomed_img)

    response = _generate_with_sampling(
        qwen_model,
        messages,
        max_new_tokens=128,
        temperature=temperature,
        top_p=top_p,
        step_name=f"next_action_regionfocus(idx={index})",
    )

    if debug_text:
        dump_prompt_debug(
            messages, response, task_id=task_id, step_name="next_action_regionfocus", index=index,
        )

    norm_point = parse_point_from_text(response)
    if norm_point is None:
        return None, response

    nx, ny = norm_point
    # 여기서부터는 zoom_x/zoom_y로 원본 이미지 좌표계에 역투영해야 해서 실제 픽셀좌표가
    # 필요하다. 0~1000 상대좌표를 모델이 실제로 본 이미지(zoomed_img, resized_w x
    # resized_h) 기준 픽셀좌표로 변환한다 - 기존 tool-call 방식이 픽셀좌표를 직접
    # 받던 것과 동일한 스케일로 맞추는 것 뿐, 역투영 로직 자체는 그대로다.
    x_upsampled = round(nx / 1000 * resized_w)
    y_upsampled = round(ny / 1000 * resized_h)

    zoomed_width_calc = w * zoom_x
    zoomed_height_calc = h * zoom_y

    if 0 <= x_upsampled < zoomed_width_calc and 0 <= y_upsampled < zoomed_height_calc:
        x_orig = left + (x_upsampled / zoom_x)
        y_orig = top + (y_upsampled / zoom_y)
    else:
        clamped_x = max(0, min(zoomed_width_calc - 1, x_upsampled))
        clamped_y = max(0, min(zoomed_height_calc - 1, y_upsampled))
        x_orig = left + (clamped_x / zoom_x)
        y_orig = top + (clamped_y / zoom_y)

    if isinstance(original_image, Image.Image):
        img_width, img_height = original_image.size
    else:
        img_height, img_width = original_image.shape[:2]

    x_orig = max(0, min(x_orig, img_width - 1))
    y_orig = max(0, min(y_orig, img_height - 1))

    projected_point = (round(x_orig), round(y_orig))

    if debug_image:
        debug_dir = f"./debug/{task_id}" if task_id else "./debug"
        os.makedirs(debug_dir, exist_ok=True)

        original_pil = (
            original_image.copy()
            if isinstance(original_image, Image.Image)
            else Image.fromarray(original_image).copy()
        )

        zoomed_debug = plot_points_on_image(
            zoomed_img, [(x_upsampled, y_upsampled)], colors=[(255, 0, 255)], markers=["star"], sizes=[15]
        )
        original_debug = plot_points_on_image(
            original_pil, [projected_point], colors=[(255, 0, 255)], markers=["star"], sizes=[15]
        )

        zoomed_debug.save(os.path.join(debug_dir, f"RegionFocus_upsampled_{index}.png"))
        original_debug.save(os.path.join(debug_dir, f"RegionFocus_unprojected_{index}.png"))

    return projected_point, response


def next_action_regionfocus_toolcall_norm1000(
    qwen_model,
    instruction,
    zoomed_img_bytes,
    left,
    top,
    zoom_x,
    zoom_y,
    offset_w,  # vestigial: crop_and_upsample() 참고 - 실제로 아래 로직에서 안 씀
    offset_h,  # vestigial: 위와 동일
    w,
    h,
    original_image,
    debug_image=False,
    debug_text=False,
    task_id=None,
    index=None,
    temperature=0.0,
    top_p=1.0,
    min_pixels=DEFAULT_MIN_PIXELS,
    max_pixels=DEFAULT_MAX_PIXELS,
):
    """
    [Step4 ablation 옵션C] next_action_regionfocus()와 역할/위치는 완전히 동일
    (crop+upsample된 이미지 위에서 좌표를 다시 찍고 원본 이미지 좌표로 역투영)하지만,
    좌표를 묻는 방식만 old RegionFocus가 쓰던 tool-call(computer_use function-calling)
    스키마로 바꾸고, 그 coordinate 값 자체는 old(raw pixel)가 아니라 new와 동일한
    0~1000 정규화로 받는다.

    3-way ablation:
        A(old)     = tool-call schema + raw pixel coordinate  (region_focus_JH.py 등 옛 코드)
        B(new,기본) = point-text(build_point_prompt_messages) + 0~1000 정규화 (next_action_regionfocus)
        C(이 함수)  = tool-call schema + 0~1000 정규화 좌표

    old(A)가 new(B)보다 Step4에서 훨씬 큰 RegionFocus uplift(27.99%->37.81% vs
    27.91%->28.07%)를 보였던 게 "tool-call schema 구조" 때문인지 "raw pixel 좌표
    표현" 때문인지 분리하기 위한 실험용 함수. C가 B와 비슷하면 schema는 무관하고
    좌표 표현(0~1000 vs raw pixel)이 원인, C가 A와 비슷하면 schema 자체가 원인이라는
    뜻이 된다.
    """
    raw_zoomed_img = Image.open(io.BytesIO(zoomed_img_bytes))

    resized_h, resized_w = smart_resize(
        raw_zoomed_img.height, raw_zoomed_img.width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    zoomed_img = raw_zoomed_img.resize((resized_w, resized_h))
    extra_zoom_x = resized_w / raw_zoomed_img.width
    extra_zoom_y = resized_h / raw_zoomed_img.height
    zoom_x = zoom_x * extra_zoom_x
    zoom_y = zoom_y * extra_zoom_y

    tool = ComputerUseTool(resized_w, resized_h, coord_mode="norm1000")
    prompt_text = (
        f'For this zoomed-in screenshot, identify the precise point that best matches the '
        f'instruction: "{instruction}", by calling the computer_use function with a '
        f'left_click action.'
    )
    messages = build_grounding_messages(instruction, zoomed_img, tool, prompt_text=prompt_text)

    response = _generate_with_sampling(
        qwen_model,
        messages,
        max_new_tokens=128,
        temperature=temperature,
        top_p=top_p,
        step_name=f"next_action_regionfocus_toolcall_norm1000(idx={index})",
    )

    if debug_text:
        dump_prompt_debug(
            messages, response, task_id=task_id,
            step_name="next_action_regionfocus_toolcall_norm1000", index=index,
        )

    tool_call = parse_tool_call(response)
    if not tool_call:
        return None, response
    coord = (tool_call.get("arguments") or {}).get("coordinate")
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None, response
    try:
        nx, ny = float(coord[0]), float(coord[1])
    except (TypeError, ValueError):
        return None, response

    # 여기서부터는 next_action_regionfocus()와 동일한 역투영 로직 - 차이는 오직
    # "0~1000 정규화 좌표를 모델이 실제로 본 이미지(resized_w x resized_h) 기준
    # 픽셀좌표로 변환"하는 첫 두 줄뿐 (old raw-pixel 버전은 이 변환 없이 coordinate를
    # 그대로 픽셀좌표로 씀).
    x_upsampled = round(nx / 1000 * resized_w)
    y_upsampled = round(ny / 1000 * resized_h)

    zoomed_width_calc = w * zoom_x
    zoomed_height_calc = h * zoom_y

    if 0 <= x_upsampled < zoomed_width_calc and 0 <= y_upsampled < zoomed_height_calc:
        x_orig = left + (x_upsampled / zoom_x)
        y_orig = top + (y_upsampled / zoom_y)
    else:
        clamped_x = max(0, min(zoomed_width_calc - 1, x_upsampled))
        clamped_y = max(0, min(zoomed_height_calc - 1, y_upsampled))
        x_orig = left + (clamped_x / zoom_x)
        y_orig = top + (clamped_y / zoom_y)

    if isinstance(original_image, Image.Image):
        img_width, img_height = original_image.size
    else:
        img_height, img_width = original_image.shape[:2]

    x_orig = max(0, min(x_orig, img_width - 1))
    y_orig = max(0, min(y_orig, img_height - 1))

    projected_point = (round(x_orig), round(y_orig))

    if debug_image:
        debug_dir = f"./debug/{task_id}" if task_id else "./debug"
        os.makedirs(debug_dir, exist_ok=True)

        original_pil = (
            original_image.copy()
            if isinstance(original_image, Image.Image)
            else Image.fromarray(original_image).copy()
        )

        zoomed_debug = plot_points_on_image(
            zoomed_img, [(x_upsampled, y_upsampled)], colors=[(255, 0, 255)], markers=["star"], sizes=[15]
        )
        original_debug = plot_points_on_image(
            original_pil, [projected_point], colors=[(255, 0, 255)], markers=["star"], sizes=[15]
        )

        zoomed_debug.save(os.path.join(debug_dir, f"RegionFocus_upsampled_{index}.png"))
        original_debug.save(os.path.join(debug_dir, f"RegionFocus_unprojected_{index}.png"))

    return projected_point, response


def next_action_regionfocus_toolcall_pixel(
    qwen_model,
    instruction,
    zoomed_img_bytes,
    left,
    top,
    zoom_x,
    zoom_y,
    offset_w,  # vestigial: crop_and_upsample() 참고 - 실제로 아래 로직에서 안 씀
    offset_h,  # vestigial: 위와 동일
    w,
    h,
    original_image,
    debug_image=False,
    debug_text=False,
    task_id=None,
    index=None,
    temperature=0.0,
    top_p=1.0,
    min_pixels=DEFAULT_MIN_PIXELS,
    max_pixels=DEFAULT_MAX_PIXELS,
):
    """
    [Step4 - old 프롬프트 그대로] bak/region_focus.py의 원래(수정 전)
    next_action_regionfocus()를 그대로 이식한 버전. tool-call 스키마, coordinate
    description("픽셀 좌표"), 좌표 처리(0~1000 변환 없이 tool_call의 coordinate를
    그대로 x_upsampled/y_upsampled로 사용) 전부 old와 100% 동일. 유일한 차이는
    _generate_with_sampling()이 원래도 명시적 do_sample 제어를 갖고 있었으므로
    (이 부분은 old/new 모두 동일 - Step1과 달리 Step4는 애초에 결정성 버그가 없었음)
    실질적으로 old의 next_action_regionfocus()와 완전히 동일한 함수다.

    ground_toolcall_pixel()(Step1)과 짝을 이뤄 "old 스키마+old 좌표표현" 조합을
    양쪽 step 모두에서 재현하기 위한 컨트롤 실험용.
    """
    raw_zoomed_img = Image.open(io.BytesIO(zoomed_img_bytes))

    resized_h, resized_w = smart_resize(
        raw_zoomed_img.height, raw_zoomed_img.width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    zoomed_img = raw_zoomed_img.resize((resized_w, resized_h))
    extra_zoom_x = resized_w / raw_zoomed_img.width
    extra_zoom_y = resized_h / raw_zoomed_img.height
    zoom_x = zoom_x * extra_zoom_x
    zoom_y = zoom_y * extra_zoom_y

    tool = ComputerUseTool(display_width_px=resized_w, display_height_px=resized_h)
    prompt_text = (
        f"For this zoomed-in screenshot, identify the precise point that best matches "
        f'the instruction: "{instruction}", by calling the computer_use function with a '
        f"left_click action."
    )
    messages = build_grounding_messages(instruction, zoomed_img, tool, prompt_text=prompt_text)

    response = _generate_with_sampling(
        qwen_model,
        messages,
        max_new_tokens=128,
        temperature=temperature,
        top_p=top_p,
        step_name=f"next_action_regionfocus_toolcall_pixel(idx={index})",
    )

    if debug_text:
        dump_prompt_debug(
            messages, response, task_id=task_id, step_name="next_action_regionfocus_toolcall_pixel", index=index,
        )

    tool_call = parse_tool_call(response)
    if tool_call is None:
        return None, response

    try:
        click_point = tool_call["arguments"]["coordinate"]
    except (KeyError, TypeError, ValueError):
        return None, response

    x_upsampled, y_upsampled = click_point
    x_upsampled, y_upsampled = round(x_upsampled), round(y_upsampled)

    zoomed_width_calc = w * zoom_x
    zoomed_height_calc = h * zoom_y

    if 0 <= x_upsampled < zoomed_width_calc and 0 <= y_upsampled < zoomed_height_calc:
        x_orig = left + (x_upsampled / zoom_x)
        y_orig = top + (y_upsampled / zoom_y)
    else:
        clamped_x = max(0, min(zoomed_width_calc - 1, x_upsampled))
        clamped_y = max(0, min(zoomed_height_calc - 1, y_upsampled))
        x_orig = left + (clamped_x / zoom_x)
        y_orig = top + (clamped_y / zoom_y)

    if isinstance(original_image, Image.Image):
        img_width, img_height = original_image.size
    else:
        img_height, img_width = original_image.shape[:2]

    x_orig = max(0, min(x_orig, img_width - 1))
    y_orig = max(0, min(y_orig, img_height - 1))

    projected_point = (round(x_orig), round(y_orig))

    if debug_image:
        debug_dir = f"./debug/{task_id}" if task_id else "./debug"
        os.makedirs(debug_dir, exist_ok=True)

        original_pil = (
            original_image.copy()
            if isinstance(original_image, Image.Image)
            else Image.fromarray(original_image).copy()
        )

        zoomed_debug = plot_points_on_image(
            zoomed_img, [(x_upsampled, y_upsampled)], colors=[(255, 0, 255)], markers=["star"], sizes=[15]
        )
        original_debug = plot_points_on_image(
            original_pil, [projected_point], colors=[(255, 0, 255)], markers=["star"], sizes=[15]
        )

        zoomed_debug.save(os.path.join(debug_dir, f"RegionFocus_upsampled_{index}.png"))
        original_debug.save(os.path.join(debug_dir, f"RegionFocus_unprojected_{index}.png"))

    return projected_point, response


def next_action_regionfocus_aggregation(
    qwen_model, instruction, image, points, debug_image=False, debug_text=False, task_id=None
):
    """여러 후보 좌표 중 instruction에 가장 잘 맞는 것을 모델에게 고르게 한다."""
    if not points:
        return None, "No points to aggregate"

    if len(points) == 1:
        return points[0], "Only one point available, selected automatically."

    vis_image = (
        Image.open(image).copy()
        if isinstance(image, str)
        else (image.copy() if isinstance(image, Image.Image) else Image.fromarray(image).copy())
    )

    labels = [str(i + 1) for i in range(len(points))]
    aggregated_image = plot_points_on_image(
        vis_image,
        points,
        colors=[(255, 0, 255, 128) for _ in range(len(points))],
        markers=["star" for _ in range(len(points))],
        sizes=[8 for _ in range(len(points))],
        labels=labels,
    )

    debug_dir = f"./debug/{task_id}" if task_id else "./debug"
    if debug_image:
        os.makedirs(debug_dir, exist_ok=True)
        aggregated_image.save(os.path.join(debug_dir, "RegionFocus_aggregated.png"))

    selection_prompt = (
        f"In the image, I've identified {len(points)} potential points (numbered 1-{len(points)}) "
        f'that might match the instruction: "{instruction}". '
        f"Carefully analyze each point and select the ONE that best matches the instruction. "
        f"Sometimes, multiple points may overlap, and you need to select one from the overlapping "
        f"area. Additionally, the correct point might sometimes cover the target, and you need to "
        f"distinguish this scenario. "
        f'Provide your final answer in this format: "Selected point: #" where # is the number of '
        f"the best point."
    )

    # 자유 텍스트("Selected point: #")로 답해야 하는 작업이라 tool 스키마 없이 질의한다.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": aggregated_image},
                {"type": "text", "text": selection_prompt},
            ],
        }
    ]

    response = _generate_with_sampling(
        qwen_model, messages, max_new_tokens=256, temperature=0.0, step_name="aggregation"
    )

    match = re.search(r"Selected point:\s*(\d+)", response)

    if debug_text:
        dump_prompt_debug(
            messages, response, task_id=task_id, step_name="aggregation",
            extra=f"Parsed selection: {match.group(1) if match else '(파싱 실패 - 1번으로 fallback)'}",
        )

    if match:
        selected_idx = int(match.group(1)) - 1
        if 0 <= selected_idx < len(points):
            selected_point = points[selected_idx]
            if debug_image:
                final_image = plot_points_on_image(
                    vis_image, [selected_point], colors=[(0, 255, 0)], markers=["star"], sizes=[20]
                )
                final_image.save(os.path.join(debug_dir, "RegionFocus_final.png"))
            return selected_point, response

    return points[0], response + "\n(No valid selection found, using first point as fallback.)"


def ground_with_regionfocus(
    qwen_model, instruction, image,
    debug_image=False, debug_text=False, debug_mode="always", task_id=None,
    min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS,
    step4_format="point_text",
    step1_format="point_text",
) -> dict:
    """
    베이스라인 Qwen25VLModel.ground_with_regionfocus()의 로컬 모델 버전 (순수, zoom 없음).
    1) 초기 grounding (gui_grounding.ground) -> 2) 판단 -> 3) 틀렸으면 region_focus로
    재탐색 -> 4) crop/zoom 4가지 비율로 정밀화 -> 5) 후보 종합, 순서 그대로.

    반환 스키마는 gui_grounding.ground()와 동일하게 "result" 키("positive"/"wrong_format")를
    항상 포함하도록 통일했다 (기존엔 RegionFocus 경로를 타면 이 키가 빠져서, 두 grounding
    경로를 같은 인터페이스로 다루는 상위 코드에서 KeyError가 날 수 있었음).

    debug_image / debug_text: 각각 독립적으로 켤 수 있음 (자세한 설명은 파일 상단 참고).
    debug_mode: "always"(기본) / "incorrect" - judge_inference의 판정 게이팅 참고
    (judge_inference 함수 docstring). 단, Step 1(초기 grounding)의 프롬프트 텍스트 덤프는
    judge 판정 전에 실행되는 단계라 debug_mode와 무관하게 debug_text가 켜져 있으면 항상
    저장된다 (파일 하나짜리라 용량 부담이 거의 없어서 이 부분만 예외로 뒀다).

    step4_format / step1_format: "point_text"(기본) | "toolcall_norm1000" | "toolcall_pixel".
    각각 Step4(crop/zoom 후 좌표 재추출) / Step1(초기 grounding)에서만 쓰는 좌표
    요청 방식을 바꾼다. 나머지 step(2/3/5)은 이 값들과 무관하게 항상 기존 방식
    (자유 텍스트 판정/point_text) 그대로.
        point_text        : 기본, LoRA 학습 포맷 그대로.
        toolcall_norm1000 : old tool-call schema + 0~1000 정규화 좌표 (schema만 old화).
        toolcall_pixel     : old 프롬프트 100% 그대로(description도 raw pixel, 좌표
                              변환도 0~1000 없이 그대로) - do_sample만 결정적으로 고정.
    자세한 설명은 next_action_regionfocus_toolcall_norm1000()/_toolcall_pixel(),
    ground_toolcall_norm1000()/_toolcall_pixel() docstring 참고.
    """
    _valid_formats = ("point_text", "toolcall_norm1000", "toolcall_pixel")
    assert step4_format in _valid_formats, f"unknown step4_format: {step4_format}"
    assert step1_format in _valid_formats, f"unknown step1_format: {step1_format}"
    debug_dir = f"./debug/{task_id}" if task_id else "./debug"
    if debug_image or debug_text:
        os.makedirs(debug_dir, exist_ok=True)

    overall_start = time.time()

    def _log(msg):
        print(f"[RegionFocus +{time.time() - overall_start:.1f}s] {msg}")

    pil_image = Image.open(image) if isinstance(image, str) else image

    # Step 1: 초기 grounding (local_ground와 동일한 smart_resize 기준으로 원본 크기 재계산)
    _log("Step 1/5: 초기 grounding 시작")
    step1_fn = {
        "point_text": local_ground,
        "toolcall_norm1000": local_ground_toolcall_norm1000,
        "toolcall_pixel": local_ground_toolcall_pixel,
    }[step1_format]
    initial_result = step1_fn(
        qwen_model, instruction, pil_image, min_pixels=min_pixels, max_pixels=max_pixels,
        debug_text=debug_text, task_id=task_id,
    )
    resized_height, resized_width = smart_resize(
        pil_image.height, pil_image.width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    original_image = pil_image.resize((resized_width, resized_height))
    _log(f"Step 1/5 완료 - point={initial_result['point']}")

    # Step 2: 초기 grounding 판단
    if initial_result["point"]:
        point_px = [
            round(initial_result["point"][0] * original_image.width),
            round(initial_result["point"][1] * original_image.height),
        ]
        _log("Step 2/5: 초기 grounding 판단(judge_inference) 시작")
        is_correct, judge_response = judge_inference(
            qwen_model, instruction, original_image, point_px,
            debug_image=debug_image, debug_text=debug_text, debug_mode=debug_mode, task_id=task_id,
        )
        _log(f"Step 2/5 완료 - {'정답, 여기서 종료' if is_correct else '오답, RegionFocus 진행'}")
        if is_correct:
            _log(f"총 소요시간 {time.time() - overall_start:.1f}초")
            return initial_result
    else:
        is_correct = False
        judge_response = "No valid point found in initial grounding."
        _log("Step 2/5: 초기 grounding에서 유효한 point를 못 찾음, RegionFocus 진행")

    # Step 3: RegionFocus로 재탐색 (temperature를 올려가며 하나 찾으면 중단)
    region_points = []
    for temp in [0.0, 0.3, 0.5, 0.7, 0.9]:
        _log(f"Step 3/5: region_focus 재시도 (temperature={temp})")
        point, response = region_focus(
            qwen_model,
            instruction,
            original_image,
            temperature=temp,
            top_p=0.90,
            debug_text=debug_text,
            task_id=task_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if point:
            region_points.append(point)
            break

    if not region_points:
        _log("Step 3/5 실패 - RegionFocus 후보를 못 찾아서 초기 결과 반환")
        _log(f"총 소요시간 {time.time() - overall_start:.1f}초")
        return initial_result
    _log(f"Step 3/5 완료 - point={region_points[0]}")

    # Step 4: crop/zoom 비율 4가지로 후보 좌표 정밀화
    zoomed_results = []
    ratio_list = [[0.5, 0.5], [0.3, 0.3], [0.4, 0.8], [0.8, 0.4]]
    point = region_points[0]
    for i, ratio in enumerate(ratio_list):
        _log(f"Step 4/5: crop/zoom {i+1}/{len(ratio_list)} (ratio={ratio}) 시작")
        left, top, w, h = calculate_crop_region(
            [round(point[0] * original_image.width), round(point[1] * original_image.height)],
            original_image,
            debug_image=debug_image,
            task_id=task_id,
            index=i,
            ratio_x=ratio[0],
            ratio_y=ratio[1],
        )
        zoomed_bytes, zoom_x, zoom_y, offset_w, offset_h = crop_and_upsample(
            (left, top, w, h), original_image, keep_aspect_ratio=True,
            debug_image=debug_image, task_id=task_id, index=i,
        )
        step4_fn = {
            "point_text": next_action_regionfocus,
            "toolcall_norm1000": next_action_regionfocus_toolcall_norm1000,
            "toolcall_pixel": next_action_regionfocus_toolcall_pixel,
        }[step4_format]
        action_point, action_response = step4_fn(
            qwen_model,
            instruction,
            zoomed_bytes,
            left,
            top,
            zoom_x,
            zoom_y,
            offset_w,
            offset_h,
            w,
            h,
            original_image,
            debug_image=debug_image,
            debug_text=debug_text,
            task_id=task_id,
            index=i,
            temperature=0.0,
            top_p=1.0,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if action_point:
            zoomed_results.append((action_point, action_response))
            _log(f"Step 4/5: crop/zoom {i+1}/{len(ratio_list)} 완료 - point={action_point}")
        else:
            _log(f"Step 4/5: crop/zoom {i+1}/{len(ratio_list)} 실패 (유효한 tool_call 없음)")

    if not zoomed_results:
        _log("Step 4/5 전부 실패 - 후보 없음, 초기 결과로 대체")
        _log(f"총 소요시간 {time.time() - overall_start:.1f}초")
        if initial_result["point"]:
            return initial_result
        return {
            "result": "wrong_format",
            "point": None,
            "bbox": None,
            "raw_response": "no valid points found from zoomed regions",
        }

    # Step 5: 후보 종합
    _log(f"Step 5/5: 후보 {len(zoomed_results)}개 종합 시작")
    final_points = [p for p, _ in zoomed_results]
    if len(final_points) > 1:
        best_point, agg_response = next_action_regionfocus_aggregation(
            qwen_model, instruction, original_image, final_points,
            debug_image=debug_image, debug_text=debug_text, task_id=task_id,
        )
    else:
        best_point, agg_response = zoomed_results[0]
    _log(f"Step 5/5 완료 - 최종 point={best_point}")
    _log(f"총 소요시간 {time.time() - overall_start:.1f}초")

    return {
        "result": "positive",
        "point": [best_point[0] / original_image.width, best_point[1] / original_image.height],
        "bbox": None,
        "regionfocus_applied": True,
        "initial_point": initial_result["point"],
        "initial_correct": is_correct,
        "num_candidates": len(zoomed_results),
        "raw_response": agg_response,
    }


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
    ap.add_argument("--debug_image", action="store_true",
                    help="crop/zoom/판단 과정의 중간 이미지들을 ./debug/<task_id>/*.png로 저장")
    ap.add_argument("--debug_text", action="store_true",
                    help="각 단계에 실제로 들어간 프롬프트+응답 원문을 ./debug/<task_id>/prompt_*.txt로 저장")
    ap.add_argument("--debug_mode", choices=["always", "incorrect"], default="always",
                    help="always: 판정과 무관하게 항상 저장 / incorrect: judge가 오답으로 판단한 "
                         "샘플만 저장 (정답 조기종료 샘플은 스킵)")
    ap.add_argument("--step4_format", choices=["point_text", "toolcall_norm1000", "toolcall_pixel"],
                    default="point_text",
                    help="point_text: 기본(학습 포맷 그대로) / toolcall_norm1000: old tool-call "
                         "schema + 0~1000 정규화 좌표 / toolcall_pixel: old 프롬프트 그대로(raw pixel)")
    ap.add_argument("--step1_format", choices=["point_text", "toolcall_norm1000", "toolcall_pixel"],
                    default="point_text",
                    help="point_text: 기본(학습 포맷 그대로) / toolcall_norm1000: old tool-call "
                         "schema + 0~1000 정규화 좌표 / toolcall_pixel: old 프롬프트 그대로(raw pixel). "
                         "--step4_format과 독립적으로 켤 수 있음")
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
    result = ground_with_regionfocus(
        model, args.instruction, args.image,
        debug_image=args.debug_image, debug_text=args.debug_text, debug_mode=args.debug_mode,
        task_id=args.task_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        step4_format=args.step4_format, step1_format=args.step1_format,
    )
    print(result)


if __name__ == "__main__":
    _cli()
