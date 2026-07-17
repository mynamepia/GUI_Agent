"""
region_focus.py

베이스라인 논문(Qwen25VLModel)의 RegionFocus 파이프라인을 로컬 Qwen2.5-VL 모델
(qwen.py의 QwenVLModel)로 재현한 모듈.

원본은 OpenAI 호환 서빙 엔드포인트(_call_endpoint, vLLM 등)를 통해 추론했지만,
여기서는 gui_grounding.py에서 만든 로컬 function-calling 파이프라인
(ComputerUseTool / build_grounding_messages / parse_tool_call)을 그대로 재사용해서
같은 알고리즘(초기 grounding -> 판단 -> crop/zoom 반복 -> 후보 종합)을 로컬 모델로 돌린다.

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

필요 패키지: qwen.py, gui_grounding.py와 동일 (torch, transformers, qwen-vl-utils,
pillow, opencv-python, numpy)
"""

import os
import re
import io
import math
import time
import base64
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageColor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from qwen_vl_utils import process_vision_info

from qwen import QwenVLModel, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS
from gui_grounding import (
    ComputerUseTool,
    build_grounding_messages,
    parse_tool_call,
    ground as local_ground,
)


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
    debug=False,
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

    if debug:
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


def crop_and_upsample(bbox, image, debug=False, task_id=None, index=None, keep_aspect_ratio=True):
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    img_width, img_height = img.size

    left, top, w, h = bbox
    left = max(0, left)
    top = max(0, top)
    w = min(w, img_width - left)
    h = min(h, img_height - top)

    cropped = img.crop((left, top, left + w, top + h))

    if debug:
        debug_dir = f"./debug/{task_id}" if task_id else "./debug"
        os.makedirs(debug_dir, exist_ok=True)
        crop_filename = f"crop_{index}.png" if index is not None else "crop.png"
        cropped.save(os.path.join(debug_dir, crop_filename))

    viewport_width = img_width
    viewport_height = img_height

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

        offset_w = float(viewport_width - new_w) / 2
        offset_h = float(viewport_height - new_h) / 2

        zoom_x = zoom_factor
        zoom_y = zoom_factor

    if debug:
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
    보려고 넣었다.
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
def judge_inference(qwen_model, instruction, image, point, debug=False, task_id=None):
    """초기 grounding 결과(point)가 instruction에 정확히 맞는지 모델에게 YES/NO로 판단시킨다."""
    pil_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image).copy()

    highlighted_image = plot_points_on_image(
        pil_image, [point], colors=[(255, 0, 255, 128)], markers=["star"], sizes=[12]
    )

    debug_dir = f"./debug/{task_id}" if task_id else "./debug"
    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        highlighted_image.save(os.path.join(debug_dir, "initial_point_highlighted.png"))

    judge_prompt = (
        f'Given the instruction: "{instruction}", I highlighted a pink star on the image, '
        f"Is this pink star position correct and precise for the instruction? "
        f"Sometimes, the point might cover the target, which is correct, and you need to "
        f"distinguish this scenario. "
        f"Answer YES if it accurately identifies the element mentioned in the instruction. "
        f"Answer NO if it's incorrect or imprecise. "
        f"Thoughts: Please explain your reasoning and be specific about why the point is "
        f"correct or incorrect."
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

    upper = response.upper()
    is_correct = ("YES" in upper or "CORRECT" in upper or "정확" in response or "정밀" in response) and not (
        "NO" in upper or "INCORRECT" in upper or "부정확" in response or "부정밀" in response
    )

    if debug:
        with open(os.path.join(debug_dir, "judgment_response.txt"), "w") as f:
            f.write(f"Instruction: {instruction}\n\n")
            f.write(f"Point: {point}\n\n")
            f.write(f"Judgment: {'CORRECT' if is_correct else 'INCORRECT'}\n\n")
            f.write(f"Response:\n{response}")

    return is_correct, response


def region_focus(
    qwen_model,
    instruction,
    image,
    temperature=0.0,
    top_p=1.0,
    debug=False,
    task_id=None,
    min_pixels=DEFAULT_MIN_PIXELS,
    max_pixels=DEFAULT_MAX_PIXELS,
):
    """
    initial grounding이 틀렸다고 판단됐을 때, 다른 temperature로 다시 좌표 후보를 뽑는다.
    image는 이미 smart_resize로 정렬된 상태(= ground_with_regionfocus의 original_image)라고 가정.
    """
    pil_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image).copy()
    img_width, img_height = pil_image.size

    tool = ComputerUseTool(display_width_px=img_width, display_height_px=img_height)
    prompt_text = (
        f'Given the instruction: "{instruction}", locate the most relevant coordinates in '
        f"the image that best matches the instruction, by calling the computer_use function "
        f"with a left_click action."
    )
    messages = build_grounding_messages(instruction, pil_image, tool, prompt_text=prompt_text)

    response = _generate_with_sampling(
        qwen_model,
        messages,
        max_new_tokens=128,
        temperature=temperature,
        top_p=top_p,
        step_name=f"region_focus(temp={temperature})",
    )

    tool_call = parse_tool_call(response)
    if tool_call is None:
        return None, response

    try:
        x, y = tool_call["arguments"]["coordinate"]
        return [x / img_width, y / img_height], response
    except (KeyError, TypeError, ValueError):
        return None, response


def next_action_regionfocus(
    qwen_model,
    instruction,
    zoomed_img_bytes,
    left,
    top,
    zoom_x,
    zoom_y,
    offset_w,
    offset_h,
    w,
    h,
    original_image,
    debug=False,
    task_id=None,
    index=None,
    temperature=0.0,
    top_p=1.0,
    min_pixels=DEFAULT_MIN_PIXELS,
    max_pixels=DEFAULT_MAX_PIXELS,
):
    """확대(crop+upsample)된 영역 위에서 다시 좌표를 찍고, 원본 이미지 좌표로 역투영한다."""
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
        step_name=f"next_action_regionfocus(idx={index})",
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

    if debug:
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


def next_action_regionfocus_aggregation(qwen_model, instruction, image, points, debug=False, task_id=None):
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
    if debug:
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

    if debug:
        with open(os.path.join(debug_dir, "aggregation_response.txt"), "w") as f:
            f.write(f"Instruction: {instruction}\n\nResponse:\n{response}")

    match = re.search(r"Selected point:\s*(\d+)", response)
    if match:
        selected_idx = int(match.group(1)) - 1
        if 0 <= selected_idx < len(points):
            selected_point = points[selected_idx]
            if debug:
                final_image = plot_points_on_image(
                    vis_image, [selected_point], colors=[(0, 255, 0)], markers=["star"], sizes=[20]
                )
                final_image.save(os.path.join(debug_dir, "RegionFocus_final.png"))
            return selected_point, response

    return points[0], response + "\n(No valid selection found, using first point as fallback.)"


def ground_with_regionfocus(
    qwen_model: QwenVLModel,
    instruction: str,
    image,
    debug: bool = False,
    task_id=None,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> dict:
    """
    베이스라인 Qwen25VLModel.ground_with_regionfocus()의 로컬 모델 버전.
    1) 초기 grounding (gui_grounding.ground) -> 2) 판단 -> 3) 틀렸으면 region_focus로
    재탐색 -> 4) crop/zoom 4가지 비율로 정밀화 -> 5) 후보 종합, 순서 그대로.
    """
    debug_dir = f"./debug/{task_id}" if task_id else "./debug"
    if debug:
        os.makedirs(debug_dir, exist_ok=True)

    overall_start = time.time()

    def _log(msg):
        print(f"[RegionFocus +{time.time() - overall_start:.1f}s] {msg}")

    pil_image = Image.open(image) if isinstance(image, str) else image

    # Step 1: 초기 grounding (local_ground와 동일한 smart_resize 기준으로 원본 크기 재계산)
    _log("Step 1/5: 초기 grounding 시작")
    initial_result = local_ground(
        qwen_model, instruction, pil_image, min_pixels=min_pixels, max_pixels=max_pixels
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
            qwen_model, instruction, original_image, point_px, debug=debug, task_id=task_id
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
            debug=debug,
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
            debug=debug,
            task_id=task_id,
            index=i,
            ratio_x=ratio[0],
            ratio_y=ratio[1],
        )
        zoomed_bytes, zoom_x, zoom_y, offset_w, offset_h = crop_and_upsample(
            (left, top, w, h), original_image, keep_aspect_ratio=True, debug=debug, task_id=task_id, index=i
        )
        action_point, action_response = next_action_regionfocus(
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
            debug=debug,
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
            "point": region_points[0],
            "bbox": None,
            "raw_response": "no valid points found from zoomed regions",
        }

    # Step 5: 후보 종합
    _log(f"Step 5/5: 후보 {len(zoomed_results)}개 종합 시작")
    final_points = [p for p, _ in zoomed_results]
    if len(final_points) > 1:
        best_point, agg_response = next_action_regionfocus_aggregation(
            qwen_model, instruction, original_image, final_points, debug=debug, task_id=task_id
        )
    else:
        best_point, agg_response = zoomed_results[0]
    _log(f"Step 5/5 완료 - 최종 point={best_point}")
    _log(f"총 소요시간 {time.time() - overall_start:.1f}초")

    return {
        "point": [best_point[0] / original_image.width, best_point[1] / original_image.height],
        "bbox": None,
        "regionfocus_applied": True,
        "initial_point": initial_result["point"],
        "initial_correct": is_correct,
        "num_candidates": len(zoomed_results),
        "raw_response": agg_response,
    }


if __name__ == "__main__":
    model = QwenVLModel()
    result = ground_with_regionfocus(model, "Open 버튼을 클릭", "/srv/project/data/processed/images/screenspot_v2/ss_0.jpg", debug=True, task_id="demo")
    print(result)