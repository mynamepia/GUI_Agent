"""
region_focus_hypo2.py

[가설2 실험용 변형] region_focus.py의 복사본 + judge를 YES/NO 이분 결정 대신
0~100 점수로 답하게 하는 버전을 추가한 파일.

가설2: judge를 YES/NO 이분류가 아니라 YES/NO/MODERATE 3-way로 바꾸면
       confidence calibration이 나아질까? 아니면 다 "moderate"로 몰려서
       오히려 문제가 될까?

실험 절차(합의된 대로):
    Phase A (이 파일이 담당) - 아직 YES/NO/MODERATE 판정 루틴을 만들지 않는다.
        judge가 reason은 그대로 내되, ans(YES/NO) 대신 0~100 점수만 내게 하고,
        이 점수를 데이터셋 전체에 대해 뽑아서 "분포"부터 확인한다
        (다 50 근처로 몰리는지, 양극단으로 갈리는지, actual_hit과 상관관계가
        있는지 등). 여기서 threshold를 미리 정해버리면 실험 목적이 무너지므로
        이 파일은 의도적으로 correct/incorrect 판정을 하지 않는다.
    Phase B (다음 단계, 이 파일 밖) - Phase A에서 얻은 점수 분포를 보고 나서
        YES/NO/MODERATE 3-way 판단 루틴(임계값 2개)을 설계한다.

원본 region_focus.py의 함수(judge_inference, ground_with_regionfocus 등)는
전부 그대로 남겨뒀다 - 이 실험이 기존 파이프라인을 건드리지 않고 독립적으로
돌아가야 하고, 나중에 baseline(YES/NO)과 비교할 때도 같은 파일 안에서
바로 대조할 수 있게 하기 위함. 이 파일에서 새로 추가한 것만:
    - _parse_judge_score()          : judge 응답에서 {"reason":..., "score":0~100} 파싱
    - judge_score_inference()       : YES/NO 대신 점수를 내는 judge (judge_inference의 점수판)
    - ground_and_score()            : Step1(초기 grounding) + judge_score_inference까지만
                                       수행하고 RegionFocus 나머지 단계(3~5)는 안 돈다
                                       (Phase A는 점수 분포만 필요하지 전체 파이프라인
                                       결과는 아직 필요 없어서 - CPU/VRAM 여유도 아낄 수 있음)
    - _cli()에 --score_mode/--jsonl/--out 추가 : 데이터셋 전체를 돌려서 점수 분포용
                                       jsonl(id/platform/category/score/reason/actual_hit)을
                                       뽑아내는 벌크 모드.

필요 패키지: 원본 region_focus.py와 동일. 추가로 coord_utils.load_jsonl 사용(--jsonl 모드).
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
    ComputerUseTool,
    build_grounding_messages,
    parse_tool_call,
    ground as local_ground,
    dump_prompt_debug,
)


# ---------------------------------------------------------------------------
# 순수 유틸 (모델 호출 없음) - region_focus.py에서 그대로 가져옴
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

    if debug_image:
        upsampled_filename = f"upsampled_{index}.png" if index is not None else "upsampled.png"
        upsampled.save(os.path.join(debug_dir, upsampled_filename))

    output_buffer = io.BytesIO()
    upsampled.save(output_buffer, format="PNG")
    screenshot_bytes = output_buffer.getvalue()

    return screenshot_bytes, zoom_x, zoom_y, offset_w, offset_h


# ---------------------------------------------------------------------------
# 로컬 모델 추론 헬퍼
# ---------------------------------------------------------------------------
def _generate_with_sampling(
    qwen_model: QwenVLModel,
    messages: list,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    step_name: str = "",
) -> str:
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


def _generate_with_sampling_batch(
    qwen_model: QwenVLModel,
    messages_list: list,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    step_name: str = "",
) -> list:
    """
    _generate_with_sampling()의 배치 버전. messages_list = [messages_1, messages_2, ...]
    (각 messages_i는 단일 샘플 호출에서 쓰던 것과 동일한 [{"role": "user", "content": [...]}]
    형식)를 받아서 generate() 한 번으로 전부 처리하고, 응답 문자열 리스트를 입력 순서
    그대로 반환한다.

    주의(왼쪽 패딩): decoder-only 모델을 배치로 생성할 때는 패딩을 왼쪽에 둬야
    한다 - 오른쪽 패딩이면 실제 토큰 뒤에 패딩이 붙어서 생성이 엉뚱한 위치(패딩
    바로 뒤)에서 시작해버릴 수 있음. 여기서 padding_side="left"를 명시적으로
    맞춰준다. 단일 샘플 호출(_generate_with_sampling)에는 영향 없음 - batch=1이면
    애초에 패딩이 생기지 않으므로.

    trimming(out_ids[len(in_ids):])이 왼쪽 패딩에서도 그대로 맞는 이유: input_ids는
    (batch, max_len)의 사각 텐서라 배치 내 모든 행의 길이가 동일(max_len)하고,
    generate()는 배치 전체에 대해 매 스텝 같은 위치에 새 토큰을 이어붙이므로
    len(in_ids)(=max_len) 이후가 곧 그 행의 새로 생성된 토큰들이다.
    """
    model, processor = qwen_model.model, qwen_model.processor
    processor.tokenizer.padding_side = "left"

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    image_inputs, video_inputs = process_vision_info(messages_list)
    inputs = processor(
        text=texts,
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
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    n_samples = len(messages_list)
    label = f"[{step_name}] " if step_name else "[generate_batch] "
    print(f"{label}완료 - {elapsed:.1f}초 (배치 {n_samples}개, 평균 {elapsed / max(n_samples,1):.2f}초/샘플)")

    return output_texts


# ---------------------------------------------------------------------------
# 원본 judge (YES/NO 이분) - 그대로 보존, baseline 대조용
# ---------------------------------------------------------------------------
def _parse_judge_verdict(response: str):
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            ans = str(obj.get("ans", "")).strip().upper()
            if ans in ("YES", "NO"):
                return ans, obj.get("reason")
        except (json.JSONDecodeError, AttributeError):
            pass

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
    """원본(YES/NO) judge. 가설2 대조군으로 그대로 남겨둠."""
    pil_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image).copy()

    highlighted_image = plot_points_on_image(
        pil_image, [point], colors=[(255, 0, 255, 128)], markers=["star"], sizes=[12]
    )

    judge_prompt = (
        f'Instruction: "{instruction}"\n'
        f"A pink star marks a candidate click point. The star may only partially cover the "
        f"target, and that still counts as correct.\n\n"
        f'Reply with ONLY this JSON: {{"reason": "<short reason>", "ans": "YES/NO"}}\n'
        f"Think through the reason first, then decide. Be strict: the star must precisely "
        f"match the correct element. If there is any real doubt, or the star seems close but "
        f"not exactly on the target, answer NO."
    )

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
        is_correct = False
    else:
        is_correct = ans == "YES"

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


# ---------------------------------------------------------------------------
# [가설2 신규] 점수 기반 judge - Phase A: YES/NO 없이 점수 분포만 확인
# ---------------------------------------------------------------------------
def _parse_judge_score(response: str):
    """
    judge_score_inference 응답에서 {"reason": "...", "score": <0~100>} JSON을 파싱한다.

    가설2 Phase A는 YES/NO 이분 결정 자체가 실험 대상이 아니라 "점수 분포 관찰"이
    목적이라, 여기서는 threshold 판단을 절대 하지 않고 점수(및 reason)만 그대로
    반환한다 - 3-way(YES/NO/MODERATE) 판단 루틴은 이 분포를 보고 난 뒤 Phase B에서
    별도로 설계한다.

    Returns: (score: float|None, reason: str|None) - 완전히 파싱 실패하면 (None, None).
    """
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            score = float(obj.get("score"))
            score = max(0.0, min(100.0, score))
            return score, obj.get("reason")
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            pass

    # 폴백: JSON 강제가 실패했을 때, 응답 텍스트에서 "score" 근처 숫자를 정규식으로 탐색.
    m = re.search(r'"?score"?\s*[:\-]?\s*(\d{1,3})', response, re.IGNORECASE)
    if m:
        score = float(m.group(1))
        return max(0.0, min(100.0, score)), None

    return None, None


def judge_score_inference(
    qwen_model, instruction, image, point,
    debug_image=False, debug_text=False, task_id=None,
):
    """
    [가설2 실험용] judge_inference()의 YES/NO 이분 결정을 없애고, 0~100 점수를
    내게 하는 버전. reason은 원본과 동일하게 먼저 서술하게 유지(판단 근거를
    사후 정당화 대신 먼저 풀어놓는 v4 프롬프트 구조는 그대로 가져옴).

    원본과의 핵심 차이: 여기서는 correct/incorrect를 이 함수 안에서 결정하지
    않는다. debug_mode에 따른 게이팅(원본의 "incorrect일 때만 저장")도 없앴다 -
    모든 샘플의 점수가 분포 분석에 필요하므로 golden 판정 없이는 어떤 샘플을
    스킵할지조차 정할 수 없기 때문에, debug_image/debug_text가 켜져 있으면
    항상 저장한다.

    Returns: (score: float|None, reason: str|None, raw_response: str)
    """
    pil_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image).copy()

    highlighted_image = plot_points_on_image(
        pil_image, [point], colors=[(255, 0, 255, 128)], markers=["star"], sizes=[12]
    )

    judge_score_prompt = (
        f'Instruction: "{instruction}"\n'
        f"A pink star marks a candidate click point. The star may only partially cover the "
        f"target, and that still counts as correct.\n\n"
        f'Reply with ONLY this JSON: {{"reason": "<short reason>", "score": <integer 0-100>}}\n'
        f"Think through the reason first, then give a confidence score from 0 (definitely the "
        f"wrong element) to 100 (definitely the correct element) for how precisely the star "
        f"matches the target described in the instruction."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": highlighted_image},
                {"type": "text", "text": judge_score_prompt},
            ],
        }
    ]

    response = _generate_with_sampling(
        qwen_model, messages, max_new_tokens=256, temperature=0.0, step_name="judge_score_inference"
    )

    score, parsed_reason = _parse_judge_score(response)

    if debug_image:
        debug_dir = f"./debug/{task_id}" if task_id else "./debug"
        os.makedirs(debug_dir, exist_ok=True)
        highlighted_image.save(os.path.join(debug_dir, "initial_point_highlighted.png"))

    if debug_text:
        extra = f"Point: {point}\nParsed score: {score}"
        if parsed_reason:
            extra += f"\nParsed reason: {parsed_reason}"
        dump_prompt_debug(messages, response, task_id=task_id, step_name="judge_score_inference", extra=extra)

    return score, parsed_reason, response


def judge_score_inference_batch(
    qwen_model, items, debug_image=False, debug_text=False, task_id=None,
):
    """
    judge_score_inference()의 배치 버전. items = [(instruction, image, point), ...]를
    받아서 generate() 한 번으로 전부 채점하고 [(score, reason, raw_response), ...]를
    입력 순서 그대로 반환한다.

    grounding(Step 1)은 여기서 안 건드림 - local_ground()는 기존에 검증된
    프롬프트/포맷을 그대로 쓰는 게 안전해서, 배치화는 지금 우리가 새로 짠 judge
    점수 단계에만 적용했다. Step 1까지 배치로 묶으려면 gui_grounding.py의
    grounding 프롬프트를 여기서 다시 구현해야 해서 위험 부담이 커짐 - 필요해지면
    별도로 진행.
    """
    messages_list = []
    highlighted_images = []
    for instruction, image, point in items:
        pil_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image).copy()
        highlighted_image = plot_points_on_image(
            pil_image, [point], colors=[(255, 0, 255, 128)], markers=["star"], sizes=[12]
        )
        highlighted_images.append(highlighted_image)

        judge_score_prompt = (
            f'Instruction: "{instruction}"\n'
            f"A pink star marks a candidate click point. The star may only partially cover the "
            f"target, and that still counts as correct.\n\n"
            f'Reply with ONLY this JSON: {{"reason": "<short reason>", "score": <integer 0-100>}}\n'
            f"Think through the reason first, then give a confidence score from 0 (definitely the "
            f"wrong element) to 100 (definitely the correct element) for how precisely the star "
            f"matches the target described in the instruction."
        )
        messages_list.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": highlighted_image},
                        {"type": "text", "text": judge_score_prompt},
                    ],
                }
            ]
        )

    responses = _generate_with_sampling_batch(
        qwen_model, messages_list, max_new_tokens=256, temperature=0.0,
        step_name=f"judge_score_inference_batch(n={len(items)})",
    )

    results = []
    for i, response in enumerate(responses):
        score, parsed_reason = _parse_judge_score(response)
        point = items[i][2]

        if debug_image:
            debug_dir = f"./debug/{task_id}" if task_id else "./debug"
            os.makedirs(debug_dir, exist_ok=True)
            highlighted_images[i].save(os.path.join(debug_dir, f"initial_point_highlighted_{i}.png"))

        if debug_text:
            extra = f"Point: {point}\nParsed score: {score}"
            if parsed_reason:
                extra += f"\nParsed reason: {parsed_reason}"
            dump_prompt_debug(
                messages_list[i], response, task_id=task_id,
                step_name="judge_score_inference_batch", index=i, extra=extra,
            )

        results.append((score, parsed_reason, response))

    return results


def ground_and_score(
    qwen_model: QwenVLModel,
    instruction: str,
    image,
    debug_image: bool = False,
    debug_text: bool = False,
    task_id=None,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> dict:
    """
    [가설2 실험 Phase A 전용] 초기 grounding(Step 1) + 점수 judge(Step 2')까지만 수행하고,
    RegionFocus의 나머지 단계(재탐색/crop-zoom/aggregation, 원본의 Step 3~5)는 아예
    돌리지 않는다.

    이유: Phase A의 목적은 "지금 judge가 내는 점수의 분포가 어떻게 생겼는지"를 보는
    것뿐이라 - 아직 없는 임계값으로 RegionFocus를 트리거할지 말지부터 정해버리면
    실험 목적이 훼손된다. 그리고 어차피 안 쓸 Step3~5(generate() 5~6회 추가 호출)를
    안 도니까, 지금 VRAM/CPU가 빠듯한 상황에서 점수 분포만 뽑는 목적에는 계산량도
    훨씬 가볍다.
    """
    pil_image = Image.open(image) if isinstance(image, str) else image

    initial_result = local_ground(
        qwen_model, instruction, pil_image, min_pixels=min_pixels, max_pixels=max_pixels,
        debug_text=debug_text, task_id=task_id,
    )
    resized_height, resized_width = smart_resize(
        pil_image.height, pil_image.width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    original_image = pil_image.resize((resized_width, resized_height))

    if not initial_result["point"]:
        return {
            "result": initial_result.get("result", "wrong_format"),
            "point": None,
            "point_px": None,
            "score": None,
            "reason": None,
            "raw_judge_response": None,
        }

    point_px = [
        round(initial_result["point"][0] * original_image.width),
        round(initial_result["point"][1] * original_image.height),
    ]
    score, reason, judge_response = judge_score_inference(
        qwen_model, instruction, original_image, point_px,
        debug_image=debug_image, debug_text=debug_text, task_id=task_id,
    )

    return {
        "result": initial_result.get("result", "positive"),
        "point": initial_result["point"],
        "point_px": point_px,
        "score": score,
        "reason": reason,
        "raw_judge_response": judge_response,
    }


def _compute_actual_hit(point_frac, raw_width, raw_height, gt_bbox):
    """point(0~1 비율) + 원본 raw 이미지 크기 + gt_bbox(raw px)로 실제 hit 여부 계산."""
    if gt_bbox is None or point_frac is None:
        return None
    x1, y1, x2, y2 = gt_bbox
    px = point_frac[0] * raw_width
    py = point_frac[1] * raw_height
    return bool(x1 <= px <= x2 and y1 <= py <= y2)


def _ground_initial(qwen_model, instruction, image, debug_text, task_id, min_pixels, max_pixels):
    """
    ground_and_score()의 Step 1(초기 grounding)만 떼어낸 헬퍼 - 배치 모드에서 judge
    채점 전에 각 샘플의 점을 먼저 구해야 해서 재사용한다. local_ground()는 배치화
    안 하고 그대로 순차 호출한다(이유는 judge_score_inference_batch 독스트링 참고).

    Returns: (original_image: PIL.Image|None, point_px: [x,y]|None, initial_result: dict)
    """
    pil_image = Image.open(image) if isinstance(image, str) else image
    initial_result = local_ground(
        qwen_model, instruction, pil_image, min_pixels=min_pixels, max_pixels=max_pixels,
        debug_text=debug_text, task_id=task_id,
    )
    resized_height, resized_width = smart_resize(
        pil_image.height, pil_image.width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    original_image = pil_image.resize((resized_width, resized_height))
    if not initial_result["point"]:
        return None, None, initial_result
    point_px = [
        round(initial_result["point"][0] * original_image.width),
        round(initial_result["point"][1] * original_image.height),
    ]
    return original_image, point_px, initial_result


# ---------------------------------------------------------------------------
# 원본 RegionFocus 전체 파이프라인 (region_focus / next_action_regionfocus /
# next_action_regionfocus_aggregation / ground_with_regionfocus) - 그대로 보존.
# 가설2 Phase A에서는 쓰지 않지만, 나중에 Phase B에서 3-way 판정을 실제
# RegionFocus 트리거 조건으로 연결할 때 참고/재사용하기 위해 남겨둠.
# ---------------------------------------------------------------------------
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

    if debug_text:
        dump_prompt_debug(
            messages, response, task_id=task_id, step_name=f"region_focus_temp{temperature}",
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
    debug_image=False,
    debug_text=False,
    task_id=None,
    index=None,
    temperature=0.0,
    top_p=1.0,
    min_pixels=DEFAULT_MIN_PIXELS,
    max_pixels=DEFAULT_MAX_PIXELS,
):
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
        step_name=f"next_action_regionfocus(idx={index})",
    )

    if debug_text:
        dump_prompt_debug(
            messages, response, task_id=task_id, step_name="next_action_regionfocus", index=index,
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
    qwen_model: QwenVLModel,
    instruction: str,
    image,
    debug_image: bool = False,
    debug_text: bool = False,
    debug_mode: str = "always",
    task_id=None,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> dict:
    """원본(YES/NO judge) 전체 파이프라인. 가설2 Phase A에서는 안 쓰지만 대조/재사용을 위해 보존."""
    debug_dir = f"./debug/{task_id}" if task_id else "./debug"
    if debug_image or debug_text:
        os.makedirs(debug_dir, exist_ok=True)

    overall_start = time.time()

    def _log(msg):
        print(f"[RegionFocus +{time.time() - overall_start:.1f}s] {msg}")

    pil_image = Image.open(image) if isinstance(image, str) else image

    _log("Step 1/5: 초기 grounding 시작")
    initial_result = local_ground(
        qwen_model, instruction, pil_image, min_pixels=min_pixels, max_pixels=max_pixels,
        debug_text=debug_text, task_id=task_id,
    )
    resized_height, resized_width = smart_resize(
        pil_image.height, pil_image.width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    original_image = pil_image.resize((resized_width, resized_height))
    _log(f"Step 1/5 완료 - point={initial_result['point']}")

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
    가설2 실험용 CLI.

    기본(--score_mode 없음): 원본과 동일하게 ground_with_regionfocus(YES/NO judge)를 돌린다.
    --score_mode: ground_and_score(점수 judge, Phase A)를 돌린다 - 단일 이미지 하나만 볼 때.
    --score_mode --jsonl <path>: 데이터셋 전체를 돌면서 점수를 뽑아 --out에 jsonl로 저장
        (id/platform/category/point/score/reason/actual_hit) - 분포 분석은 이 결과를
        가지고 별도로 한다 (pandas/matplotlib 등, 이 파일 밖 작업).
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="스크린샷 이미지 경로 (단일 이미지 모드)")
    ap.add_argument("--instruction", default=None, help="grounding할 지시문 (단일 이미지 모드)")
    ap.add_argument("--model_id", default=None, help="베이스 모델 id (기본값: qwen.py의 MODEL_ID)")
    ap.add_argument("--adapter_dir", default=None,
                    help="LoRA 어댑터 디렉토리 (train.py --output_dir로 저장된 checkpoint-XXX 폴더)")
    ap.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--debug_image", action="store_true")
    ap.add_argument("--debug_text", action="store_true")
    ap.add_argument("--debug_mode", choices=["always", "incorrect"], default="always",
                    help="--score_mode 없을 때(원본 YES/NO judge)만 적용됨")
    ap.add_argument("--task_id", default="demo")

    ap.add_argument("--score_mode", action="store_true",
                    help="가설2: YES/NO 대신 점수(0~100)를 내는 judge_score_inference 사용")
    ap.add_argument("--jsonl", default=None,
                    help="[--score_mode 전용] 데이터셋 jsonl 경로 - 지정하면 --image/--instruction 대신 전체를 일괄 처리")
    ap.add_argument("--out", default=None,
                    help="[--jsonl 모드] 결과 jsonl 저장 경로 (id/platform/category/score/reason/actual_hit)")
    ap.add_argument("--limit", type=int, default=None, help="[--jsonl 모드] 앞에서부터 N개만 처리")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="[--jsonl 모드] judge 채점 단계를 몇 개씩 묶어서 generate() 한 번에 처리할지"
                         "(grounding/Step1은 배치화 안 함, 이유는 judge_score_inference_batch 참고). "
                         "VRAM이 빠듯하면 1~4 정도로 작게 시작해서 늘려볼 것.")
    ap.add_argument("--rejudge_from", default=None,
                    help="[--jsonl 모드 전용, base vs LoRA judge 비교용] 이전에 --score_mode --jsonl로 "
                         "뽑아둔 결과 jsonl 경로(예: hypo2_scores.jsonl). 지정하면 Step1(grounding)을 "
                         "다시 안 돌리고 그 파일에 저장된 point(id로 join)를 그대로 재사용해서, "
                         "지금 이 실행의 모델(--adapter_dir 유무)로 judge만 다시 매긴다. grounding은 "
                         "고정한 채 judge 모델만 바꿔서 비교하기 위한 용도(base 모델로 이 플래그를 켜고 "
                         "돌리면 'LoRA가 찍은 점을 base가 어떻게 채점하는지' 비교 가능).")
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

    if args.score_mode and args.jsonl:
        # 벌크 모드: 데이터셋 전체를 돌면서 점수 분포용 jsonl을 뽑는다.
        # Step 1(grounding)은 샘플마다 순차 호출, judge 채점만 --batch_size개씩 묶어서
        # generate() 한 번으로 처리한다 (judge_score_inference_batch 참고).
        from coord_utils import load_jsonl

        records = load_jsonl(args.jsonl)
        if args.limit:
            records = records[: args.limit]

        # --rejudge_from: 이전 실행에서 나온 point를 그대로 재사용 (Step1/grounding 재실행 안 함).
        # base vs LoRA judge 비교처럼 "grounding은 고정하고 judge 모델만 바꿔서" 보고 싶을 때 사용.
        rejudge_point_map = None
        if args.rejudge_from:
            rejudge_point_map = {}
            with open(args.rejudge_from, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prev_row = json.loads(line)
                        rejudge_point_map[prev_row.get("id")] = prev_row.get("point")
            print(f"[rejudge_from] {args.rejudge_from}에서 point {len(rejudge_point_map)}개 로드 - "
                  f"이번 실행은 Step1(grounding)을 다시 안 돌리고 이 point들을 그대로 재사용함")

        batch_size = max(1, args.batch_size)
        out_f = open(args.out, "w", encoding="utf-8") if args.out else None
        n_score_none = 0
        n_done = 0
        try:
            for chunk_start in range(0, len(records), batch_size):
                chunk = records[chunk_start: chunk_start + batch_size]
                step1_label = "point 재사용(rejudge_from)" if rejudge_point_map is not None else "Step1(grounding) 시작"
                print(f"[chunk {chunk_start + 1}~{chunk_start + len(chunk)}/{len(records)}] {step1_label}")

                # --- Step 1: 청크 내 각 샘플의 초기 grounding point ---
                # rejudge_from이 없으면 순차로 새로 grounding, 있으면 이전 결과의 point를 그대로 씀
                # (이 경우 generate() 호출 없이 이미지 열고 smart_resize만 해서 원본 좌표계를 맞춘다 -
                # --min_pixels/--max_pixels를 원래 실행과 동일하게 줘야 좌표계가 정확히 일치함).
                chunk_meta = []  # (rec, raw_w, raw_h, original_image, point_px)
                for rec in chunk:
                    rid = rec.get("id")
                    raw_w, raw_h = rec.get("resolution", [None, None])
                    if raw_w is None:
                        raw_img_for_size = Image.open(rec["image_path"])
                        raw_w, raw_h = raw_img_for_size.size

                    if rejudge_point_map is not None:
                        prev_point = rejudge_point_map.get(rid)
                        if prev_point is None:
                            original_image, point_px = None, None
                        else:
                            pil_image = Image.open(rec["image_path"])
                            resized_h, resized_w = smart_resize(
                                pil_image.height, pil_image.width,
                                min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                            )
                            original_image = pil_image.resize((resized_w, resized_h))
                            point_px = [
                                round(prev_point[0] * original_image.width),
                                round(prev_point[1] * original_image.height),
                            ]
                    else:
                        original_image, point_px, _initial = _ground_initial(
                            model, rec["instruction"], rec["image_path"], args.debug_text,
                            rid if rid is not None else args.task_id, args.min_pixels, args.max_pixels,
                        )
                    chunk_meta.append((rec, raw_w, raw_h, original_image, point_px))

                # --- Step 2': 유효한 point가 나온 샘플만 모아서 judge 배치 채점 ---
                score_items = []
                score_idx_map = []
                for j, (rec, raw_w, raw_h, original_image, point_px) in enumerate(chunk_meta):
                    if point_px is not None:
                        score_items.append((rec["instruction"], original_image, point_px))
                        score_idx_map.append(j)

                score_results = {}
                if score_items:
                    print(f"[chunk {chunk_start + 1}~{chunk_start + len(chunk)}] "
                          f"judge 배치 채점 시작 (유효 point {len(score_items)}/{len(chunk)}개)")
                    batch_out = judge_score_inference_batch(
                        model, score_items,
                        debug_image=args.debug_image, debug_text=args.debug_text,
                        task_id=args.task_id,
                    )
                    for k, j in enumerate(score_idx_map):
                        score_results[j] = batch_out[k]

                # --- 결과 기록 ---
                for j, (rec, raw_w, raw_h, original_image, point_px) in enumerate(chunk_meta):
                    rid = rec.get("id")
                    point = None if point_px is None else [
                        point_px[0] / original_image.width, point_px[1] / original_image.height,
                    ]
                    score, reason, _raw = score_results.get(j, (None, None, None))
                    if score is None:
                        n_score_none += 1

                    # 주의: 데이터셋 원본 jsonl의 bbox 필드명은 "bbox"임 ("gt_bbox"가 아님 -
                    # eval_regionfocus.py의 score_regionfocus_result()도 rec.get("bbox")를 읽어서
                    # 자기 출력에서만 "gt_bbox"로 이름을 바꿔 저장함). 여기서 rec.get("gt_bbox")로
                    # 읽으면 항상 None이 나오는 버그가 났었어서 고침.
                    actual_hit = _compute_actual_hit(point, raw_w, raw_h, rec.get("bbox"))

                    row = {
                        "id": rid,
                        "platform": rec.get("platform"),
                        "category": rec.get("category"),
                        "point": point,
                        "score": score,
                        "reason": reason,
                        "actual_hit": actual_hit,
                    }
                    n_done += 1
                    print(f"  [{n_done}/{len(records)}] id={rid} platform={rec.get('platform')} "
                          f"score={score} actual_hit={actual_hit}")

                    if out_f:
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
        finally:
            if out_f:
                out_f.close()

        print(f"\n완료: {len(records)}개 중 점수 파싱 실패 {n_score_none}개")
        return

    if args.score_mode:
        if not args.image or not args.instruction:
            raise SystemExit("--score_mode 단일 이미지 모드에는 --image/--instruction이 필요함 (또는 --jsonl 사용)")
        result = ground_and_score(
            model, args.instruction, args.image,
            debug_image=args.debug_image, debug_text=args.debug_text,
            task_id=args.task_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        )
        print(result)
        return

    # 원본(YES/NO) 전체 파이프라인
    if not args.image or not args.instruction:
        raise SystemExit("--image/--instruction이 필요함")
    result = ground_with_regionfocus(
        model, args.instruction, args.image,
        debug_image=args.debug_image, debug_text=args.debug_text, debug_mode=args.debug_mode,
        task_id=args.task_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    print(result)


if __name__ == "__main__":
    _cli()