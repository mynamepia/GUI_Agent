"""
gui_actor_region_focus.py

region_focus.py의 RegionFocus 알고리즘(초기 grounding -> judge 판단 -> 재탐색 ->
crop/zoom 4비율 정밀화 -> 후보 종합)을 gui_actor_grounding.GUIActorModel
(microsoft/GUI-Actor-3B-Qwen2.5-VL) 위에서 재현한 모듈.

[LoRA 버전(region_focus.py)과 구조적으로 다른 점]

1) 좌표 프레임이 훨씬 단순하다. LoRA 경로(gui_grounding.py/qwen.py)는 우리가 직접
   smart_resize로 리사이즈한 이미지를 넣고, 모델이 "0~1000 정규화, 그 리사이즈된 이미지
   기준" 좌표로 답하기 때문에, crop/zoom 뒤에 다시 smart_resize를 태우고 그 배율만큼
   zoom_x/zoom_y를 보정하는 코드가 필요했다(region_focus.py의 next_action_regionfocus
   참고). GUI-Actor는 gui_actor.inference()가 내부적으로 어떤 해상도로 리사이즈하든
   상관없이 항상 "원래 넣어준 이미지 기준 0~1 정규화" 좌표를 돌려준다(HF 모델 카드
   사용 예시로 확인 - ground-truth bbox와 predicted point가 같은 프레임에 있음). 그래서
   여기서는 매 단계 넣어준 이미지의 width/height로 나누기만 하면 되고, 별도 smart_resize
   보정이 필요 없다.

2) Step3(재탐색)가 별도 모델 호출이 없다. LoRA 버전은 "재탐색"을 위해 temperature를
   0.0/0.3/0.5/0.7/0.9로 바꿔가며 최대 5번 다시 물어봐야 했다(포인트를 하나만 내는
   모델이라 다양성을 얻으려면 반복 샘플링이 필요). GUI-Actor는 pointer head가 한 번의
   forward pass에서 attention 상위 K개 후보(topk_points)를 동시에 낸다("no extra
   inference cost"로 다중 후보를 낸다는 게 이 모델의 핵심 특징) - 그래서 Step1에서 이미
   topk=4 정도로 뽑아두면, Step3는 그중 1순위가 오답으로 판정났을 때 그냥 2순위
   후보를 그대로 재사용하면 된다. 추가 모델 호출이 없다.

3) judge_inference/aggregation(자유 텍스트 YES/NO, "Selected point: #")은 좌표
   추출과 무관한 일반 텍스트 생성이라, gui_actor.inference()(포인터 전용 경로) 대신
   GUIActorModel.generate()(표준 HF generate, gui_actor_grounding.py 참고)를 그대로 쓴다.
   판정 프롬프트/파싱 로직 자체(judge_prompt 문구, JSON 강제 포맷, aggregation 선택 방식)는
   region_focus.py의 것과 100% 동일하게 재사용한다 - 이 부분은 모델이 뭐든 상관없는
   순수 텍스트 QA라 새로 설계할 이유가 없었다.

필요 설치: gui_actor_grounding.py와 동일 (GUI-Actor 리포 pip install -e .).
"""

import io
import os
import sys as _sys
import time

from PIL import Image

# (v3 이동 - agent-actor3b/ 폴더로 옮기면서 추가) 이 파일은 vlm_agent/agent-actor3b/에 있고,
# 아래에서 import하는 gui_grounding.py/region_focus.py는 vlm_agent/ 루트에 있다 - 같은 폴더가
# 아니라서 파이썬이 자동으로 찾아주지 않는다. agent_loop.py와 동일한 부트스트랩 패턴(qwen.py가
# 있는 폴더를 찾아 sys.path에 추가)을 그대로 써서, 이 파일을 직접 실행(--cli)하든 다른 곳에서
# import하든 vlm_agent/ 루트가 항상 sys.path에 잡히게 한다.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "vlm_agent")):
    _candidate = os.path.abspath(_candidate)
    if os.path.isfile(os.path.join(_candidate, "qwen.py")):
        if _candidate not in _sys.path:
            _sys.path.insert(0, _candidate)
        break

from gui_grounding import dump_prompt_debug
from region_focus import (
    calculate_crop_region,
    crop_and_upsample,
    plot_points_on_image,
    _parse_judge_verdict,
)


# ---------------------------------------------------------------------------
# Step 2: judge (자유 텍스트 YES/NO) - region_focus.judge_inference()와 프롬프트/파싱 동일,
# 모델 호출만 GUIActorModel.generate()로 교체
# ---------------------------------------------------------------------------
def judge_inference_gui_actor(
    gui_actor_model, instruction, image, point,
    debug_image=False, debug_text=False, debug_mode="always", task_id=None,
):
    pil_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image).copy()

    highlighted_image = plot_points_on_image(
        pil_image, [point], colors=[(255, 0, 255, 128)], markers=["star"], sizes=[12]
    )

    # region_focus.judge_inference()와 완전히 동일한 프롬프트 - 판정 기준을 두 백엔드 사이에서
    # 다르게 둘 이유가 없어서 그대로 재사용.
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

    _t0 = time.time()
    response = gui_actor_model.generate(messages, max_new_tokens=256, temperature=0.0)
    print(f"[judge_inference_gui_actor] generate() 완료 - {time.time() - _t0:.1f}초")

    ans, parsed_reason = _parse_judge_verdict(response)
    is_correct = False if ans is None else (ans == "YES")

    should_dump = debug_mode == "always" or (debug_mode == "incorrect" and not is_correct)

    if debug_image and should_dump:
        debug_dir = f"./debug/{task_id}" if task_id else "./debug"
        os.makedirs(debug_dir, exist_ok=True)
        highlighted_image.save(os.path.join(debug_dir, "initial_point_highlighted.png"))

    if debug_text and should_dump:
        extra = f"Point: {point}\nParsed ans: {ans}\nJudgment: {'CORRECT' if is_correct else 'INCORRECT'}"
        if parsed_reason:
            extra += f"\nParsed reason: {parsed_reason}"
        dump_prompt_debug(
            messages, response, task_id=task_id, step_name="judge_inference_gui_actor", extra=extra,
        )

    return is_correct, response


# ---------------------------------------------------------------------------
# Step 4: crop/zoom 후 재grounding - GUI-Actor는 좌표가 항상 "넣어준 이미지 기준 0~1"이라
# smart_resize 보정 없이 바로 projection만 하면 된다.
# ---------------------------------------------------------------------------
def next_action_regionfocus_gui_actor(
    gui_actor_model, instruction, zoomed_img_bytes, left, top, zoom_x, zoom_y, w, h,
    original_image, debug_image=False, debug_text=False, task_id=None, index=None,
):
    zoomed_img = Image.open(io.BytesIO(zoomed_img_bytes))

    _t0 = time.time()
    pred = gui_actor_model.predict_point(instruction, zoomed_img, topk=1)
    print(f"[next_action_regionfocus_gui_actor(idx={index})] predict_point() 완료 - {time.time() - _t0:.1f}초")

    if debug_text:
        dump_prompt_debug(
            [{"role": "user", "content": [{"type": "text", "text": instruction}]}],
            str(pred), task_id=task_id, step_name="next_action_regionfocus_gui_actor", index=index,
        )

    topk_points = pred.get("topk_points") or []
    if not topk_points:
        return None, str(pred)

    x_frac, y_frac = topk_points[0]
    x_upsampled = float(x_frac) * zoomed_img.width
    y_upsampled = float(y_frac) * zoomed_img.height

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

    return projected_point, str(pred)


# ---------------------------------------------------------------------------
# Step 5: 후보 종합 (자유 텍스트 "Selected point: #") - region_focus.
# next_action_regionfocus_aggregation()과 프롬프트 동일, 모델 호출만 교체.
# ---------------------------------------------------------------------------
def aggregation_gui_actor(
    gui_actor_model, instruction, image, points,
    debug_image=False, debug_text=False, task_id=None,
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
        vis_image, points,
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

    _t0 = time.time()
    response = gui_actor_model.generate(messages, max_new_tokens=256, temperature=0.0)
    print(f"[aggregation_gui_actor] generate() 완료 - {time.time() - _t0:.1f}초")

    import re
    match = re.search(r"Selected point:\s*(\d+)", response)

    if debug_text:
        dump_prompt_debug(
            messages, response, task_id=task_id, step_name="aggregation_gui_actor",
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


# ---------------------------------------------------------------------------
# 전체 파이프라인
# ---------------------------------------------------------------------------
def ground_with_regionfocus_gui_actor(
    gui_actor_model, instruction, image,
    debug_image=False, debug_text=False, debug_mode="always", task_id=None,
    topk=4,
    **_ignored,
    # eval_webvoyager_v3.py의 _build_click_ground_fn(off 경로 등)이 min_pixels/max_pixels/
    # max_new_tokens 같은, 여기선 안 쓰는 kwargs를 관성적으로 넘길 수 있어 조용히 무시한다.
    # (GUI-Actor의 해상도는 GUIActorModel 생성 시점의 min_pixels/max_pixels로 이미 고정됨 -
    # gui_actor_grounding.py 참고.)
) -> dict:
    """
    gui_grounding.ground()/region_focus.ground_with_regionfocus()와 동일한 반환 스키마:
        {"result": "positive"|"wrong_format", "point": [x_norm, y_norm]|None, "raw_response": str, ...}

    1) 초기 grounding(topk 후보 동시 추출) -> 2) judge 판단 -> 3) 오답이면 topk 2순위 후보
    재사용(추가 모델 호출 없음) -> 4) crop/zoom 4비율 정밀화 -> 5) 후보 종합. region_focus.py의
    5단계와 순서는 같지만 Step1/3의 구현이 다르다(모듈 docstring 참고).
    """
    debug_dir = f"./debug/{task_id}" if task_id else "./debug"
    if debug_image or debug_text:
        os.makedirs(debug_dir, exist_ok=True)

    overall_start = time.time()

    def _log(msg):
        print(f"[RegionFocus(GUI-Actor) +{time.time() - overall_start:.1f}s] {msg}")

    pil_image = Image.open(image) if isinstance(image, str) else image
    # GUI-Actor는 내부적으로 어떤 해상도로 리사이즈하든 "넣어준 이미지 기준 0~1" 좌표를
    # 돌려주므로, LoRA 경로처럼 별도 smart_resize로 원본을 미리 맞출 필요가 없다 - 원본을
    # 그대로 이후 모든 단계(judge 오버레이, crop 기준, 최종 좌표 프레임)의 기준으로 쓴다.
    original_image = pil_image

    # Step 1: 초기 grounding - topk 후보를 한 번의 forward pass로 같이 뽑는다.
    _log(f"Step 1/5: 초기 grounding 시작(topk={topk} 후보 동시 추출)")
    _t0 = time.time()
    pred = gui_actor_model.predict_point(instruction, original_image, topk=topk)
    print(f"[ground_with_regionfocus_gui_actor] Step1 predict_point() 완료 - {time.time() - _t0:.1f}초")

    if debug_text:
        dump_prompt_debug(
            [{"role": "user", "content": [{"type": "text", "text": instruction}]}],
            str(pred), task_id=task_id, step_name="ground_gui_actor_step1",
        )

    topk_points = pred.get("topk_points") or []
    if not topk_points:
        _log("Step 1/5 실패 - 유효한 point를 못 찾음")
        return {"result": "wrong_format", "point": None, "raw_response": str(pred)}

    def _clamp01(p):
        return [max(0.0, min(1.0, float(p[0]))), max(0.0, min(1.0, float(p[1])))]

    initial_point = _clamp01(topk_points[0])
    initial_result = {"result": "positive", "point": initial_point, "raw_response": str(pred)}
    _log(f"Step 1/5 완료 - point={initial_point} (후보 {len(topk_points)}개)")

    # Step 2: 초기 grounding 판단
    point_px = [
        round(initial_point[0] * original_image.width),
        round(initial_point[1] * original_image.height),
    ]
    _log("Step 2/5: 초기 grounding 판단(judge_inference_gui_actor) 시작")
    is_correct, judge_response = judge_inference_gui_actor(
        gui_actor_model, instruction, original_image, point_px,
        debug_image=debug_image, debug_text=debug_text, debug_mode=debug_mode, task_id=task_id,
    )
    _log(f"Step 2/5 완료 - {'정답, 여기서 종료' if is_correct else '오답, RegionFocus 진행'}")
    if is_correct:
        _log(f"총 소요시간 {time.time() - overall_start:.1f}초")
        return initial_result

    # Step 3: topk 2순위 이후 후보를 그대로 재사용 (추가 모델 호출 없음 - 모듈 docstring 참고)
    region_points = [_clamp01(p) for p in topk_points[1:]]
    if not region_points:
        _log("Step 3/5 실패 - topk 후보가 1개뿐이라 재탐색 불가, 초기 결과 반환")
        _log(f"총 소요시간 {time.time() - overall_start:.1f}초")
        return initial_result
    point = region_points[0]
    _log(f"Step 3/5 완료 - topk 2순위 후보 재사용 point={point}")

    # Step 4: crop/zoom 4비율 정밀화
    zoomed_results = []
    ratio_list = [[0.5, 0.5], [0.3, 0.3], [0.4, 0.8], [0.8, 0.4]]
    for i, ratio in enumerate(ratio_list):
        _log(f"Step 4/5: crop/zoom {i+1}/{len(ratio_list)} (ratio={ratio}) 시작")
        left, top, w, h = calculate_crop_region(
            [round(point[0] * original_image.width), round(point[1] * original_image.height)],
            original_image, debug_image=debug_image, task_id=task_id, index=i,
            ratio_x=ratio[0], ratio_y=ratio[1],
        )
        zoomed_bytes, zoom_x, zoom_y, offset_w, offset_h = crop_and_upsample(
            (left, top, w, h), original_image, keep_aspect_ratio=True,
            debug_image=debug_image, task_id=task_id, index=i,
        )
        action_point, action_response = next_action_regionfocus_gui_actor(
            gui_actor_model, instruction, zoomed_bytes, left, top, zoom_x, zoom_y, w, h,
            original_image, debug_image=debug_image, debug_text=debug_text, task_id=task_id, index=i,
        )
        if action_point:
            zoomed_results.append((action_point, action_response))
            _log(f"Step 4/5: crop/zoom {i+1}/{len(ratio_list)} 완료 - point={action_point}")
        else:
            _log(f"Step 4/5: crop/zoom {i+1}/{len(ratio_list)} 실패 (유효한 point 없음)")

    if not zoomed_results:
        _log("Step 4/5 전부 실패 - 후보 없음, 초기 결과로 대체")
        _log(f"총 소요시간 {time.time() - overall_start:.1f}초")
        return initial_result

    # Step 5: 후보 종합
    _log(f"Step 5/5: 후보 {len(zoomed_results)}개 종합 시작")
    final_points = [p for p, _ in zoomed_results]
    if len(final_points) > 1:
        best_point, agg_response = aggregation_gui_actor(
            gui_actor_model, instruction, original_image, final_points,
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
    """region_focus.py의 _cli()와 대칭되는 단발성 테스트 CLI."""
    import argparse

    from gui_actor_grounding import GUIActorModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--model_id", default="microsoft/GUI-Actor-3B-Qwen2.5-VL")
    ap.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--min_pixels", type=int, default=None)
    ap.add_argument("--max_pixels", type=int, default=None)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--debug_image", action="store_true")
    ap.add_argument("--debug_text", action="store_true")
    ap.add_argument("--debug_mode", choices=["always", "incorrect"], default="always")
    ap.add_argument("--task_id", default="demo")
    args = ap.parse_args()

    model = GUIActorModel(
        model_id=args.model_id, attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )
    result = ground_with_regionfocus_gui_actor(
        model, args.instruction, args.image,
        debug_image=args.debug_image, debug_text=args.debug_text, debug_mode=args.debug_mode,
        task_id=args.task_id, topk=args.topk,
    )
    print(result)


if __name__ == "__main__":
    _cli()
