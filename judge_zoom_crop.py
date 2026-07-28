"""
judge_inference용 crop + zoom 전처리 함수.

배경 (2026-07):
    ZoomClick(arXiv:2512.05941, github.com/Princeton-AI2-Lab/ZoomClick)의 zoom 파라미터를
    참고해서 judge에게 넘기기 전 point 주변을 crop+zoom한다.

    - shrink 방식: 논문 ablation(Table 6)에서 한 번에 1/4로 자르는 것보다 1/2을 두 번
      적용하는 two-step(1/2+1/2)이 항상 더 좋았음 (62.1% -> 63.9%). 그래서 in_ratio=0.5,
      in_depth=2를 기본값으로 둠.
    - 최소 crop 크기(m): 논문 본문/표에는 수치가 명시돼 있지 않았음. 다만 공식 github의
      grounding/eval_sspro_zoomclick.py 실행 예시(README)에 --in_min_crop 768 로
      하드코딩되어 있는 걸 확인함. 단, 이 768px는 ScreenSpot-Pro(2560~3840px대 고해상도
      스크린샷)에 맞춰진 절대 픽셀값이라, 해상도가 낮은 일반 데스크톱/모바일 스크린샷에
      그대로 쓰면 "거의 안 잘리거나(이미지가 768px보다 작을 때)" "지나치게 넓게 남는" 문제가
      생길 수 있음. 그래서 여기서는 768을 상한으로 참고하되, 원본 이미지의 짧은 변에 대한
      비율(min_crop_ratio)로 한 번 더 clamp해서 해상도가 달라져도 상대적으로 일관된 zoom이
      되도록 함. (이 비율 값 자체는 논문에 없고, 762px가 ScreenSpot-Pro 이미지들의 짧은 변
      대비 대략 25~35% 수준이었다는 점에서 역산해 잡은 값 - 직접 벤치마크로 튜닝 권장)
"""

from dataclasses import dataclass

from PIL import Image


@dataclass
class ResolutionStats:
    """데이터셋 이미지들의 짧은 변(short side) 해상도 분포 통계."""
    n: int
    short_side_min: float
    short_side_p25: float
    short_side_median: float
    short_side_p75: float
    short_side_max: float


def analyze_dataset_resolutions(images) -> ResolutionStats:
    """
    (2026-07 추가) ScreenSpot-v2처럼 우리가 실제 쓰는 데이터셋의 해상도 분포를 계산.

    ZoomClick 공식 github의 in_min_crop=768은 ScreenSpot-Pro(짧은 변 대비 대략
    25~35%)에 맞춰진 값이라 그대로 재사용하지 않고, 우리 데이터셋 자체의 짧은 변
    분포에서 min_crop을 역산하기 위한 전처리 함수.

    Args:
        images: 아래 중 하나의 원소로 구성된 리스트/이터러블
            - PIL.Image 객체
            - (W, H) 튜플
            - 이미지 파일 경로 (str)

    Returns:
        ResolutionStats
    """
    short_sides = []
    for img in images:
        if isinstance(img, Image.Image):
            w, h = img.size
        elif isinstance(img, (tuple, list)) and len(img) == 2:
            w, h = img
        else:  # 파일 경로로 취급
            with Image.open(img) as im:
                w, h = im.size
        short_sides.append(min(w, h))

    if not short_sides:
        raise ValueError("images가 비어 있음 - 해상도 통계를 계산할 수 없음")

    short_sides.sort()
    n = len(short_sides)

    def pct(p):
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return short_sides[idx]

    return ResolutionStats(
        n=n,
        short_side_min=short_sides[0],
        short_side_p25=pct(0.25),
        short_side_median=pct(0.5),
        short_side_p75=pct(0.75),
        short_side_max=short_sides[-1],
    )


def calibrate_min_crop_px(stats: ResolutionStats, ratio: float = 0.3, use: str = "median") -> int:
    """
    데이터셋 해상도 분포(stats)로부터 in_min_crop_px를 역산.

    ratio: min_crop_px / short_side 비율. ZoomClick의 768px가 ScreenSpot-Pro 짧은 변
        대비 대략 25~35% 수준이었다는 점에서 역산한 기본값(0.3). 데이터셋이 바뀌면
        (여기선 ScreenSpot-v2) 이 ratio 자체를 validation set으로 다시 ablation하는 걸
        권장 (예: 0.2 / 0.3 / 0.4 몇 개 후보로 judge 정확도 비교).
    use: 어떤 통계량을 기준으로 역산할지.
        - "median" (기본, 무난): 데이터셋 전체를 대표
        - "p25": 모바일처럼 작은 이미지 비중이 크고 더 타이트하게 자르고 싶을 때
        - "p75": 큰 이미지 기준으로 여유 있게 자르고 싶을 때
    """
    base = getattr(stats, f"short_side_{use}")
    return int(round(base * ratio))


def calibrate_min_crop_px_per_group(images_by_group: dict, ratio: float = 0.3, use: str = "median") -> dict:
    """
    ScreenSpot-v2는 mobile / desktop / web이 섞여 있고 플랫폼별 해상도 편차가 큼.
    전체를 하나의 median으로 뭉뚱그리면 왜곡될 수 있어서, 플랫폼별로 따로 통계 내고
    min_crop도 플랫폼별로 다르게 쓰는 걸 권장.

    Args:
        images_by_group: {"mobile": [...], "desktop": [...], "web": [...]} 형태.
            각 값은 analyze_dataset_resolutions에 넣을 수 있는 리스트(경로/PIL/사이즈).
        ratio, use: calibrate_min_crop_px와 동일.

    Returns:
        {"mobile": {"stats": ResolutionStats, "min_crop_px": int}, "desktop": {...}, ...}
    """
    result = {}
    for group, imgs in images_by_group.items():
        stats = analyze_dataset_resolutions(imgs)
        result[group] = {
            "stats": stats,
            "min_crop_px": calibrate_min_crop_px(stats, ratio=ratio, use=use),
        }
    return result


def crop_and_zoom_for_judge(
    image,
    point,
    in_ratio: float = 0.5,       # ZoomClick 기본 shrink ratio
    in_depth: int = 2,           # two-step(1/2+1/2)이 one-step(1/4)보다 우수 (paper Table 6)
    in_min_crop_px: int = 768,   # 공식 github 기본값 (grounding/eval_sspro_zoomclick.py)
    min_crop_ratio: float = 0.3, # 원본 짧은 변 대비 최소 crop 비율 (해상도 적응용, 저자 미명시 - 직접 설정)
    upscale_to: int | None = 896,  # crop 후 짧은 변을 이 값까지 upscale (28의 배수로 패치 정렬)
):
    """
    judge_inference 호출 직전에 사용. point 주변을 crop한 뒤 필요하면 upscale까지 해서
    반환한다. Qwen 계열은 patch=14, merge=2 -> 토큰 1개=28x28px 이므로, 같은 토큰 예산을
    좁은 영역에 재배분해 marker/target의 유효 해상도(토큰당 픽셀 밀도)를 높이는 것이 목적.

    Args:
        image: PIL.Image, 원본 스크린샷 (judge_inference의 pil_image)
        point: (x, y) 원본 이미지 좌표계 기준 예측 좌표
        in_ratio: 스텝당 축소 비율 (0<in_ratio<1)
        in_depth: 축소 스텝 횟수 (2 = "1/2를 두 번" 방식)
        in_min_crop_px: crop 한 변의 절대 최소 픽셀 (paper github 기본값)
        min_crop_ratio: crop 한 변 / 원본 짧은 변의 최소 비율 (해상도 무관하게 문맥 보존)
        upscale_to: crop 후 짧은 변을 이 값까지 강제 upscale. None이면 upscale 안 함.

    Returns:
        cropped_image: PIL.Image, 최종 (crop [+ upscale]) 이미지
        crop_offset: (x0, y0), 원본 좌표 -> crop 좌표 변환 오프셋
        point_in_cropped: (new_x, new_y), 최종 이미지 좌표계에서의 point
            (plot_points_on_image에 이 좌표를 넘겨서 별을 찍어야 함)
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    W, H = image.size
    x, y = point
    short_side = min(W, H)

    # 1) two-step shrink로 목표 crop 크기 계산 (in_ratio^in_depth)
    target_side = short_side * (in_ratio ** in_depth)

    # 2) 하한 적용: (a) 논문 github 기본값(px), (b) 해상도 비례 하한(ratio) 중 더 큰 쪽을 하한으로 사용
    #    -> 저해상도 이미지에서 768px가 원본보다 커지는 것을 막고, 고해상도 이미지에서
    #       768px가 너무 타이트해지는 것도 막는다.
    min_crop = max(min(in_min_crop_px, short_side), short_side * min_crop_ratio)
    crop_side = max(target_side, min_crop)
    crop_side = min(crop_side, short_side)  # 원본보다 커지지 않도록 clamp

    half = crop_side / 2.0
    x0 = int(round(max(0, min(x - half, W - crop_side))))
    y0 = int(round(max(0, min(y - half, H - crop_side))))
    x1 = int(round(x0 + crop_side))
    y1 = int(round(y0 + crop_side))

    cropped = image.crop((x0, y0, x1, y1))
    new_x, new_y = x - x0, y - y0

    # 3) 필요시 upscale (crop이 이미 충분히 크면 건드리지 않음 - 불필요한 blur 방지)
    if upscale_to is not None:
        cur_short = min(cropped.size)
        if cur_short < upscale_to:
            scale = upscale_to / cur_short
            new_w = int(round(cropped.width * scale))
            new_h = int(round(cropped.height * scale))
            cropped = cropped.resize((new_w, new_h), Image.LANCZOS)
            new_x *= scale
            new_y *= scale

    return cropped, (x0, y0), (new_x, new_y)


# --- 1단계: ScreenSpot-v2 해상도 분포로 min_crop 보정 (스크립트 시작 시 한 번만 실행) ---
#
# from glob import glob
#
# # (a) 전체 데이터셋 하나로 뭉뚱그려 계산하는 경우
# image_paths = glob("/path/to/screenspot_v2/images/*.png")
# stats = analyze_dataset_resolutions(image_paths)
# CALIBRATED_MIN_CROP_PX = calibrate_min_crop_px(stats, ratio=0.3, use="median")
# print(stats, "-> min_crop_px =", CALIBRATED_MIN_CROP_PX)
#
# # (b) mobile/desktop/web처럼 플랫폼별 편차가 큰 경우 (ScreenSpot-v2는 이 케이스에 해당,
# #     권장 방식). task_id나 메타데이터에서 platform을 뽑아서 group을 나눠야 함.
# images_by_group = {
#     "mobile": glob("/path/to/screenspot_v2/images/mobile_*.png"),
#     "desktop": glob("/path/to/screenspot_v2/images/desktop_*.png"),
#     "web": glob("/path/to/screenspot_v2/images/web_*.png"),
# }
# calibrated = calibrate_min_crop_px_per_group(images_by_group, ratio=0.3, use="median")
# # calibrated == {"mobile": {"stats": ..., "min_crop_px": 210}, "desktop": {...}, "web": {...}}
# MIN_CROP_PX_BY_PLATFORM = {g: v["min_crop_px"] for g, v in calibrated.items()}

# --- 2단계: judge_inference에서 보정된 min_crop_px 사용 (기존 함수 앞부분만 발췌 수정) ---
#
# def judge_inference(
#     qwen_model, instruction, image, point,
#     debug_image=False, debug_text=False, debug_mode="always", task_id=None,
#     use_zoom=True, platform=None,  # platform: "mobile" | "desktop" | "web"
# ):
#     pil_image = image.copy() if isinstance(image, Image.Image) else Image.fromarray(image).copy()
#
#     if use_zoom:
#         # 플랫폼별로 보정된 min_crop_px를 쓰고, 없으면 전체 median 기준값으로 fallback
#         min_crop_px = MIN_CROP_PX_BY_PLATFORM.get(platform, CALIBRATED_MIN_CROP_PX)
#         pil_image, _, point_for_star = crop_and_zoom_for_judge(
#             pil_image, point, in_min_crop_px=min_crop_px
#         )
#     else:
#         point_for_star = point
#
#     highlighted_image = plot_points_on_image(
#         pil_image, [point_for_star], colors=[(255, 0, 255, 128)], markers=["star"], sizes=[12]
#     )
#     # ... 이하 동일 (judge_prompt, messages, _generate_with_sampling 등)