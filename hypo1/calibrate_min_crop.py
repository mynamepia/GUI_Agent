"""
calibrate_min_crop.py (v2, 2026-07)

ScreenSpot-v2(우리 데이터셋)의 platform별 해상도 분포를 계산해서 judge_inference용
min_crop_px를 역산하는 1회성(offline) 스크립트.

[v1 -> v2로 바뀐 이유]
v1은 이미지 파일 경로/이름에 "mobile"/"desktop"/"web" 문자열이 있는지로 platform을
추측했는데, 실제 데이터셋은 android/ios/macos/windows/gitlab/forum/shop/tool 8개
platform을 쓰고 이 문자열이 경로에 안 박혀 있어서 전부 "all"로 뭉뚱그려지는 문제가
있었다. 게다가 이 platform 정보는 이미 evaluation 파이프라인이 쓰는 jsonl의
"platform" 필드에 정확하게 들어있다 (eval_regionfocus_WJ.py의 _get_platform()이
쓰는 것과 동일한 필드).

그래서 v2는 파일명 추측을 완전히 버리고, coord_utils.load_jsonl()로 jsonl을 그대로
읽어서 각 record의 rec["platform"]으로 정확하게 그룹핑한다. 해상도도 이미지를 다시
열 필요 없이 rec["resolution"]을 그대로 쓴다 (score_regionfocus_result()가 이미 이
필드를 신뢰하고 쓰고 있는 것과 동일한 가정 - 없는 record만 image_path를 열어 fallback).

이렇게 하면:
    - jsonl과 100% 같은 platform 정의를 쓰므로 eval_regionfocus_WJ.py의
      _lookup_min_crop_px(min_crop_cfg, platform)이 항상 정확히 매칭됨.
    - annotation_json(순수 JSON 배열)을 쓸 필요가 없어짐 - 애초에 데이터가 jsonl이라
      그 방식은 안 맞았음.

사용 예:
    python calibrate_min_crop.py \
        --jsonl /srv/project/data/processed/val.jsonl \
        --ratio 0.3 --use median \
        --out min_crop_config.json

    # jsonl에 "resolution" 필드가 없는 경우에만 사용됨 (이미지 열어서 크기 확인):
    python calibrate_min_crop.py --jsonl /srv/project/data/processed/val.jsonl --image_root /srv/project/data/processed/images
"""

import argparse
import json
import os

from coord_utils import load_jsonl
from judge_zoom_crop import (
    analyze_dataset_resolutions,
    calibrate_min_crop_px,
    calibrate_min_crop_px_per_group,
)


def _get_platform(rec):
    """eval_regionfocus_WJ.py의 _get_platform()과 동일한 규칙 (같은 fallback을 씀)."""
    return rec.get("platform") or "unknown"


def _resolution_of(rec, image_root=None):
    """
    rec["resolution"]을 우선 사용 (evaluation.py/eval_regionfocus_WJ.py가 이미 신뢰하는
    필드라 이미지 I/O 없이 바로 씀). 없으면 image_path를 열어서 PIL로 크기를 구한다.
    둘 다 없으면 None 반환(해당 record는 스킵).
    """
    res = rec.get("resolution")
    if res:
        return (res[0], res[1])

    path = rec.get("image_path")
    if not path:
        return None
    if image_root and not os.path.isabs(path):
        path = os.path.join(image_root, path)
    if not os.path.exists(path):
        return None

    from PIL import Image
    with Image.open(path) as im:
        return im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, nargs="+",
                    help="platform/해상도 계산에 쓸 jsonl 파일(들). 여러 개 넘기면 다 합쳐서 계산 "
                         "(예: --jsonl data/train.jsonl data/val.jsonl)")
    ap.add_argument("--image_root", default=None,
                    help="record에 resolution 필드가 없을 때만 사용: image_path가 상대경로면 이 루트를 붙임")
    ap.add_argument("--ratio", type=float, default=0.3,
                    help="min_crop_px = short_side_<use> * ratio (초기값, validation으로 재검증 권장)")
    ap.add_argument("--use", choices=["median", "p25", "p75"], default="median")
    ap.add_argument("--out", default="min_crop_config.json")
    args = ap.parse_args()

    records = []
    for path in args.jsonl:
        records.extend(load_jsonl(path))

    if not records:
        raise SystemExit(f"{args.jsonl}에서 record를 하나도 못 읽었음 - 경로 확인할 것")

    images_by_group = {}
    n_skipped = 0
    for rec in records:
        wh = _resolution_of(rec, image_root=args.image_root)
        if wh is None:
            n_skipped += 1
            continue
        platform = _get_platform(rec)
        images_by_group.setdefault(platform, []).append(wh)

    if n_skipped:
        print(f"[경고] {n_skipped}개 record는 resolution도 없고 image_path도 못 열어서 스킵함")

    if not images_by_group:
        raise SystemExit(
            "platform별 해상도 정보를 하나도 못 모았음 - jsonl에 'resolution'과 'platform' "
            "필드가 실제로 있는지 확인할 것 (예: head -1 data/val.jsonl | python -m json.tool)"
        )

    print(
        f"[jsonl 'platform' 필드 사용] 그룹: "
        f"{{ {', '.join(f'{k}: {len(v)}장' for k, v in sorted(images_by_group.items()))} }}"
    )

    # 플랫폼별 통계 + min_crop_px
    per_group = calibrate_min_crop_px_per_group(images_by_group, ratio=args.ratio, use=args.use)

    # 전체(global) 통계 + min_crop_px (platform이 config에 없을 때의 fallback용)
    all_wh = [wh for whs in images_by_group.values() for wh in whs]
    global_stats = analyze_dataset_resolutions(all_wh)
    global_min_crop = calibrate_min_crop_px(global_stats, ratio=args.ratio, use=args.use)

    config = {
        "ratio": args.ratio,
        "use": args.use,
        "n_images_total": len(all_wh),
        "n_skipped": n_skipped,
        "global_min_crop_px": global_min_crop,
        "global_stats": vars(global_stats),
        "per_platform_min_crop_px": {g: v["min_crop_px"] for g, v in per_group.items()},
        "per_platform_stats": {g: vars(v["stats"]) for g, v in per_group.items()},
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"\n-> {args.out} 에 저장됨. eval_regionfocus_WJ.py --min_crop_config로 넘길 것.")


if __name__ == "__main__":
    main()