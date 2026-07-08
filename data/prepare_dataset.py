"""
GUI grounding 데이터셋 준비 스크립트

- train/val : Wave-UI (agentsea/wave-ui)  -> SFT 학습용
- test      : ScreenSpot-v2 (OS-Copilot/ScreenSpot-v2) -> 평가용, 학습에 절대 섞지 않음

두 데이터셋의 스키마가 다르기 때문에 아래 통일 포맷(JSONL, 1줄 = 1 sample)으로 변환한다.

unified schema:
{
  "id": str,
  "image_path": str,             # 로컬에 저장된 이미지 경로
  "instruction": str,            # "click on the X" 같은 지시문
  "bbox": [x1, y1, x2, y2],      # 절대 픽셀 좌표 (top-left, bottom-right)
  "point": [cx, cy],             # bbox 중심점 (절대 픽셀)
  "resolution": [w, h],          # 이미지 원본 해상도
  "category": str,               # icon / text / button / ... (있으면)
  "platform": str,               # desktop / mobile / web / windows / ios / ...
  "source": str,                 # 원본 데이터셋 이름
}

사용법:
  python data/prepare_dataset.py --train_samples 4000 --val_samples 500 --out_dir ./data/processed
  python data/prepare_dataset.py --train_samples 10000 --val_samples 500 \
      --platform_quota "web=0.4,mobile=0.3,desktop=0.3"
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def save_image(img: Image.Image, out_dir: Path, name: str) -> str:
    img = img.convert("RGB")
    path = out_dir / f"{name}.jpg"
    if not path.exists():
        img.save(path, quality=90)
    return str(path)


def parse_platform_quota(s: str) -> dict:
    """"web=0.4,mobile=0.3,desktop=0.3" 같은 문자열을 {"web":0.4,...} dict로 변환.
    비율 합은 1이어야 함 (나머지는 quota에 없는 플랫폼들이 채움)."""
    quota = {}
    for pair in s.split(","):
        k, v = pair.split("=")
        quota[k.strip().lower()] = float(v)
    total = sum(quota.values())
    if total > 1.0 + 1e-6:
        raise ValueError(f"--platform_quota 비율 합이 1을 넘음 (현재 {total})")
    return quota


def prepare_wave_ui(out_root: Path, train_samples: int, val_samples: int, platform_quota: dict):
    """Wave-UI -> train/val jsonl. bbox는 이미 [x1,y1,x2,y2] 절대좌표.

    이전에는 wave-ui train split 전체(63.5k rows, ~27GB 추정)를 다 받지 않으려고
    streaming=True로 필요한 만큼만 가져왔음. 하지만 그 방식은 스트림 순서대로만 뽑히기 때문에
    플랫폼별 쿼터(quota)를 맞출 수가 없어서(예: web이 90%를 차지하는 편중 문제), 여기서는
    전체를 한 번에 로컬로 받은 뒤 플랫폼별로 버킷에 나눠 담고 quota 비율대로 뽑는 방식으로 바꿨다.
    다운로드 용량/시간이 늘어나는 대신, train/val의 플랫폼 분포를 직접 통제할 수 있게 된다.
    """
    img_dir = out_root / "images" / "wave_ui"
    img_dir.mkdir(parents=True, exist_ok=True)

    n_total = train_samples + val_samples

    ds = load_dataset("agentsea/wave-ui", split="train")
    ds = ds.shuffle(seed=42)

    # quota에 없는 플랫폼은 전부 "other" 버킷으로 모아서, quota 부족분을 채우는 데 쓴다.
    buckets = {p: [] for p in platform_quota}
    buckets["other"] = []
    # quota보다 넉넉히(1.2배) 모아두면 bbox/이미지 결측으로 걸러지는 샘플이 있어도 여유가 생긴다.
    needed = {p: int(n_total * r * 1.2) + 1 for p, r in platform_quota.items()}

    for row in tqdm(ds, desc="wave-ui (scan)"):
        bbox = row.get("bbox")
        instruction = row.get("instruction") or row.get("name")
        if not bbox or not instruction or row.get("image") is None:
            continue
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue

        platform = (row.get("platform") or "unknown").lower()
        key = platform if platform in platform_quota else "other"
        if key != "other" and len(buckets[key]) >= needed[key]:
            # 이미 이 플랫폼 몫은 충분히 모았으면 스킵 (other는 계속 모음 - 부족분 채우기용)
            continue
        buckets[key].append(row)

        if all(len(buckets[p]) >= needed[p] for p in platform_quota) and len(buckets["other"]) >= n_total:
            break

    selected = []
    for platform, ratio in platform_quota.items():
        want = int(n_total * ratio)
        got = buckets[platform][:want]
        if len(got) < want:
            print(f"[warn] platform={platform} 요청 {want}개인데 {len(got)}개만 모여서 부족분은 other로 채움")
        selected.extend(got)

    shortfall = n_total - len(selected)
    if shortfall > 0:
        selected.extend(buckets["other"][:shortfall])

    random.Random(42).shuffle(selected)

    records = []
    for i, row in enumerate(tqdm(selected, desc="wave-ui (convert)")):
        bbox = row["bbox"]
        resolution = row.get("resolution")
        instruction = (row.get("instruction") or row.get("name")).strip()
        x1, y1, x2, y2 = bbox

        img_path = save_image(row["image"], img_dir, f"wave_{i}")
        w, h = resolution if resolution else row["image"].size

        records.append({
            "id": f"wave_ui_{i}",
            "image_path": img_path,
            "instruction": instruction,
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "point": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
            "resolution": [w, h],
            "category": row.get("type", "unknown"),
            "platform": row.get("platform", "unknown"),
            "source": "wave_ui",
        })

    actual_total = len(records)
    if actual_total < n_total:
        # quota를 다 채우지 못해 총량이 부족한 경우, train_samples를 그대로 다 채우고
        # val_samples를 없애버리면(단순 슬라이싱) val이 통째로 비어버리는 문제가 생긴다.
        # 부족한 만큼은 val_samples 쪽에서 비례 배분으로 줄여서 val이 항상 확보되게 한다.
        print(f"[warn] 요청한 총량 {n_total}개 중 {actual_total}개만 모여서 "
              f"train/val을 비율에 맞게 축소함")
        val_count = max(1, round(val_samples * actual_total / n_total))
        train_count = actual_total - val_count
    else:
        train_count = train_samples
        val_count = val_samples

    train_records = records[:train_count]
    val_records = records[train_count:train_count + val_count]
    return train_records, val_records


def prepare_screenspot_v2(out_root: Path, test_samples: int | None):
    """ScreenSpot-v2 -> test jsonl.

    주의: 원본 리포 OS-Copilot/ScreenSpot-v2는 raw json 3개 + zip 파일 구조라
    `load_dataset()`으로 바로 못 읽음 (HF datasets가 zip 안 이미지를 json으로 파싱하려다
    UnicodeDecodeError 발생 - 리포 자체의 알려진 이슈, discussions #1/#2 참고).
    대신 커뮤니티가 올려둔 parquet 미러 HongxinLi/ScreenSpot_v2를 사용한다.
    이 미러는 bbox가 [x1,y1,x2,y2] "정규화(0~1) 비율" 좌표라 절대 픽셀로 변환 필요.
    """
    img_dir = out_root / "images" / "screenspot_v2"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HongxinLi/ScreenSpot_v2", split="test")
    if test_samples:
        ds = ds.shuffle(seed=42).select(range(min(test_samples, len(ds))))

    records = []
    for i, row in enumerate(tqdm(ds, desc="screenspot-v2")):
        bbox = row.get("bbox")
        instruction = row.get("instruction")
        img = row.get("image")
        if not bbox or not instruction or img is None:
            continue

        img_path = save_image(img, img_dir, f"ss_{i}")
        w, h = img.size

        # 정규화(0~1) xyxy -> 절대 픽셀 xyxy
        nx1, ny1, nx2, ny2 = bbox
        x1, y1, x2, y2 = nx1 * w, ny1 * h, nx2 * w, ny2 * h

        records.append({
            "id": f"screenspot_v2_{i}",
            "image_path": img_path,
            "instruction": instruction.strip(),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "point": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
            "resolution": [w, h],
            "category": row.get("data_type", "unknown"),
            "platform": row.get("data_source", "unknown"),
            "source": "screenspot_v2",
        })
    return records


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {path} ({len(records)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./data/processed")
    ap.add_argument("--train_samples", type=int, default=4000,
                     help="mini PC(CPU) 기준 권장 시작값. 필요시 늘리기")
    ap.add_argument("--val_samples", type=int, default=500)
    ap.add_argument("--test_samples", type=int, default=None,
                     help="None이면 ScreenSpot-v2 전체(약 1.3k) 사용")
    ap.add_argument("--platform_quota", type=str, default="web=0.4,mobile=0.3,desktop=0.3",
                     help="Wave-UI train/val 플랫폼별 샘플링 비율. 콤마로 구분, 합은 1 이하 "
                          "(나머지는 quota에 없는 플랫폼들로 채움). 예: 'web=0.4,mobile=0.3,desktop=0.3'")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    platform_quota = parse_platform_quota(args.platform_quota)
    train_records, val_records = prepare_wave_ui(
        out_root, args.train_samples, args.val_samples, platform_quota
    )
    write_jsonl(train_records, out_root / "train.jsonl")
    write_jsonl(val_records, out_root / "val.jsonl")

    test_records = prepare_screenspot_v2(out_root, args.test_samples)
    write_jsonl(test_records, out_root / "test.jsonl")

import json
from collections import defaultdict
from statistics import mean, median

# 데이터 분포 출력
def stats(path):
    platform_ratios = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)

            bbox = rec["bbox"]
            width_img, height_img = rec["resolution"]
            platform = rec.get("platform", "unknown")

            x1, y1, x2, y2 = bbox

            bbox_w = max(0, x2 - x1)
            bbox_h = max(0, y2 - y1)

            bbox_area = bbox_w * bbox_h
            image_area = width_img * height_img

            ratio = bbox_area / image_area

            platform_ratios[platform].append(ratio)
            
    for platform, ratios in sorted(platform_ratios.items()):
        ratios_pct = [r * 100 for r in ratios]

        print(f"\n[{platform}]")
        print(f"n                : {len(ratios)}")
        print(f"avg bbox ratio   : {mean(ratios_pct):.4f}%")
        print(f"median bbox ratio: {median(ratios_pct):.4f}%")
        print(f"min              : {min(ratios_pct):.4f}%")
        print(f"max              : {max(ratios_pct):.4f}%")


if __name__ == "__main__":
    # main()
    stats("data/processed/train.jsonl")
    stats("data/processed/val.jsonl")