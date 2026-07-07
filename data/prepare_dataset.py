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
"""

import argparse
import json
import os
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


def prepare_wave_ui(out_root: Path, train_samples: int, val_samples: int):
    """Wave-UI -> train/val jsonl. bbox는 이미 [x1,y1,x2,y2] 절대좌표.

    주의: wave-ui train split 전체는 63.5k rows (~27GB 추정). train_samples+val_samples만
    필요하므로 streaming=True로 받아서 그만큼만 다운로드한다 (non-streaming으로 받으면
    전체를 먼저 로컬에 캐싱한 뒤 일부만 쓰게 되어 디스크/시간 낭비가 큼).
    """
    img_dir = out_root / "images" / "wave_ui"
    img_dir.mkdir(parents=True, exist_ok=True)

    n_total = train_samples + val_samples

    ds = load_dataset("agentsea/wave-ui", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10_000)
    ds = ds.take(n_total)

    records = []
    for i, row in enumerate(tqdm(ds, desc="wave-ui", total=n_total)):
        bbox = row.get("bbox")
        resolution = row.get("resolution")
        instruction = row.get("instruction") or row.get("name")
        if not bbox or not instruction or row.get("image") is None:
            continue

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue

        img_path = save_image(row["image"], img_dir, f"wave_{i}")
        w, h = resolution if resolution else row["image"].size

        records.append({
            "id": f"wave_ui_{i}",
            "image_path": img_path,
            "instruction": instruction.strip(),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "point": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
            "resolution": [w, h],
            "category": row.get("type", "unknown"),
            "platform": row.get("platform", "unknown"),
            "source": "wave_ui",
        })

    train_records = records[:train_samples]
    val_records = records[train_samples:train_samples + val_samples]
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
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    train_records, val_records = prepare_wave_ui(out_root, args.train_samples, args.val_samples)
    write_jsonl(train_records, out_root / "train.jsonl")
    write_jsonl(val_records, out_root / "val.jsonl")

    test_records = prepare_screenspot_v2(out_root, args.test_samples)
    write_jsonl(test_records, out_root / "test.jsonl")


if __name__ == "__main__":
    main()