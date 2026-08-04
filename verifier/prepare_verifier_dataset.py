"""
verifier/prepare_verifier_dataset.py

verifier 학습용 소스 데이터를 이미 만들어둔 train.jsonl/train_stage2.jsonl/val.jsonl/
test.jsonl과 절대 안 겹치게, HuggingFace wave-ui에서 새로 받아오는 스크립트.

[왜 새로 받는가]
처음엔 val.jsonl(grounding LoRA가 gradient update에 안 쓴 held-out)을 재활용하는
방안을 검토했는데, val.jsonl은 원래 grounding 쪽 자체 검증(학습 중 loss 모니터링,
train.py의 post-training report)에 쓰라고 떼어둔 데이터라 verifier 데이터로 섞으면
그 역할이 흐려진다는 지적이 있어서, HF에서 완전히 새로 받아 별도 pool을 만드는
쪽으로 바꿈. 대신 이미 로컬에 있는 4개 jsonl과 겹치는 샘플은 걸러낸다.

[중복 판정 방식]
wave-ui 원본에 안정적인 전역 id가 없어서(prepare_dataset.py도 그때그때 셔플 순서
기준으로 "wave_ui_{i}"를 새로 매길 뿐), instruction 텍스트 + bbox(소수 1자리 반올림)
+ resolution 조합을 키로 써서 기존 4개 jsonl과 겹치는지 판정한다. 우연히 같은
이미지의 같은 지시문에 같은 bbox·해상도가 나올 확률은 사실상 0에 가까움.
prepare_dataset.py가 seed=42로 셔플했던 것과 겹칠 가능성을 더 줄이려고 여기 기본
seed는 다르게(123) 둠 - 어차피 dedup 필터가 최종 안전망이라 필수는 아니지만.

[2026-08 추가: --reuse_platforms - 플랫폼별 fallback]
실제로 돌려보니 wave-ui는 web 비중이 압도적으로 커서(prepare_dataset.py 주석 참고),
mobile/desktop은 기존 train/train_stage2/val이 이미 그 적은 풀을 상당수 가져가버려서
새로 dedup까지 통과하는 fresh 샘플이 거의 안 남는다(실측: mobile 0개, desktop
192/600개). 이 경우 mobile/desktop처럼 fresh 풀이 마른 플랫폼은 --reuse_platforms로
지정하면, HF에서 새로 긁는 대신 --existing_jsonls에 이미 있는 로컬 레코드를 dedup
없이 그대로 재사용한다(grounding LoRA가 이미 본 데이터가 섞여도 감수 - 사용자 결정).
web처럼 fresh 풀이 넉넉한 플랫폼만 HF에서 새로 받아온다. 각 row의 "_source" 필드로
"fresh_hf"/"local_reuse"를 남겨서 나중에 구분 가능.

사용법:
  cd vlm_agent/verifier
  python prepare_verifier_dataset.py \
      --existing_jsonls ../data/processed/train.jsonl ../data/processed/train_stage2.jsonl \
                        ../data/processed/val.jsonl \
      --reuse_platforms mobile,desktop \
      --n_total 2000 \
      --platform_quota "web=0.4,mobile=0.3,desktop=0.3" \
      --out_dir ../data/processed \
      --out verifier_source.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[saved] {path} ({len(records)} rows)")


def _dedup_key(instruction, bbox, resolution):
    return (
        (instruction or "").strip(),
        tuple(round(float(v), 1) for v in bbox),
        tuple(resolution),
    )


def build_exclusion_set(existing_records):
    keys = set()
    for r in existing_records:
        if r.get("bbox") and r.get("resolution"):
            keys.add(_dedup_key(r["instruction"], r["bbox"], r["resolution"]))
    return keys


def build_local_pool_by_platform(existing_records, quota):
    """--reuse_platforms 용 fallback pool. --existing_jsonls의 레코드를 그대로
    재사용(dedup 없음 - 이미 grounding LoRA가 봤어도 감수하기로 함)."""
    buckets = defaultdict(list)
    for r in existing_records:
        p = (r.get("platform") or "unknown").lower()
        key = p if p in quota else "other"
        buckets[key].append(r)
    return buckets


def parse_platform_quota(s: str) -> dict:
    """prepare_dataset.py와 동일한 포맷: "web=0.4,mobile=0.3,desktop=0.3".
    quota에 없는 플랫폼은 전부 "other"로 묶여서 총량 부족분을 채우는 데만 쓰인다."""
    quota = {}
    for pair in s.split(","):
        k, v = pair.split("=")
        quota[k.strip().lower()] = float(v)
    total = sum(quota.values())
    if total > 1.0 + 1e-6:
        raise ValueError(f"--platform_quota 비율 합이 1을 넘음 (현재 {total})")
    return quota


def save_image(img, out_dir: Path, name: str) -> str:
    img = img.convert("RGB")
    path = out_dir / f"{name}.jpg"
    if not path.exists():
        img.save(path, quality=90)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing_jsonls", nargs="+", required=True,
                     help="이미 만들어둔 jsonl들 (train/train_stage2/val/test). 여기 있는 "
                          "(instruction, bbox, resolution)과 겹치는 wave-ui 샘플은 전부 제외")
    ap.add_argument("--n_total", type=int, default=3000)
    ap.add_argument("--platform_quota", type=str, default="web=0.4,mobile=0.3,desktop=0.3",
                     help="prepare_dataset.py와 동일한 포맷")
    ap.add_argument("--out_dir", default="../data/processed")
    ap.add_argument("--out", default="verifier_source.jsonl",
                     help="--out_dir 밑에 저장될 파일명")
    ap.add_argument("--reuse_platforms", type=str, default="",
                     help="콤마로 구분된 플랫폼 목록 (예: 'mobile,desktop'). 여기 지정된 "
                          "플랫폼은 HF에서 새로 안 받고 --existing_jsonls의 로컬 레코드를 "
                          "dedup 없이 그대로 재사용한다 (wave-ui에 그 플랫폼 fresh 샘플이 "
                          "부족할 때 씀).")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    reuse_platforms = {p.strip().lower() for p in args.reuse_platforms.split(",") if p.strip()}

    existing_records = []
    for path in args.existing_jsonls:
        existing_records.extend(load_jsonl(path))
    exclusion = build_exclusion_set(existing_records)
    print(f"[prepare_verifier_dataset.py] 기존 데이터 {len(exclusion)}개 키로 제외 대상 구성")

    out_root = Path(args.out_dir)
    img_dir = out_root / "images" / "wave_ui_verifier"
    img_dir.mkdir(parents=True, exist_ok=True)

    quota = parse_platform_quota(args.platform_quota)
    fresh_platforms = {p for p in quota if p not in reuse_platforms}

    rng = random.Random(args.seed)
    local_pool = build_local_pool_by_platform(existing_records, quota)
    for bucket in local_pool.values():
        rng.shuffle(bucket)

    records = []
    report = {}

    # --- 1) reuse_platforms: HF 안 긁고 로컬 레코드 그대로 재사용 ---
    for platform in reuse_platforms & set(quota):
        want = int(args.n_total * quota[platform])
        got = local_pool.get(platform, [])[:want]
        for r in got:
            r = dict(r)
            r["_source"] = "local_reuse"
            records.append(r)
        report[platform] = {"want": want, "got": len(got), "mode": "local_reuse"}
        if len(got) < want:
            print(f"[warn] platform={platform}(reuse) 요청 {want}개인데 로컬에 {len(got)}개만 있음")

    # --- 2) fresh_platforms: HF에서 새로 받고 dedup ---
    if fresh_platforms:
        ds = load_dataset("agentsea/wave-ui", split="train")
        ds = ds.shuffle(seed=args.seed)

        buckets = {p: [] for p in fresh_platforms}
        buckets["other"] = []
        # quota보다 넉넉히(1.3배) 모아두면 dedup/bbox 결측으로 걸러지는 샘플이 있어도 여유가 생김.
        needed = {p: int(args.n_total * quota[p] * 1.3) + 1 for p in fresh_platforms}
        n_skipped_dup = 0

        for row in tqdm(ds, desc="wave-ui (scan, dedup)"):
            bbox = row.get("bbox")
            instruction = row.get("instruction") or row.get("name")
            if not bbox or not instruction or row.get("image") is None:
                continue
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue

            resolution = row.get("resolution") or row["image"].size
            key = _dedup_key(instruction, bbox, resolution)
            if key in exclusion:
                n_skipped_dup += 1
                continue

            platform = (row.get("platform") or "unknown").lower()
            if platform in reuse_platforms:
                continue  # reuse_platforms는 위에서 이미 로컬로 채웠으니 HF에서 또 안 뽑음
            bkey = platform if platform in fresh_platforms else "other"
            if bkey != "other" and len(buckets[bkey]) >= needed[bkey]:
                continue
            buckets[bkey].append(row)

            if all(len(buckets[p]) >= needed[p] for p in fresh_platforms):
                break

        fresh_selected = []
        for platform in fresh_platforms:
            want = int(args.n_total * quota[platform])
            got = buckets[platform][:want]
            if len(got) < want:
                print(f"[warn] platform={platform} 요청 {want}개인데 {len(got)}개만 모여서 부족분은 other로 채움")
            report[platform] = {"want": want, "got": len(got), "mode": "fresh_hf"}
            fresh_selected.extend(got)

        shortfall = int(args.n_total * sum(quota[p] for p in fresh_platforms)) - len(fresh_selected)
        if shortfall > 0:
            fresh_selected.extend(buckets["other"][:shortfall])

        for i, row in enumerate(tqdm(fresh_selected, desc="wave-ui (convert)")):
            bbox = row["bbox"]
            instruction = (row.get("instruction") or row.get("name")).strip()
            x1, y1, x2, y2 = bbox
            img_path = save_image(row["image"], img_dir, f"waveverif_{i}")
            resolution = row.get("resolution") or row["image"].size
            w, h = resolution

            records.append({
                "id": f"wave_ui_verifier_{i}",
                "image_path": img_path,
                "instruction": instruction,
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "point": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                "resolution": [w, h],
                "category": row.get("type", "unknown"),
                "platform": row.get("platform", "unknown"),
                "source": "wave_ui_verifier",
                "_source": "fresh_hf",
            })

        print(f"[prepare_verifier_dataset.py] 중복이라 스킵한 개수: {n_skipped_dup}")

    rng.shuffle(records)
    write_jsonl(records, out_root / args.out)

    for platform, info in report.items():
        print(f"  {platform:<10}[{info['mode']:<11}] 목표 {info['want']:>5}개 -> 확보 {info['got']:>5}개")
    if len(records) < args.n_total:
        print(
            f"[prepare_verifier_dataset.py] 주의: 요청한 {args.n_total}개 중 {len(records)}개만 "
            f"확보됨 (--n_total을 줄이거나 --platform_quota/--reuse_platforms를 조정할 것)"
        )


if __name__ == "__main__":
    main()
