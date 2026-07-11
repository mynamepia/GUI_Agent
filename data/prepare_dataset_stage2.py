"""
prepare_dataset_stage2.py

Qwen-GUI-3B(ZonUI-3B) 논문의 2단계(Two-Stage) 학습 아이디어를 우리 파이프라인에 적용하기 위한
Stage 2용 데이터 준비 스크립트.

배경:
- Stage 1(이미 완료됨: train.jsonl 10000개, platform_quota web=0.55/mobile=0.3/desktop=0.15로
  학습한 기존 체크포인트)은 cross-platform 일반 grounding 능력을 잡는 단계였다.
- Stage 2는 그 위에서 "해상도/밀집 레이아웃 적응"만 짧게 추가로 학습시키는 단계다.
  논문은 이때 고해상도 web-hybrid 데이터를 새로 구했지만, 우리는 prepare_dataset.py의
  save_image()가 원본 해상도를 그대로 저장해두기 때문에(리사이즈 없음) 새 데이터를 받을
  필요가 없다 - 기존 train.jsonl에서 web 샘플만 추려서, train.py를 --max_pixels를 높여
  재실행하면 논문의 "고해상도 특화 학습"과 동일한 효과를 낼 수 있다.
- 다만 web만 100% 넣으면 desktop/mobile 능력이 퇴화(catastrophic forgetting)할 위험이
  있어서, desktop/mobile을 소량(rehearsal_ratio) 섞어 Stage 1에서 얻은 성능을 보존한다.

사용법:
  python data/prepare_dataset_stage2.py \
      --train_jsonl data/processed/train.jsonl \
      --out data/processed/train_stage2.jsonl \
      --rehearsal_ratio 0.2
"""

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_stage2_dataset(records: list, rehearsal_ratio: float, seed: int = 42) -> list:
    """web 플랫폼 샘플 전부 + desktop/mobile(그 외 플랫폼) 일부(rehearsal_ratio만큼)를 섞어서
    Stage 2 학습셋을 만든다. web 비중을 높여 해상도/밀집 레이아웃 적응에 집중하면서도,
    rehearsal 데이터로 Stage 1에서 학습한 desktop/mobile 능력이 잊혀지는 걸 막는다."""
    web = [r for r in records if r.get("platform") == "web"]
    other = [r for r in records if r.get("platform") != "web"]

    rng = random.Random(seed)
    rng.shuffle(other)

    n_rehearsal = int(len(web) * rehearsal_ratio)
    rehearsal = other[:n_rehearsal]

    stage2 = web + rehearsal
    rng.shuffle(stage2)
    return stage2, len(web), len(rehearsal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="./data/processed/train.jsonl",
                     help="Stage 1에서 쓴 원본 train.jsonl 경로")
    ap.add_argument("--out", default="./data/processed/train_stage2.jsonl",
                     help="Stage 2 학습에 쓸 jsonl 저장 경로")
    ap.add_argument("--rehearsal_ratio", type=float, default=0.2,
                     help="web 샘플 수 대비 desktop/mobile을 얼마나 섞을지 비율 (0이면 web만)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_jsonl(args.train_jsonl)
    stage2_records, n_web, n_rehearsal = build_stage2_dataset(
        records, args.rehearsal_ratio, args.seed
    )
    write_jsonl(stage2_records, args.out)

    print(f"[prepare_dataset_stage2.py] source={len(records)}개 중 web={n_web}개 + "
          f"rehearsal(desktop/mobile)={n_rehearsal}개 = 총 {len(stage2_records)}개")
    print(f"[prepare_dataset_stage2.py] saved to {args.out}")


if __name__ == "__main__":
    main()
