"""
verifier/split_train_val.py

generate_verifier_data.py가 만든 라벨 데이터(verifier_train_raw.jsonl 등)에서
train_verifier.py의 --val_jsonl로 쓸 검증셋을 떼어낸다.

[왜 이게 필요한가]
train_verifier.py는 처음부터 --val_jsonl 옵션이 있었지만, 정작 그 검증셋을 만드는
스크립트가 없었다. grounding 쪽 val.jsonl은 이미 verifier 데이터를 만들 때(TRAIN
데이터만 쓰기로 한 결정, generate_verifier_data.py 상단 docstring 참고) 절대 쓰지
않기로 했으니, verifier 자체의 검증셋은 verifier가 만든 TRAIN 후보 데이터 안에서
따로 떼어내야 한다.

[레코드 단위가 아니라 _source_id 단위로 쪼갠다]
generate_verifier_data.py 출력은 소스 샘플(_source_id) 하나당 후보 좌표가 최대
5개(gt_center 포함) 딸려 있다. 이 후보 행 단위로 무작위 분할하면 같은 이미지+
지시문의 다른 후보가 train과 val에 각각 나뉘어 들어갈 수 있는데, 그러면 verifier가
val에서 "이미 train에서 본 이미지"를 판정하게 되는 셈이라 val 점수가 실제보다
낙관적으로 나온다(진짜 검증이 아님). 그래서 _source_id를 먼저 셔플해서 통째로
train/val 그룹으로 나누고, 그 다음 각 그룹에 속한 모든 후보 행을 해당 파일에 쓴다.

사용법:
  cd vlm_agent/verifier
  python split_train_val.py \
      --jsonl verifier_train_raw.dedup.jsonl \
      --val_ratio 0.1 \
      --train_out verifier_train.jsonl \
      --val_out verifier_val.jsonl
"""

import argparse
import json
import random


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True,
                     help="generate_verifier_data.py 출력 (예: verifier_train_raw.dedup.jsonl)")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                     help="_source_id 기준 검증셋 비율 (기본 10%)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_out", required=True)
    ap.add_argument("--val_out", required=True)
    args = ap.parse_args()

    rows = load_jsonl(args.jsonl)

    # _source_id를 처음 등장한 순서대로 모아서 셔플 - 개별 후보 행이 아니라 소스
    # 샘플(이미지+지시문) 단위로 분할해야 train/val 사이에 같은 이미지가 새는 걸 막는다.
    seen = set()
    source_ids = []
    for r in rows:
        sid = r["_source_id"]
        if sid not in seen:
            seen.add(sid)
            source_ids.append(sid)

    rng = random.Random(args.seed)
    rng.shuffle(source_ids)

    n_val = max(1, round(len(source_ids) * args.val_ratio))
    val_ids = set(source_ids[:n_val])
    train_ids = set(source_ids[n_val:])

    n_train_rows = 0
    n_val_rows = 0
    with open(args.train_out, "w", encoding="utf-8") as f_train, \
         open(args.val_out, "w", encoding="utf-8") as f_val:
        for r in rows:
            if r["_source_id"] in val_ids:
                f_val.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_val_rows += 1
            else:
                f_train.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_train_rows += 1

    print(
        f"[split_train_val.py] 소스 샘플 {len(source_ids)}개 -> "
        f"train {len(train_ids)}개(행 {n_train_rows}) / val {len(val_ids)}개(행 {n_val_rows})"
    )
    print(f"[split_train_val.py] 저장: {args.train_out}, {args.val_out}")


if __name__ == "__main__":
    main()
