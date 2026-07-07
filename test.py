"""
test.py

학습된 (혹은 베이스) Qwen2.5-VL GUI grounding 모델을 실제 데이터로 돌려보는 inference/평가 스크립트.
evaluation.py의 채점 로직을 그대로 가져다 써서, Wave-UI / ScreenSpot-v2 jsonl에 대해
  1) 이미지+지시문을 넣고 모델이 생성한 좌표를 뽑고 (inference)
  2) 정답 bbox와 비교해서 click accuracy 등을 계산 (evaluation.py)
을 한 번에 수행한다.

데이터 포맷은 evaluation.py 상단 docstring과 동일 (coord_utils.load_jsonl로 읽는 jsonl,
resolution/point/bbox 필드, 선택적으로 dataset/platform/category).

사용법:
  # 베이스 모델 그대로 평가
  python test.py --jsonl data/processed/screenspot_v2_test.jsonl

  # LoRA 어댑터를 얹어서 평가 (train.py의 --output_dir을 그대로 지정)
  python test.py --jsonl data/processed/screenspot_v2_test.jsonl \\
      --adapter_dir ./checkpoints/qwen2.5vl-3b-gui-lora

  # Wave-UI + ScreenSpot-v2를 한 번에 (파일별로 dataset 태그가 자동으로 붙어서 breakdown도 나옴)
  python test.py \\
      --jsonl data/processed/wave_ui_test.jsonl data/processed/screenspot_v2_test.jsonl \\
      --adapter_dir ./checkpoints/qwen2.5vl-3b-gui-lora --out preds.jsonl

  # 병합(merge)된 모델을 평가하고 싶으면 --adapter_dir 없이 --model_id에 병합 체크포인트 경로만 지정
  python test.py --jsonl ... --model_id ./checkpoints/qwen2.5vl-3b-gui-merged
"""

import argparse
import json
from pathlib import Path

from coord_utils import PROMPT_TEMPLATE, load_jsonl
from evaluation import aggregate_metrics, format_report, score_prediction
from qwen import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, MODEL_ID, generate_text, load_model_and_processor


def _load_records_with_dataset_tag(jsonl_paths):
    """
    각 jsonl에 dataset 필드가 없으면 파일 이름으로 태그를 붙여서, 여러 데이터셋을
    한 번에 넣어도 evaluation.py의 by_dataset breakdown이 항상 의미 있게 나오도록 한다.
    """
    records = []
    for path in jsonl_paths:
        recs = load_jsonl(path)
        tag = Path(path).stem
        for rec in recs:
            rec.setdefault("dataset", tag)
        records.extend(recs)
    return records


def run_inference(model, processor, records, max_new_tokens=32, verbose=True):
    rows = []
    for i, rec in enumerate(records):
        prompt = PROMPT_TEMPLATE.format(instruction=rec["instruction"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rec["image_path"]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        pred_text = generate_text(model, processor, messages, max_new_tokens=max_new_tokens)
        row = score_prediction(rec, pred_text)
        rows.append(row)
        if verbose:
            status = "O" if row["hit"] else ("?" if not row["parsed_ok"] else "X")
            print(f"[{i + 1}/{len(records)}] {status} id={row['id']} pred={pred_text!r}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True,
                     help="평가할 jsonl 경로 (여러 개면 Wave-UI/ScreenSpot-v2 등을 한 번에 평가)")
    ap.add_argument("--model_id", default=MODEL_ID, help="베이스 모델 id 혹은 병합된 체크포인트 경로")
    ap.add_argument("--adapter_dir", default=None, help="LoRA adapter 디렉토리 (train.py의 output_dir)")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="스모크 테스트용 샘플 수 제한")
    ap.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--out", default=None, help="샘플별 예측 결과를 저장할 jsonl 경로")
    ap.add_argument("--metrics_out", default=None, help="집계된 지표를 저장할 json 경로")
    ap.add_argument("--quiet", action="store_true", help="샘플별 진행 로그 끄기")
    args = ap.parse_args()

    model, processor = load_model_and_processor(
        model_id=args.model_id,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    if args.adapter_dir:
        from peft import PeftModel

        print(f"[test.py] Loading LoRA adapter from {args.adapter_dir} ...")
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    model.eval()

    records = _load_records_with_dataset_tag(args.jsonl)
    if args.limit is not None:
        records = records[: args.limit]
    print(f"[test.py] {len(records)} samples loaded from {len(args.jsonl)} file(s)")

    rows = run_inference(
        model, processor, records,
        max_new_tokens=args.max_new_tokens,
        verbose=not args.quiet,
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[test.py] predictions saved to {args.out}")

    metrics = aggregate_metrics(rows)
    print(format_report(metrics, title=" / ".join(args.jsonl)))

    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[test.py] metrics saved to {args.metrics_out}")


if __name__ == "__main__":
    main()