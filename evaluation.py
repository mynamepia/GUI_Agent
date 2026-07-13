"""
evaluation.py

GUI grounding 평가 지표 + 평가 루프.
Wave-UI, ScreenSpot-v2 두 데이터셋(둘 다 prepare_dataset.py로 만든 jsonl 포맷을 그대로 사용)을
평가하기 위한 모듈. train.py(학습 직후 val set 평가)와 test.py(체크포인트 단독 평가)에서
공통으로 import해서 쓴다. 좌표 파싱/변환은 coord_utils.py 걸 그대로 재사용하고, 여기서는
"모델 출력 vs 정답"을 채점하는 로직만 담당한다.

[데이터 포맷 가정]
jsonl의 각 레코드는 최소한 다음 필드를 갖는다고 가정:
    id           : str
    image_path   : str
    instruction  : str
    resolution   : [w, h]           - 원본 이미지 픽셀 해상도
    point        : [x, y]           - 학습 타깃용 기준점(픽셀), 보통 bbox 중심 (train.py에서 사용)
    bbox         : [x1, y1, x2, y2] - 정답 bbox(픽셀) - 클릭 정확도 판정용
선택적으로 아래 필드가 있으면 그 기준으로도 breakdown을 낸다 (없으면 "unknown"으로 묶임):
    dataset      : str   예) "wave_ui", "screenspot_v2"
    platform     : str   ScreenSpot-v2 예) "mobile" / "desktop" / "web"
    category     : str   ScreenSpot-v2 예) "text" / "icon" (element_type 필드로 와도 인식)

[지표]
click accuracy : 모델이 생성한 "(x,y)"(0~1000 정규화)를 픽셀로 되돌린 뒤, 정답 bbox 안에
                 들어가면 hit으로 집계. ScreenSpot / ScreenSpot-v2 / Wave-UI 계열 GUI
                 grounding 논문들에서 공통으로 쓰는 표준 지표.
parse_fail_rate: 모델 출력에서 "(x,y)" 좌표를 아예 못 뽑아낸 비율. 포맷을 안 지킨 것도
                 정답이 아닌 것으로 집계한다 (accuracy 분모에는 포함, hit=False 처리).
"""

import argparse
import json
from collections import defaultdict

from tqdm import tqdm

from coord_utils import PROMPT_TEMPLATE, load_jsonl, norm1000_to_point, parse_point_from_text
from qwen import generate_text


def _get_category(rec):
    return rec.get("category") or rec.get("element_type") or "unknown"


def _get_platform(rec):
    return rec.get("platform") or "unknown"


def _get_dataset(rec):
    return rec.get("dataset") or rec.get("source") or "unknown"


_GENERIC_LABELS = {
    "background", "banner", "button", "checkbox", "dropdown", "file", "header",
    "icon", "image", "input", "label", "link", "list item", "menu item",
    "option", "radio button", "section", "tab", "text", "text bubble", "toggle",
}
_ROLE_WORDS = {
    "statictext", "link", "listitem", "button", "image", "menuitem", "group",
    "text", "icon", "generic",
}


def is_uninformative_instruction(instruction):
    s = (instruction or "").strip()
    low = s.lower()
    if low in _GENERIC_LABELS:
        return True
    if "," in s and "->" not in s and "[" not in s:
        parts = [p.strip().lower() for p in s.split(",")]
        if parts and all(p in _ROLE_WORDS for p in parts):
            return True
    return False


def score_prediction(rec, pred_text):
    resolution = rec["resolution"]
    bbox = rec.get("bbox")

    norm_point = parse_point_from_text(pred_text)
    parsed_ok = norm_point is not None
    pred_px = norm1000_to_point(norm_point, resolution) if parsed_ok else None

    hit = False
    if parsed_ok and bbox is not None:
        x1, y1, x2, y2 = bbox
        px, py = pred_px
        hit = (x1 <= px <= x2) and (y1 <= py <= y2)

    return {
        "id": rec.get("id"),
        "dataset": _get_dataset(rec),
        "platform": _get_platform(rec),
        "category": _get_category(rec),
        "instruction": rec.get("instruction"),
        "pred_text": pred_text,
        "pred_point_px": list(pred_px) if pred_px is not None else None,
        "gt_bbox": bbox,
        "parsed_ok": parsed_ok,
        "hit": hit,
    }


def _accuracy(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "parse_fail_rate": 0.0}
    hits = sum(1 for r in rows if r["hit"])
    fails = sum(1 for r in rows if not r["parsed_ok"])
    return {"n": n, "accuracy": hits / n, "parse_fail_rate": fails / n}


def _aggregate_core(rows):
    metrics = {"overall": _accuracy(rows)}

    by_dataset = defaultdict(list)
    by_platform = defaultdict(list)
    by_category = defaultdict(list)
    by_platform_category = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(r)
        by_platform[r["platform"]].append(r)
        by_category[r["category"]].append(r)
        by_platform_category[(r["platform"], r["category"])].append(r)

    if set(by_dataset.keys()) != {"unknown"}:
        metrics["by_dataset"] = {ds: _accuracy(rs) for ds, rs in sorted(by_dataset.items())}
    if set(by_platform.keys()) != {"unknown"}:
        metrics["by_platform"] = {p: _accuracy(rs) for p, rs in sorted(by_platform.items())}
    if set(by_category.keys()) != {"unknown"}:
        metrics["by_category"] = {c: _accuracy(rs) for c, rs in sorted(by_category.items())}
    if len(by_platform_category) > 1:
        metrics["by_platform_category"] = {
            f"{p}/{c}": _accuracy(rs) for (p, c), rs in sorted(by_platform_category.items())
        }

    return metrics


def aggregate_metrics(rows):
    metrics = _aggregate_core(rows)

    clean_rows = [r for r in rows if not is_uninformative_instruction(r.get("instruction"))]
    if len(clean_rows) != len(rows):
        metrics["clean"] = _aggregate_core(clean_rows)
        metrics["clean"]["_removed_uninformative_n"] = len(rows) - len(clean_rows)

    return metrics


def format_report(metrics, title="Evaluation"):
    lines = [f"=== {title} ==="]
    ov = metrics["overall"]
    lines.append(
        f"Overall: acc={ov['accuracy']:.4f}  (n={ov['n']}, parse_fail={ov['parse_fail_rate']:.4f})"
    )

    def _section(name, key):
        if key not in metrics:
            return
        lines.append(f"-- {name} --")
        for k, v in metrics[key].items():
            lines.append(f"  {k:<24} acc={v['accuracy']:.4f}  (n={v['n']})")

    _section("by dataset", "by_dataset")
    _section("by platform", "by_platform")
    _section("by category", "by_category")
    _section("by platform x category", "by_platform_category")

    if "clean" in metrics:
        clean = metrics["clean"]
        removed = clean.get("_removed_uninformative_n", 0)
        cov = clean["overall"]
        lines.append(f"-- clean (uninformative instruction {removed}개 제외) --")
        lines.append(f"  overall                  acc={cov['accuracy']:.4f}  (n={cov['n']})")
        if "by_platform" in clean:
            for k, v in clean["by_platform"].items():
                lines.append(f"  platform/{k:<15} acc={v['accuracy']:.4f}  (n={v['n']})")

    return "\n".join(lines)


def run_generation_eval(model, processor, jsonl_path, max_new_tokens=32, limit=None, predictions_out_path=None):
    records = load_jsonl(jsonl_path)
    if limit is not None:
        records = records[:limit]

    rows = []
    was_training = getattr(model, "training", False)
    model.eval()
    for rec in tqdm(records, desc="generation eval"):
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
        rows.append(score_prediction(rec, pred_text))
    if was_training:
        model.train()

    if predictions_out_path:
        with open(predictions_out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return aggregate_metrics(rows)


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="평가할 데이터셋 jsonl (test/val)")
    ap.add_argument("--model_id", default=None, help="지정하면 이 체크포인트로 새로 추론해서 평가")
    ap.add_argument("--adapter_dir", default=None, help="LoRA adapter 디렉토리 (model_id와 같이 사용)")
    ap.add_argument("--predictions", default=None, help="이미 뽑아둔 예측 jsonl (score_prediction과 같은 필드를 가진 파일)")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="예측 결과를 저장할 jsonl 경로")
    ap.add_argument("--min_pixels", type=int, default=None, help="지정 안 하면 qwen.py의 DEFAULT_MIN_PIXELS 사용")
    ap.add_argument("--max_pixels", type=int, default=None, help="지정 안 하면 qwen.py의 DEFAULT_MAX_PIXELS 사용")
    args = ap.parse_args()

    if args.predictions:
        rows = load_jsonl(args.predictions)
        metrics = aggregate_metrics(rows)
    elif args.model_id:
        from qwen import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, load_model_and_processor

        model, processor = load_model_and_processor(
            model_id=args.model_id,
            min_pixels=args.min_pixels if args.min_pixels is not None else DEFAULT_MIN_PIXELS,
            max_pixels=args.max_pixels if args.max_pixels is not None else DEFAULT_MAX_PIXELS,
        )
        if args.adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter_dir)

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        metrics = run_generation_eval(
            model, processor, args.jsonl,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            predictions_out_path=args.out,
        )
    else:
        raise SystemExit("--model_id 또는 --predictions 둘 중 하나는 필요합니다.")

    print(format_report(metrics, title=args.jsonl))


if __name__ == "__main__":
    _cli()