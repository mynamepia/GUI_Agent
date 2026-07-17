"""
eval_regionfocus.py

region_focus.ground_with_regionfocus()를 test/val jsonl 전체에 대해 돌려서
evaluation.py와 같은 방식(click accuracy, by_platform/by_category breakdown,
clean subset)으로 채점하는 배치 평가 스크립트.

region_focus.py는 이미지 1장짜리 단발 실행(_cli)만 제공하고, 반환 스키마도
evaluation.score_prediction()이 기대하는 "(x,y) 텍스트" 형식이 아니라 이미
파싱된 0~1 정규화 point라서, evaluation.py를 그대로 재사용할 수 없다.
그래서 이 스크립트는:
    1) ground_with_regionfocus()로 record 하나씩 좌표를 얻고
    2) evaluation.py와 동일한 row 스키마(dict)로 변환해서
    3) evaluation.aggregate_metrics()/format_report()에 그대로 꽂는다.

[로깅]
region_focus.py/gui_grounding.py 내부는 이미 step별/generate 호출별로
print()로 소요시간과 진행 상황을 찍고 있음(Step 1/5 ..., "[judge_inference] 완료 -
2.3초", "총 소요시간 X초" 등). 이 파일들은 건드리지 않고, 샘플 하나를 처리하는
동안의 stdout을 그대로 캡처(Tee: 콘솔에도 실시간으로 찍고 동시에 문자열로도 저장)해서
각 row에 "process_log"로 같이 저장한다. 여기에 추가로 샘플당 총 소요시간
("elapsed_sec")과 매 샘플 직후 hit/miss 한 줄 요약을 콘솔에 찍는다.
evaluation.aggregate_metrics()는 필요한 필드(hit/parsed_ok/platform/category/dataset)만
보고 나머지는 무시하므로, 이 추가 필드들이 있어도 기존 evaluation.py 파이프라인과
100% 호환된다.

주의: RegionFocus는 한 샘플당 judge/재탐색/crop-zoom 4회/aggregation까지
model.generate()를 최대 7~8회 호출할 수 있어서, plain grounding(1회 호출) 대비
훨씬 느리다. 전체 1272개를 한 번에 돌리기 전에 --limit 20~30 정도로 먼저
스모크 테스트하는 걸 추천한다.

SMOKE TEST
python eval_regionfocus.py --jsonl data/processed/test.jsonl --adapter_dir ./checkpoints/qwen2.5vl-3b-gui-lora-stage2/checkpoint-4130 --max_pixels 700000 --limit 20 --out regionfocus_smoke.jsonl --metrics_out regionfocus_smoke.json

FULL
python eval_regionfocus.py --jsonl data/processed/test.jsonl --adapter_dir ./checkpoints/qwen2.5vl-3b-gui-lora-stage2/checkpoint-4130 --max_pixels 700000 --out regionfocus_full.jsonl --metrics_out regionfocus_full.json
"""

import argparse
import contextlib
import io
import json
import sys
import time

from tqdm import tqdm

from coord_utils import load_jsonl
from evaluation import aggregate_metrics, format_report
from qwen import QwenVLModel, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS
from region_focus import ground_with_regionfocus
from gui_grounding import ground as local_ground


def _get_category(rec):
    return rec.get("category") or rec.get("element_type") or "unknown"


def _get_platform(rec):
    return rec.get("platform") or "unknown"


def _get_dataset(rec):
    return rec.get("dataset") or rec.get("source") or "unknown"


class _Tee:
    """sys.stdout에 쓰는 내용을 여러 스트림(원래 콘솔 + 캡처용 버퍼)에 동시에 씀."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def score_regionfocus_result(rec: dict, result: dict) -> dict:
    """
    ground_with_regionfocus()/gui_grounding.ground()의 반환 dict(0~1 정규화 point)를
    evaluation.py의 row 스키마(score_prediction()과 동일한 필드)로 변환해서 채점한다.
    """
    resolution = rec["resolution"]
    bbox = rec.get("bbox")

    point_norm = result.get("point")
    parsed_ok = point_norm is not None
    pred_px = None
    if parsed_ok:
        pred_px = [point_norm[0] * resolution[0], point_norm[1] * resolution[1]]

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
        "pred_text": result.get("raw_response"),
        "pred_point_px": pred_px,
        "gt_bbox": bbox,
        "parsed_ok": parsed_ok,
        "hit": hit,
        # region_focus 전용 부가 정보 (디버깅/분석용, 채점(aggregate_metrics)에는 안 쓰임)
        "regionfocus_applied": result.get("regionfocus_applied", False),
        "initial_correct": result.get("initial_correct"),
        "num_candidates": result.get("num_candidates"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="평가할 데이터셋 jsonl (test/val)")
    ap.add_argument("--model_id", default=None, help="베이스 모델 id (기본값: qwen.py의 MODEL_ID)")
    ap.add_argument("--adapter_dir", default=None,
                    help="LoRA 어댑터 디렉토리 (train.py --output_dir로 저장된 checkpoint-XXX 폴더)")
    ap.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="스모크 테스트용 샘플 수 제한")
    ap.add_argument("--no_regionfocus", action="store_true",
                    help="RegionFocus 없이 plain grounding(gui_grounding.ground)만으로 평가 (baseline 비교용)")
    ap.add_argument("--debug", action="store_true", help="region_focus.py의 디버그 이미지 저장(./debug/<id>) 활성화")
    ap.add_argument("--out", default=None, help="샘플별 결과(process_log, elapsed_sec 포함)를 저장할 jsonl 경로")
    ap.add_argument("--metrics_out", default=None, help="집계된 지표를 저장할 json 경로")
    ap.add_argument("--quiet", action="store_true", help="샘플별 hit/miss 한 줄 로그 끄기")
    args = ap.parse_args()

    model_kwargs = dict(
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        adapter_dir=args.adapter_dir,
        load_in_8bit=args.load_in_8bit,
    )
    if args.model_id:
        model_kwargs["model_id"] = args.model_id

    model = QwenVLModel(**model_kwargs)

    records = load_jsonl(args.jsonl)
    if args.limit is not None:
        records = records[: args.limit]
    mode = "plain grounding" if args.no_regionfocus else "RegionFocus"
    print(f"[eval_regionfocus.py] {len(records)}개 샘플 평가 시작 ({mode})")

    rows = []
    total_elapsed = 0.0
    for i, rec in enumerate(records):
        task_id = str(rec.get("id", f"idx{i}"))

        log_buffer = io.StringIO()
        tee = _Tee(sys.stdout, log_buffer)

        t0 = time.time()
        with contextlib.redirect_stdout(tee):
            if args.no_regionfocus:
                result = local_ground(
                    model, rec["instruction"], rec["image_path"],
                    min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                )
            else:
                result = ground_with_regionfocus(
                    model, rec["instruction"], rec["image_path"],
                    debug=args.debug, task_id=task_id,
                    min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                )
        elapsed = time.time() - t0
        total_elapsed += elapsed

        row = score_regionfocus_result(rec, result)
        row["elapsed_sec"] = round(elapsed, 2)
        row["process_log"] = log_buffer.getvalue()
        rows.append(row)

        if not args.quiet:
            status = "O" if row["hit"] else ("?" if not row["parsed_ok"] else "X")
            avg = total_elapsed / (i + 1)
            tqdm.write(
                f"[{i + 1}/{len(records)}] {status} id={row['id']} "
                f"elapsed={elapsed:.1f}s (평균 {avg:.1f}s/샘플) "
                f"pred_point_px={row['pred_point_px']} gt_bbox={row['gt_bbox']}"
            )

    print(
        f"[eval_regionfocus.py] 전체 {len(rows)}개 완료 - "
        f"총 {total_elapsed:.1f}초, 평균 {total_elapsed / max(len(rows), 1):.1f}초/샘플"
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[eval_regionfocus.py] 샘플별 결과 저장: {args.out} (process_log/elapsed_sec 포함)")

    metrics = aggregate_metrics(rows)
    print(format_report(metrics, title=args.jsonl))

    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[eval_regionfocus.py] 지표 저장: {args.metrics_out}")


if __name__ == "__main__":
    main()
