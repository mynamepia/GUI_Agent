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

[Resume 지원]
RegionFocus는 1272개 전체를 돌리면 샘플당 ~14초만 잡아도 5시간 가까이 걸려서,
중간에 끊기거나(Ctrl+C, 재부팅, 크래시) 일부러 나눠서 돌리고 싶은 경우가 생긴다.
--resume을 켜면:
    1) --out 파일이 이미 있으면 그 안의 row들을 읽어와서 각 row의 "idx"(records
       리스트에서의 0-based 위치)를 기준으로 "이미 끝난 샘플" 집합을 만든다.
    2) 메인 루프는 그 idx들을 건너뛰고 남은 샘플만 추론한다.
    3) 매 샘플이 끝날 때마다 --out에 바로 한 줄씩 append + flush한다(끝까지
       기다렸다가 한 번에 쓰지 않음) - 그래서 다음에 또 끊겨도 그 시점까지는
       디스크에 남아있다.
    4) 최종 metrics(--metrics_out)는 이번에 새로 처리한 것 + 기존에 이미
       끝나 있던 것을 합쳐서 계산한다.
--resume 없이 그냥 실행하면 이전과 동일하게 처음부터 끝까지 돌고 --out은
한 번에 덮어써진다(처음 실행할 때는 --resume을 켜도 안 켜도 결과는 같음 -
기존 파일이 없으니까. 나눠서/이어서 돌릴 계획이면 첫 실행부터 --resume을
같이 켜두는 걸 추천).

[min_crop_config: 2026-07 추가]
judge_inference의 zoom-in crop 최소 크기(min_crop_px)를 ScreenSpot-v2 자체 해상도
분포로 보정한 값을 쓰기 위한 옵션. calibrate_min_crop.py를 미리 한 번 돌려서 만든
min_crop_config.json(global_min_crop_px + per_platform_min_crop_px)을 --min_crop_config로
넘기면, 각 record의 platform(_get_platform)에 맞는 min_crop_px를 찾아서
ground_with_regionfocus(..., min_crop_px=...)로 넘겨준다. platform이 config에
없으면 global 값으로, --min_crop_config 자체를 안 주면 region_focus.py의 기본값을
그대로 쓴다(이 스크립트에서 강제로 값을 주입하지 않음).

주의: RegionFocus는 한 샘플당 judge/재탐색/crop-zoom 4회/aggregation까지
model.generate()를 최대 7~8회 호출할 수 있어서, plain grounding(1회 호출) 대비
훨씬 느리다. 전체 1272개를 한 번에 돌리기 전에 --limit 20~30 정도로 먼저
스모크 테스트하는 걸 추천한다.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time

from tqdm import tqdm

from coord_utils import load_jsonl
from evaluation import aggregate_metrics, format_report
from qwen import QwenVLModel, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS
from region_focus_WJ import ground_with_regionfocus
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


def _load_existing_rows(out_path):
    """--out 파일이 이미 있으면 그 안의 row들을 읽어서 반환 (resume용). 없으면 빈 리스트."""
    rows = []
    if out_path and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _load_min_crop_config(path):
    """
    calibrate_min_crop.py가 만든 min_crop_config.json을 로드.
    {"global_min_crop_px": int, "per_platform_min_crop_px": {"mobile": int, ...}, ...} 형태.
    path가 None이면 None을 반환(= region_focus.py 자체 기본값을 그대로 쓰겠다는 뜻).
    """
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(
        f"[eval_regionfocus.py] min_crop_config 로드: {path} "
        f"(global={cfg.get('global_min_crop_px')}, "
        f"platforms={list(cfg.get('per_platform_min_crop_px', {}).keys())})"
    )
    return cfg


def _lookup_min_crop_px(min_crop_cfg, platform):
    """
    record의 platform에 맞는 min_crop_px를 min_crop_cfg에서 찾는다.
    - min_crop_cfg가 None이면(=--min_crop_config를 안 줬으면) None을 반환해서
      ground_with_regionfocus 호출 시 이 인자를 아예 안 넘기게 한다(= region_focus.py
      자체 기본값 사용).
    - platform이 per_platform_min_crop_px에 없으면 global_min_crop_px로 fallback.
    - 그것도 없으면 None(= 역시 region_focus.py 기본값 사용).
    """
    if not min_crop_cfg:
        return None
    per_platform = min_crop_cfg.get("per_platform_min_crop_px", {})
    if platform in per_platform:
        return per_platform[platform]
    return min_crop_cfg.get("global_min_crop_px")


def score_regionfocus_result(rec: dict, result: dict, min_crop_px=None) -> dict:
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
        "min_crop_px_used": min_crop_px,  # 이번 샘플의 judge zoom-in에 실제로 쓰인 min_crop_px (None=region_focus.py 기본값)
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
    ap.add_argument("--debug_image", action="store_true",
                    help="region_focus.py의 중간 이미지 저장(./debug/<id>/*.png) 활성화")
    ap.add_argument("--debug_text", action="store_true",
                    help="각 단계에 실제로 들어간 프롬프트+응답 원문을 ./debug/<id>/prompt_*.txt로 저장")
    ap.add_argument("--debug_mode", choices=["always", "incorrect"], default="always",
                    help="always: 판정과 무관하게 항상 저장 / incorrect: judge가 오답으로 판단한 "
                         "샘플만 저장 (정답 조기종료 샘플은 스킵, RegionFocus 모드에서만 의미 있음)")
    ap.add_argument("--min_crop_config", default=None,
                    help="calibrate_min_crop.py로 만든 min_crop_config.json 경로. 주면 record의 "
                         "platform에 맞는 min_crop_px를 ground_with_regionfocus에 넘겨줌. "
                         "안 주면 region_focus.py 자체 기본값을 그대로 씀.")
    ap.add_argument("--out", default=None, help="샘플별 결과(process_log, elapsed_sec 포함)를 저장할 jsonl 경로")
    ap.add_argument("--metrics_out", default=None, help="집계된 지표를 저장할 json 경로")
    ap.add_argument("--quiet", action="store_true", help="샘플별 hit/miss 한 줄 로그 끄기")
    ap.add_argument("--resume", action="store_true",
                    help="--out에 이미 저장된 샘플은 건너뛰고 이어서 실행 (매 샘플마다 --out에 바로 append)")
    args = ap.parse_args()

    if args.resume and not args.out:
        ap.error("--resume을 쓰려면 --out이 필요함 (이어할 대상 파일이 있어야 함)")

    model_kwargs = dict(
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        adapter_dir=args.adapter_dir,
        load_in_8bit=args.load_in_8bit,
    )
    if args.model_id:
        model_kwargs["model_id"] = args.model_id

    model = QwenVLModel(**model_kwargs)

    min_crop_cfg = _load_min_crop_config(args.min_crop_config)

    records = load_jsonl(args.jsonl)
    if args.limit is not None:
        records = records[: args.limit]
    mode = "plain grounding" if args.no_regionfocus else "RegionFocus"

    # --- resume: 기존 --out에서 이미 끝난 샘플의 idx를 읽어온다 ---
    existing_rows = _load_existing_rows(args.out) if args.resume else []
    done_idx = {r["idx"] for r in existing_rows if "idx" in r}
    rows = list(existing_rows)  # 최종 metrics는 기존 것 + 새로 처리한 것을 합쳐서 계산

    if done_idx:
        print(f"[eval_regionfocus.py] resume: 이미 완료된 {len(done_idx)}개 발견, 나머지만 이어서 진행")

    remaining = [(i, rec) for i, rec in enumerate(records) if i not in done_idx]
    print(
        f"[eval_regionfocus.py] 전체 {len(records)}개 중 이번 실행에서 {len(remaining)}개 처리 예정 ({mode})"
    )

    # --out 파일 핸들: resume이면 이어쓰기(append), 아니면 새로 씀(덮어쓰기 후 매 줄 append)
    out_f = None
    if args.out:
        out_f = open(args.out, "a" if (args.resume and os.path.exists(args.out)) else "w", encoding="utf-8")

    total_elapsed = 0.0
    try:
        for n, (i, rec) in enumerate(tqdm(remaining, desc=mode)):
            task_id = str(rec.get("id", f"idx{i}"))

            # 이 record의 platform에 맞는 min_crop_px 조회 (min_crop_cfg가 None이면 항상 None)
            min_crop_px = _lookup_min_crop_px(min_crop_cfg, _get_platform(rec))

            log_buffer = io.StringIO()
            tee = _Tee(sys.stdout, log_buffer)

            t0 = time.time()
            with contextlib.redirect_stdout(tee):
                if args.no_regionfocus:
                    result = local_ground(
                        model, rec["instruction"], rec["image_path"],
                        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                        debug_text=args.debug_text, task_id=task_id,
                    )
                else:
                    rf_kwargs = dict(
                        debug_image=args.debug_image, debug_text=args.debug_text,
                        debug_mode=args.debug_mode, task_id=task_id,
                        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                    )
                    if min_crop_px is not None:
                        # None을 명시적으로 넘기면 region_focus.py의 기본값(int)을 덮어써서
                        # 에러가 날 수 있으니, 값이 있을 때만 kwargs에 추가한다.
                        rf_kwargs["min_crop_px"] = min_crop_px
                    result = ground_with_regionfocus(
                        model, rec["instruction"], rec["image_path"], **rf_kwargs
                    )
            elapsed = time.time() - t0
            total_elapsed += elapsed

            row = score_regionfocus_result(rec, result, min_crop_px=min_crop_px)
            row["idx"] = i  # resume 시 "이미 끝난 샘플" 판별용 - records 리스트에서의 위치
            row["elapsed_sec"] = round(elapsed, 2)
            row["process_log"] = log_buffer.getvalue()
            rows.append(row)

            if out_f:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()  # 중간에 끊겨도 여기까지는 디스크에 남도록 즉시 flush

            if not args.quiet:
                status = "O" if row["hit"] else ("?" if not row["parsed_ok"] else "X")
                avg = total_elapsed / (n + 1)
                tqdm.write(
                    f"[{i + 1}/{len(records)}] {status} id={row['id']} "
                    f"elapsed={elapsed:.1f}s (이번 실행 평균 {avg:.1f}s/샘플) "
                    f"min_crop_px={min_crop_px} "
                    f"pred_point_px={row['pred_point_px']} gt_bbox={row['gt_bbox']}"
                )
    finally:
        if out_f:
            out_f.close()

    print(
        f"[eval_regionfocus.py] 이번 실행 {len(remaining)}개 완료 - "
        f"총 {total_elapsed:.1f}초, 평균 {total_elapsed / max(len(remaining), 1):.1f}초/샘플 "
        f"(전체 누적 {len(rows)}/{len(records)}개)"
    )
    if args.out:
        print(f"[eval_regionfocus.py] 샘플별 결과 저장(누적): {args.out} (process_log/elapsed_sec 포함)")

    if len(rows) < len(records):
        print(
            f"[eval_regionfocus.py] 주의: 아직 {len(records) - len(rows)}개 미완료 - "
            f"동일 명령에 --resume을 붙여서 다시 실행하면 이어서 처리됨. "
            f"지금은 완료된 {len(rows)}개만으로 잠정 metrics를 계산함."
        )

    metrics = aggregate_metrics(rows)
    print(format_report(metrics, title=args.jsonl))

    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[eval_regionfocus.py] 지표 저장: {args.metrics_out}")


if __name__ == "__main__":
    main()