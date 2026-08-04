"""
verifier/generate_verifier_data.py

Verifier(judge) 학습용 라벨 데이터를 TRAIN 데이터로부터 만드는 스크립트.

[왜 반드시 TRAIN인가]
test.jsonl(그리고 train과 겹치지 않는 val.jsonl)은 grounding LoRA/RegionFocus 전체
파이프라인의 최종 채점을 위한 held-out eval set이다. 여기서 verifier 학습 데이터를
뽑으면 grounding LoRA는 못 봤어도 verifier는 test를 보고 학습한 셈이 돼서, 나중에
"RegionFocus + 이 verifier"로 test.jsonl을 다시 채점하면 그 결과가 부풀려진
(데이터 누출된) 수치가 된다. 반드시 train.py가 실제로 학습에 쓴 train 쪽 jsonl
(보통 train_stage2.jsonl)을 넣어야 한다 - 아래 main()에 파일명 기반 경고를 넣어뒀지만
경고일 뿐이니 실행 전에 --jsonl을 직접 눈으로 확인할 것.

[데이터 생성 방식]
grounding LoRA(checkpoint-4130)로 각 샘플마다:
  1) greedy(temperature=0) 1회 + temperature>0로 몇 번 더 샘플링 -> 후보 좌표 여러 개
     (RegionFocus의 region_focus() 재탐색과 같은 원리로, 온도를 올리면 모델이 실제로
     헷갈릴 때 낼 법한 다양한 오답까지 커버할 수 있음)
  2) 각 후보를 gt bbox와 비교해서 label(1=correct/0=incorrect) 매김
  3) gt bbox 중심점도 source="gt_center", label=1로 항상 추가
     (모델이 그 샘플을 한 번도 못 맞췄어도 "진짜 정답은 이렇게 생겼다"는 깨끗한 신호를
     verifier가 반드시 보게 하기 위함 - 순수 모델 예측만 쓰면 애초에 못 맞히는 샘플은
     positive가 아예 없어서 label 불균형이 더 심해짐)

이러면 verifier가 (a) 이 grounding LoRA가 실제로 저지르는 실수 패턴(hard negative)과
(b) gt 기준 명확한 정답(clean positive)을 동시에 학습하게 된다.

[2026-08 배치 추론으로 변경]
처음엔 gui_grounding.ground()를 후보 하나마다(레코드 1개당 최대 4번) 순차 호출했는데,
GPU 한 장에서 model.generate()를 건건이 부르면 배치 효율이 전혀 안 나서 느렸다. 그래서
"레코드 단위 순차 처리"를 "temperature 단위로 여러 레코드를 한 번에 배치 처리"로 뒤집었다:
  - records를 --batch_size개씩 묶고(청크)
  - 청크 안에서 같은 temperature끼리는 messages를 한 번에 만들어서 model.generate()를
    딱 한 번만 호출한다 (레코드마다 따로 부르지 않음)
  - temperature 개수(samples_per_item)만큼 이 배치 호출을 반복
이러면 청크당 model.generate() 호출 횟수가 "레코드 수 x samples_per_item"에서
"samples_per_item"으로 줄어든다(청크 크기만큼 나눠짐). --resume은 레코드 단위가 아니라
청크 단위로 동작한다 - 청크 안 일부만 끝난 상태에서 끊기면 그 청크 전체를 다시 돈다
(batch_size를 너무 크게 안 잡으면 낭비가 작음).

사용법 (vlm_agent/ 상위 모듈을 그대로 재사용하려고 hypo1과 동일하게 PYTHONPATH=..로 실행
- 다만 아래 sys.path 부트스트랩 덕분에 PYTHONPATH 설정 없이도 대부분 자동으로 찾음):
  cd vlm_agent/verifier
  python generate_verifier_data.py \
      --jsonl ../data/processed/verifier_source.jsonl \
      --adapter_dir ../checkpoints/qwen2.5vl-3b-gui-lora-stage2/checkpoint-4130 \
      --samples_per_item 4 --batch_size 4 --max_new_tokens 32 \
      --out verifier_train_raw.jsonl --resume
"""

import argparse
import json
import os
import sys

# vlm_agent(coord_utils.py/qwen.py/gui_grounding.py가 있는 폴더)를 sys.path에 넣는다.
# PYTHONPATH=..를 매번 손으로 설정 안 해도 되게 하려는 것 - verifier/가 hypo1처럼
# vlm_agent/ 바로 밑에 있든(../ = vlm_agent), vlm_agent와 나란히(형제 폴더로) 있든
# (../vlm_agent) 둘 다 자동으로 찾아서 넣는다. 어느 쪽에도 coord_utils.py가 없으면
# (사용자가 완전히 다른 구조로 배치한 경우) 그냥 기존 PYTHONPATH/sys.path에 맡긴다.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = None  # coord_utils.py 등이 실제로 있는 폴더 (아래서 찾아서 채움)
for _candidate in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "vlm_agent")):
    _candidate = os.path.abspath(_candidate)
    if os.path.isfile(os.path.join(_candidate, "coord_utils.py")):
        _BASE_DIR = _candidate
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break

from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from coord_utils import load_jsonl, parse_point_from_text
from qwen import QwenVLModel, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS
from gui_grounding import build_point_prompt_messages


def _resolve_image_path(image_path):
    """
    jsonl에 박힌 image_path는 그 jsonl/이미지를 만들 때의 cwd 기준 상대경로인 경우가
    많다 (prepare_dataset.py, prepare_verifier_dataset.py 둘 다 그렇게 저장함).
    이 스크립트를 어느 cwd에서 실행하든(verifier/ 안에서든, gpu-work 루트에서든)
    이미지를 찾을 수 있도록 순서대로 시도한다:
      1) 절대경로거나 현재 cwd 기준으로 이미 존재하면 그대로 사용
      2) _BASE_DIR(=coord_utils.py가 있는 폴더, 보통 gpu-work 루트) 기준 상대경로로 시도
      3) 앞의 "../"를 하나씩 벗겨가며 _BASE_DIR 기준으로 다시 시도
         (prepare_verifier_dataset.py가 verifier/ 안에서 --out_dir ../data/processed로
         실행됐으면 image_path가 "../data/processed/..."로 저장되는데, 이 스크립트를
         다른 cwd에서 돌리면 그 "../"가 더 이상 안 맞기 때문)
    전부 실패하면 원본 경로를 그대로 반환한다 (에러 메시지에 원래 경로가 그대로 나오게).
    """
    if os.path.isabs(image_path) or os.path.exists(image_path):
        return image_path

    if _BASE_DIR:
        candidate = os.path.join(_BASE_DIR, image_path)
        if os.path.exists(candidate):
            return candidate

        stripped = image_path
        while stripped.startswith("../") or stripped.startswith("..\\"):
            stripped = stripped[3:]
            candidate = os.path.join(_BASE_DIR, stripped)
            if os.path.exists(candidate):
                return candidate

    return image_path


def _hit(point_px, bbox):
    if bbox is None or point_px is None:
        return False
    x1, y1, x2, y2 = bbox
    px, py = point_px
    return (x1 <= px <= x2) and (y1 <= py <= y2)


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def _generate_batch(model, messages_list, max_new_tokens, temperature, top_p=1.0):
    """
    여러 messages(각각 이미지 1장 + 텍스트 프롬프트 1개)를 한 번의 model.generate()
    호출로 배치 처리한다. qwen.generate_text()의 단일 호출 버전과 동일한 로직이되,
    입력을 리스트로 받아 processor에 한 번에 넘긴다는 점만 다르다.

    같은 배치 안의 모든 항목은 같은 temperature를 쓴다 - HF generate()는 배치
    전체에 하나의 do_sample/temperature만 적용 가능해서, 항목별로 다른 temperature를
    같은 배치에 섞을 수 없다 (그래서 generate_candidates_chunk()가 temperature별로
    배치를 나눠 돈다).
    """
    processor = model.processor
    mdl = model.model

    # decoder-only 모델은 배치 생성 시 반드시 왼쪽 패딩이어야 한다 - 오른쪽 패딩이면
    # 프롬프트가 짧은(=패딩이 붙는) 레코드는 generate()가 패딩 토큰 다음 위치부터
    # 이어서 생성해버려서 좌표 출력이 깨질 수 있다. processor 기본값은 오른쪽 패딩이라
    # (HF가 "right-padding detected" 경고를 띄우는 이유) 여기서 명시적으로 바꿔준다.
    # 이 processor는 이 스크립트(배치 생성) 전용이라 다른 곳(train_verifier.py의 오른쪽
    # 패딩 전제 풀링 로직)에는 영향 없음 - 완전히 별개의 processor 인스턴스임.
    processor.tokenizer.padding_side = "left"

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    image_inputs = []
    for m in messages_list:
        imgs, _ = process_vision_info(m)
        image_inputs.extend(imgs)

    inputs = processor(
        text=texts, images=image_inputs, padding=True, return_tensors="pt",
    ).to(mdl.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    generated_ids = mdl.generate(**inputs, **gen_kwargs)
    # 왼쪽 패딩이어도 배치 안 모든 행의 input_ids 길이는 동일하다(패딩으로 맞춰짐) -
    # generate()가 실제 생성 토큰을 항상 전체 입력 길이 뒤에 이어붙이므로, 패딩이
    # 왼쪽에 있든 오른쪽에 있든 이 트림 로직(qwen.generate_text()와 동일)은 그대로 맞다.
    trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def generate_candidates_chunk(model, chunk_recs, samples_per_item, max_new_tokens):
    """
    레코드 청크(최대 --batch_size개) 전체에 대해, temperature별로 배치 호출을 돌려서
    각 레코드의 후보 리스트를 만든다. 반환: chunk_recs와 같은 길이의 리스트, 각 원소는
    generate_candidates()가 레코드 1개에 대해 반환하던 것과 동일한 형식
    ({"point_px","label","source"} 딕셔너리 리스트).
    """
    n = len(chunk_recs)
    resolved_paths = [_resolve_image_path(r["image_path"]) for r in chunk_recs]

    n_sampled = max(samples_per_item - 1, 0)
    temps = [0.0] + [
        round(0.3 + 0.6 * i / max(n_sampled - 1, 1), 2) for i in range(n_sampled)
    ]
    temps = temps[:samples_per_item]

    candidates = [[] for _ in range(n)]
    seen_points = [set() for _ in range(n)]

    for t in temps:
        messages_list = [
            build_point_prompt_messages(chunk_recs[i]["instruction"], resolved_paths[i])
            for i in range(n)
        ]
        texts_out = _generate_batch(model, messages_list, max_new_tokens=max_new_tokens, temperature=t)
        for i, txt in enumerate(texts_out):
            norm_point = parse_point_from_text(txt)
            if norm_point is None:
                continue
            resolution = chunk_recs[i]["resolution"]
            point_px = [norm_point[0] / 1000 * resolution[0], norm_point[1] / 1000 * resolution[1]]
            key = (round(point_px[0]), round(point_px[1]))
            if key in seen_points[i]:
                continue
            seen_points[i].add(key)
            bbox = chunk_recs[i].get("bbox")
            label = 1 if _hit(point_px, bbox) else 0
            candidates[i].append({"point_px": point_px, "label": label, "source": f"model_t{t}"})

    for i, r in enumerate(chunk_recs):
        bbox = r.get("bbox")
        if bbox is not None:
            candidates[i].append({"point_px": _bbox_center(bbox), "label": 1, "source": "gt_center"})

    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True,
                     help="TRAIN 데이터 jsonl (train.py가 실제로 학습에 쓴 파일). "
                          "test/val.jsonl 절대 금지 - 데이터 누출.")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--adapter_dir", default=None, help="grounding LoRA 어댑터 (checkpoint-4130)")
    ap.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--samples_per_item", type=int, default=4,
                     help="샘플 하나당 뽑을 후보 좌표 개수 (greedy 1개 + temperature>0 나머지)")
    ap.add_argument("--batch_size", type=int, default=4,
                     help="한 번의 model.generate() 호출에 묶을 레코드 개수. VRAM 여유가 "
                          "있으면 8~16까지 올려서 더 빠르게 할 수 있음 - OOM 나면 낮출 것.")
    ap.add_argument("--max_new_tokens", type=int, default=32,
                     help="좌표 출력('(823,412)' 같은 형태)은 토큰 몇 개면 충분해서 기본값을 "
                          "128 대신 32로 낮춰둠 - 생성 시간을 직접적으로 줄여줌.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                     help="--out에 이미 처리된 _source_id는 건너뛰고 이어서 실행 (청크 단위 - "
                          "청크 안 레코드가 하나라도 안 끝났으면 청크 전체를 다시 돎)")
    args = ap.parse_args()

    lowered = args.jsonl.lower()
    if "test" in lowered or ("val" in lowered and "train" not in lowered):
        print(
            f"[generate_verifier_data.py] 경고: --jsonl 경로에 'test'/'val'이 들어있음 "
            f"({args.jsonl}) - train 데이터가 맞는지 다시 확인할 것 (verifier가 test/val을 "
            f"보면 나중에 RegionFocus+verifier를 test.jsonl로 채점할 때 데이터 누출이 됨)."
        )

    model_kwargs = dict(
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        adapter_dir=args.adapter_dir,
    )
    if args.model_id:
        model_kwargs["model_id"] = args.model_id
    model = QwenVLModel(**model_kwargs)

    records = load_jsonl(args.jsonl)
    if args.limit is not None:
        records = records[: args.limit]

    done_ids = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line).get("_source_id"))
        print(f"[generate_verifier_data.py] resume: 이미 처리된 source 샘플 {len(done_ids)}개 발견")

    mode = "a" if (args.resume and os.path.exists(args.out)) else "w"
    n_written = 0
    n_chunks = (len(records) + args.batch_size - 1) // args.batch_size
    with open(args.out, mode, encoding="utf-8") as out_f:
        for start in tqdm(range(0, len(records), args.batch_size),
                           total=n_chunks, desc="generate_verifier_data(batched)"):
            chunk = records[start:start + args.batch_size]
            chunk_ids = [r.get("id") for r in chunk]
            if args.resume and all(cid in done_ids for cid in chunk_ids):
                continue

            candidates_list = generate_candidates_chunk(
                model, chunk, args.samples_per_item, args.max_new_tokens,
            )

            for rec, candidates in zip(chunk, candidates_list):
                source_id = rec.get("id")
                # 절대경로가 아니라 _BASE_DIR(vlm_agent 루트) 기준 상대경로로 저장한다.
                # 로컬(Windows)과 서버(Linux) 양쪽에서 이 --out 파일을 학습에 쓸 수 있는데,
                # 두 머신의 절대경로(드라이브 문자, 루트 마운트 경로 등)는 다르지만 vlm_agent
                # 밑의 폴더 구조(data/processed/images/...)는 동일하다는 전제 - 그래서
                # 절대경로를 박아두면 서버에서 로드할 때 깨지고, _BASE_DIR 기준 상대경로면
                # train_verifier.py가 "자기 자신의" _BASE_DIR을 기준으로 다시 풀어서 어느
                # 머신에서든 맞게 찾는다 (아래 os.path.relpath). 슬래시는 "/"로 통일해서
                # Windows에서 만든 jsonl을 Linux 서버에서 읽어도 그대로 동작하게 한다.
                resolved_abs = os.path.abspath(_resolve_image_path(rec["image_path"]))
                if _BASE_DIR:
                    resolved_image_path = os.path.relpath(resolved_abs, _BASE_DIR).replace("\\", "/")
                else:
                    resolved_image_path = resolved_abs.replace("\\", "/")
                for cand in candidates:
                    row = {
                        "_source_id": source_id,
                        "image_path": resolved_image_path,
                        "instruction": rec["instruction"],
                        "resolution": rec["resolution"],
                        "point_px": cand["point_px"],
                        "label": cand["label"],
                        "source": cand["source"],
                        "platform": rec.get("platform"),
                        "category": rec.get("category") or rec.get("element_type"),
                        "dataset": rec.get("dataset") or rec.get("source"),
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_written += 1
            out_f.flush()

    print(f"[generate_verifier_data.py] 완료 - {args.out} (신규 {n_written} rows)")


if __name__ == "__main__":
    main()
