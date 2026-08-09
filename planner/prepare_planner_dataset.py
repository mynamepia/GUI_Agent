"""
prepare_planner_dataset.py

curate_agentnet.py(manifest) + fetch_agentnet_images.py(이미지)의 결과물을 agent/planner.py가
학습해야 할 SFT 포맷(image_path + prompt + target JSON 문자열)으로 변환하고 train/val로
나눈다. train_planner.py가 바로 읽을 수 있는 형태로 만드는 게 이 스크립트의 목표.

[2026-08-07 실측 완료 - action/code 실제 포맷]
gpu-work에서 --inspect_actions로 실제 데이터를 확인했다. 예상보다 훨씬 깨끗했다:
  - action(raw): 이미 좌표 없는 자연어 문장. 예) "Click on the plus (+) button in the
    top-right corner of the browser tab bar to open a new tab."
  - code(raw): pyautogui 호출 문자열, 좌표는 0~1 정규화값. 예) "pyautogui.click(x=0.7575,
    y=0.0553)"
  - thought(raw): 여러 문단짜리 긴 CoT (수백 단어) - 마지막 문단이 보통 "그래서 이 액션을
    고른다"는 결론 요약.
이걸 반영해서 _classify_action_type(code 기반)은 원래 짰던 pyautogui 정규식이 그대로
맞았고, target_description/reasoning 추출 로직만 아래처럼 다시 짰다.

[target_description / reasoning 소스 - 실측 반영]
target_description(클릭류 액션의 "뭘 클릭할지" 자연어 설명, 좌표 아님 - agent/planner.py
상단 docstring 참고)은 action(raw) 필드를 그대로 쓴다 - 이미 정확히 이 용도로 맞는 문장이라
따로 가공할 필요가 없었다. reasoning은 thought의 마지막 문단만 쓴다(_reasoning_from_thought) -
thought 전체를 쓰면 실제 배포 시 짧은 reasoning을 기대하는 프롬프트/few-shot 스타일과 크게
어긋나고 max_new_tokens=300 예산도 빡빡해지기 때문. 파싱 실패(액션 타입 불명 또는 thought가
너무 짧음) 행은 조용히 잘못된 라벨을 만드는 대신 스킵하고 카운트만 남긴다(skipped_unparsed_action).

[history 재구성]
agent/planner.py의 _format_history()를 그대로 import해서 쓴다 - 학습 데이터의 history 포맷과
실제 추론 시점의 history 포맷이 다르면 train/inference distribution mismatch가 생기기 때문에,
포맷 로직을 복제하지 않고 원본 함수를 그대로 재사용한다. 같은 task_id의 manifest 행들을
step_index 순으로 정렬해서, 그 이전 스텝들(이미 파싱에 성공한 것만)을 history_actions로 넘긴다.
품질 필터를 통과 못한 스텝은애초에 manifest에 없어서 history에 약간의 gap이 생길 수 있는데,
이건 심각한 문제가 아니다(실제 배포 환경에서도 history는 완벽하지 않을 수 있음).

사용법:
  # 1) 먼저 실제 포맷 확인 (필수)
  python prepare_planner_dataset.py --inspect_actions

  # 2) 변환 + train/val 분리
  python prepare_planner_dataset.py --convert
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.abspath(os.path.join(_HERE, "..", "agent"))
if _AGENT_DIR not in sys.path and os.path.isfile(os.path.join(_AGENT_DIR, "planner.py")):
    sys.path.insert(0, _AGENT_DIR)

from planner import _ACTIONS, _SYSTEM_PROMPT, _format_history  # noqa: E402


# ---------------------------------------------------------------------------
# 원본 AgentNet action/code -> planner 타깃 스키마 변환 (best-effort, 실측 전까지는 추정)
# ---------------------------------------------------------------------------

_PYAUTOGUI_CLICK_RE = re.compile(
    r"pyautogui\.(click|doubleClick|rightClick)\s*\(", re.IGNORECASE
)
_PYAUTOGUI_DRAG_RE = re.compile(r"pyautogui\.dragTo\s*\(", re.IGNORECASE)
_PYAUTOGUI_TYPE_RE = re.compile(r"pyautogui\.(write|typewrite)\s*\(", re.IGNORECASE)
_PYAUTOGUI_KEY_RE = re.compile(r"pyautogui\.(press|hotkey)\s*\(", re.IGNORECASE)
_PYAUTOGUI_SCROLL_RE = re.compile(r"pyautogui\.scroll\s*\(", re.IGNORECASE)
_QUOTED_RE = re.compile(r"""['"]([^'"]*)['"]""")
_SIGNED_NUMBER_RE = re.compile(r"-?\d+")


def _call_args_str(code: str, call_re) -> str:
    """
    call_re가 매치한 함수 호출("pyautogui.write(" 등) 뒤에서 첫 ')'까지의 인자 문자열을
    반환한다. (2026-08-07 버그 수정) 예전엔 인자가 항상 위치 인자(첫 인자가 바로 따옴표로
    시작)라고 가정했는데, 실측해보니 click은 pyautogui.click(x=0.75, y=0.05)처럼 키워드
    인자 스타일을 쓰고 있었다 - type/key/scroll도 같은 스타일(예: write(message="...")))
    일 가능성이 높아서(실제로 action 분포에 'type'이 0개로 나온 원인으로 추정됨), 위치/키워드
    인자 스타일을 모두 지원하도록 "함수 호출 뒤 첫 quoted string / 첫 숫자"를 찾는 방식으로
    바꿨다. 중첩 괄호가 있는 케이스는 다루지 않는다(pyautogui 인자에 중첩 함수 호출이 오는
    경우는 거의 없음).
    """
    m = call_re.search(code or "")
    if not m:
        return ""
    rest = code[m.end():]
    close_idx = rest.find(")")
    return rest[:close_idx] if close_idx >= 0 else rest

_TERMINATE_WORDS = ("terminate", "done", "finish", "finished", "stop", "complete", "success", "failure")


def _classify_action_type(raw_action, code: str):
    """action/code 원본에서 planner의 _ACTIONS 중 하나로 분류. 실패시 None."""
    # 1) raw_action이 이미 구조화된 dict/문자열이면 흔한 키/값부터 확인
    if isinstance(raw_action, dict):
        t = str(raw_action.get("action_type") or raw_action.get("type") or raw_action.get("action") or "").lower()
        if "double" in t and "click" in t:
            return "double_click"
        if "right" in t and "click" in t:
            return "right_click"
        if "click" in t:
            return "left_click"
        if t in ("type", "input", "write", "input_text"):
            return "type"
        if t in ("key", "hotkey", "press", "keypress"):
            return "key"
        if "scroll" in t:
            return "scroll"
        if t == "wait":
            return "wait"
        if any(w in t for w in _TERMINATE_WORDS):
            return "terminate"

    code = code or ""
    if isinstance(raw_action, str) and not code:
        code = raw_action  # action 자체가 code 문자열인 배포본도 있을 수 있음

    # (2026-08-07 추가) drag는 보통 "pyautogui.moveTo(...)\npyautogui.dragTo(...)" 두 줄
    # 조합으로 나온다(실측 확인, --inspect_skipped로 발견) - dragTo(를 click보다 먼저
    # 체크해야 함(moveTo 자체는 click 정규식과 안 겹치지만, 혹시 모를 순서 문제 방지 차원).
    if _PYAUTOGUI_DRAG_RE.search(code):
        return "drag"
    m = _PYAUTOGUI_CLICK_RE.search(code)
    if m:
        kind = m.group(1).lower()
        if "double" in kind:
            return "double_click"
        if "right" in kind:
            return "right_click"
        return "left_click"
    if _PYAUTOGUI_TYPE_RE.search(code):
        return "type"
    if _PYAUTOGUI_KEY_RE.search(code):
        return "key"
    if _PYAUTOGUI_SCROLL_RE.search(code):
        return "scroll"
    low = code.lower()
    if "wait" in low and "pyautogui" not in low:
        return "wait"
    if any(w in low for w in _TERMINATE_WORDS):
        return "terminate"
    return None


def _extract_type_text(raw_action, code: str):
    if isinstance(raw_action, dict):
        for key in ("text", "value", "content", "message"):
            v = raw_action.get(key)
            if isinstance(v, str) and v:
                return v
    args_str = _call_args_str(code or "", _PYAUTOGUI_TYPE_RE)
    qm = _QUOTED_RE.search(args_str)
    if qm:
        return qm.group(1)
    return None


def _extract_key_text(raw_action, code: str):
    if isinstance(raw_action, dict):
        for key in ("key", "keys", "text"):
            v = raw_action.get(key)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, list) and v:
                parts = [str(x) for x in v if x]
                if parts:
                    return "+".join(parts)
    args_str = _call_args_str(code or "", _PYAUTOGUI_KEY_RE)
    # "'ctrl', 'a'" 같은 다중 인자는 +로 합쳐 표현 (예: "ctrl+a"). 키워드 스타일
    # (keys="enter" 또는 keys=["ctrl","a"])이어도 따옴표로 감싸진 값만 뽑으면 동일하게 동작.
    quoted = _QUOTED_RE.findall(args_str)
    if quoted:
        return "+".join(quoted)
    return None


def _clean_thought(thought):
    if not thought or not isinstance(thought, str):
        return None
    t = thought.strip()
    if len(t) < 5:
        return None
    return t


# (2026-08-07 실측 확정) --inspect_actions로 실제 AgentNet 데이터를 확인한 결과,
# manifest의 "action" 필드는 이미 좌표 없는 깨끗한 자연어 문장이었다(예: "Click on the
# plus (+) button in the top-right corner of the browser tab bar to open a new tab.").
# 이게 정확히 planner.py가 필요로 하는 target_description 그 자체라서, 처음 짰던
# "reasoning(=thought)을 재사용" 방식보다 훨씬 낫다 - 이제 target_description은 action
# 필드에서, reasoning은 thought에서 따로 뽑는다.
#
# 다만 thought는 여러 문단짜리 긴 CoT라(문단 여러 개, 합쳐서 수백 단어) 그대로
# reasoning으로 쓰면 실제 배포 시의 짧은 한두 문장짜리 reasoning(FEWSHOT 예시 참고)과
# 스타일이 크게 어긋나고, planner.py의 max_new_tokens=300 예산 안에서 target_description+
# 다른 필드까지 다 채울 여유가 부족해질 수 있다. thought는 보통 마지막 문단이 "그래서
# 이 액션을 고른다"는 결론 요약이라(여러 샘플에서 일관되게 확인됨), 문단(빈 줄 기준)을
# 나눠서 마지막 문단만 reasoning으로 쓰고, 그래도 너무 길면 문장 단위로 잘라낸다.
def _reasoning_from_thought(thought, max_chars: int = 500):
    t = _clean_thought(thought)
    if t is None:
        return None
    paragraphs = [p.strip() for p in t.split("\n\n") if p.strip()]
    reasoning = paragraphs[-1] if paragraphs else t
    if len(reasoning) > max_chars:
        truncated = reasoning[:max_chars]
        last_period = truncated.rfind(". ")
        reasoning = truncated[: last_period + 1] if last_period > 0 else truncated.rstrip() + "..."
    return reasoning


def convert_row(row: dict, prior_actions: list):
    """
    manifest 행 하나 -> (target_dict, history_actions_for_this_row) 또는 (None, None) if 스킵.

    target_dict는 agent/planner.py 출력 스키마와 동일한 필드만 채운다(해당 없는 필드는
    아예 넣지 않음 - 실제 추론 출력 분포와 맞추기 위해서. _FEWSHOT 예시와 동일한 관례).
    """
    raw_action = row.get("action")
    code = row.get("code") or ""
    action_type = _classify_action_type(raw_action, code)
    if action_type is None or action_type not in _ACTIONS:
        return None, None

    reasoning = _reasoning_from_thought(row.get("thought")) or _clean_thought(row.get("observation"))
    if reasoning is None:
        return None, None

    target = {"reasoning": reasoning, "action": action_type}

    if action_type in ("left_click", "double_click", "right_click"):
        # action(raw) 필드가 이미 "뭘 왜 클릭하는지"를 설명하는 자연어 문장이라 이걸 그대로
        # target_description으로 쓴다(좌표는 code 쪽에만 있고 여기엔 안 섞여 있음, 실측 확인됨).
        target_desc = raw_action.strip() if isinstance(raw_action, str) else None
        if not target_desc:
            # action(raw)이 비어있거나 문자열이 아닌 예외 케이스 - reasoning으로 폴백
            target_desc = reasoning
        target["target_description"] = target_desc
    elif action_type == "drag":
        # (2026-08-07 추가) planner.py의 drag 스키마는 target_description(시작 지점)/
        # text(끝 지점)를 따로 요구하는데, AgentNet의 action(raw) 문장은 시작/끝을 한
        # 문장으로 뭉쳐서 설명한다("Drag from the fill handle... down to cell A13...") -
        # 문자열 파싱으로 안전하게 둘로 쪼갤 방법이 없어서, 일단 같은 문장을 두 필드에
        # 모두 넣는다(완벽하진 않지만 최소한 "무엇을 어디로"라는 의미 정보는 두 필드
        # 어디를 봐도 보존됨 - placeholder 문자열보다 안전한 선택). 나중에 필요하면
        # thought/action 문장에서 "from ... to ..." 패턴을 정규식으로 분리하는 걸
        # 시도해볼 수 있음.
        target_desc = raw_action.strip() if isinstance(raw_action, str) else None
        if not target_desc:
            target_desc = reasoning
        target["target_description"] = target_desc
        target["text"] = target_desc
    elif action_type == "type":
        text = _extract_type_text(raw_action, code)
        if not text:
            return None, None
        target["text"] = text
    elif action_type == "key":
        text = _extract_key_text(raw_action, code)
        if not text:
            return None, None
        target["text"] = text
    elif action_type == "scroll":
        # position/keyword 인자 둘 다 지원 (_call_args_str 참고 - click이 keyword 스타일을
        # 썼던 것과 같은 이유로 scroll도 clicks=-500 같은 keyword 스타일일 수 있음)
        args_str = _call_args_str(code, _PYAUTOGUI_SCROLL_RE)
        nm = _SIGNED_NUMBER_RE.search(args_str)
        direction = "down"
        if nm:
            direction = "up" if int(nm.group(0)) > 0 else "down"
        elif isinstance(raw_action, dict) and raw_action.get("direction") in ("up", "down"):
            direction = raw_action["direction"]
        target["text"] = direction
    elif action_type == "wait":
        pass
    elif action_type == "terminate":
        # 이 데이터는 항상 성공 궤적만 통과했으므로(_quality_ok_traj: task_completed=True)
        # terminate 스텝은 success로 라벨링한다.
        target["status"] = "success"

    return target, list(prior_actions)


def inspect_actions(manifest_path: str, n: int = 8):
    seen = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            print(f"\n===== manifest row (task_id={row.get('task_id')}, step_index={row.get('step_index')}) =====")
            print("platform:", row.get("platform"), " archive_group:", row.get("archive_group"))
            print("action (raw):", repr(row.get("action")))
            print("code (raw):", repr(row.get("code")))
            print("thought (raw):", repr(row.get("thought")))
            print("observation (raw):", repr(row.get("observation")))
            classified = _classify_action_type(row.get("action"), row.get("code") or "")
            print(f"-> _classify_action_type 결과(추정): {classified}")
            seen += 1
            if seen >= n:
                break
    if seen == 0:
        print(f"manifest가 비어있음: {manifest_path}")
    print(
        "\n[다음 할 일] 위 'action (raw)'/'code (raw)' 값이 _classify_action_type()이 기대하는 "
        "포맷과 실제로 맞는지 확인하세요. '-> _classify_action_type 결과'가 None이거나 이상하면 "
        "이 스크립트의 _classify_action_type/_extract_type_text/_extract_key_text를 실제 포맷에 "
        "맞게 고쳐야 합니다."
    )


_CODE_FUNC_RE = re.compile(r"pyautogui\.(\w+)\s*\(")


def inspect_skipped(manifest_path: str, n: int = 15):
    """
    (2026-08-07 추가) convert --convert 결과 action 분포에 'type'이 하나도 안 잡히는 등
    특정 액션 종류가 통째로 skipped_unparsed_action에 묻혀있을 수 있어서 만든 진단 모드.
    _classify_action_type이 None을 반환한 행들만 모아서: (1) 실제 code에 쓰인 pyautogui
    함수 이름 빈도, (2) 그 중 몇 개 샘플의 raw action/code/thought를 그대로 보여준다.
    """
    func_counts = defaultdict(int)
    no_func_match = 0
    samples = []
    total = 0
    skipped = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            raw_action = row.get("action")
            code = row.get("code") or ""
            classified = _classify_action_type(raw_action, code)
            if classified is not None and classified in _ACTIONS:
                continue
            skipped += 1
            m = _CODE_FUNC_RE.search(code)
            if m:
                func_counts[m.group(1)] += 1
            else:
                no_func_match += 1
            if len(samples) < n:
                samples.append(row)

    print(f"[inspect_skipped] 전체 {total}개 중 _classify_action_type 실패(스킵 후보) {skipped}개")
    print("[inspect_skipped] 스킵된 행들의 code에서 발견된 pyautogui 함수 이름 빈도:")
    for fn, cnt in sorted(func_counts.items(), key=lambda kv: -kv[1]):
        print(f"  pyautogui.{fn}(...): {cnt}개")
    print(f"  (pyautogui.* 패턴 자체가 code에 없음: {no_func_match}개)")

    print(f"\n[inspect_skipped] 샘플 {len(samples)}개:")
    for row in samples:
        print(f"\n----- task_id={row.get('task_id')} step_index={row.get('step_index')} -----")
        print("action (raw):", repr(row.get("action")))
        print("code (raw):", repr(row.get("code")))
    print(
        "\n[다음 할 일] 위 함수 이름 빈도에서 'write'/'typewrite'/'press'/'hotkey'/'scroll' 등이 "
        "보이는데도 action 분포에 해당 종류가 안 잡혔다면 _classify_action_type의 정규식이 "
        "실제 code 문법(인자 순서/따옴표 스타일 등)과 안 맞는 것 - 위 code (raw) 샘플을 보고 "
        "정규식을 맞게 고쳐야 함."
    )


def convert_manifest(
    manifest_path: str,
    images_dir: str,
    out_dir: str,
    val_ratio: float,
    seed: int,
):
    rows_by_task = defaultdict(list)
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_task[row["task_id"]].append(row)

    examples = []
    stats = defaultdict(int)
    for task_id, rows in rows_by_task.items():
        rows.sort(key=lambda r: (r.get("step_index") if r.get("step_index") is not None else 0))
        history_actions = []
        for row in rows:
            stats["total_steps"] += 1
            image_name = row.get("image")
            platform = row.get("platform")
            if not image_name or not platform:
                stats["skipped_missing_image_field"] += 1
                continue
            image_path = os.path.join(images_dir, platform, os.path.basename(image_name))
            if not os.path.isfile(image_path):
                stats["skipped_image_not_found"] += 1
                continue

            target, hist_snapshot = convert_row(row, history_actions)
            if target is None:
                stats["skipped_unparsed_action"] += 1
                continue

            history_text = _format_history(hist_snapshot)
            examples.append(
                {
                    "task_id": task_id,
                    "platform": platform,
                    "image_path": image_path,
                    "instruction": row.get("instruction") or "",
                    "history_text": history_text,
                    "target": target,
                }
            )
            stats["kept"] += 1
            # 다음 스텝의 history에는 이번 스텝(파싱 성공분만)을 누적
            history_actions.append(target)

    print("[prepare_planner_dataset] 변환 통계:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    if not examples:
        sys.exit(
            "변환된 예제가 0개입니다 - _classify_action_type이 실제 action/code 포맷을 못 잡고 "
            "있을 가능성이 높습니다. --inspect_actions로 실제 포맷을 먼저 확인하세요."
        )

    # platform별 stratified split
    random.seed(seed)
    by_platform = defaultdict(list)
    for ex in examples:
        by_platform[ex["platform"]].append(ex)

    train_examples, val_examples = [], []
    for platform, exs in by_platform.items():
        random.shuffle(exs)
        n_val = max(1, int(len(exs) * val_ratio)) if len(exs) > 5 else 0
        val_examples.extend(exs[:n_val])
        train_examples.extend(exs[n_val:])

    random.shuffle(train_examples)
    random.shuffle(val_examples)

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "planner_train.jsonl")
    val_path = os.path.join(out_dir, "planner_val.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n[prepare_planner_dataset] train: {len(train_examples)}개 -> {train_path}")
    print(f"[prepare_planner_dataset] val:   {len(val_examples)}개 -> {val_path}")
    action_counts = defaultdict(int)
    for ex in examples:
        action_counts[ex["target"]["action"]] += 1
    print("[prepare_planner_dataset] action 분포:", dict(action_counts))


def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/planner_agentnet_manifest.jsonl")
    ap.add_argument("--images_dir", default="data/processed/images/agentnet")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--inspect_actions", action="store_true", help="원본 action/code 포맷을 먼저 눈으로 확인 (필수 선행 단계)")
    ap.add_argument("--inspect_skipped", action="store_true",
                     help="_classify_action_type이 실패한(스킵되는) 행들만 모아서 원인 진단 "
                          "(예: 특정 action 종류가 통째로 안 잡히는 경우 확인용)")
    ap.add_argument("--convert", action="store_true", help="변환 + train/val 분리 실행")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_inspect", type=int, default=8)
    args = ap.parse_args()

    if not (args.inspect_actions or args.inspect_skipped or args.convert):
        ap.error("--inspect_actions, --inspect_skipped, --convert 중 하나는 지정해야 합니다.")

    if not os.path.exists(args.manifest):
        sys.exit(f"manifest 없음: {args.manifest} (curate_agentnet.py --curate 먼저 실행)")

    if args.inspect_actions:
        inspect_actions(args.manifest, n=args.n_inspect)
    if args.inspect_skipped:
        inspect_skipped(args.manifest, n=args.n_inspect)
    if args.convert:
        convert_manifest(args.manifest, args.images_dir, args.out_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    _cli()
