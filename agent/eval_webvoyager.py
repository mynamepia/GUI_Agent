"""
eval_webvoyager.py

WebVoyager 스타일 태스크에 대해 에이전트 trajectory를 수집하고, "성공했는가"를 judge에게
물어서 채점하는 배치 평가 하네스. eval_regionfocus.py(정답 bbox가 있는 grounding 배치
평가)와 짝을 이루는, "정답이 없는 open-ended 태스크"용 버전.

[judge는 pluggable]
judge_fn 시그니처:
    judge_fn(instruction: str, screenshots: list[PIL.Image], final_answer: str | None)
        -> {"success": bool, "raw_response": str, "reason": str | None}

지금 제공하는 구현 두 가지:
    make_qwen_judge(qwen_model)   - 로컬 Qwen2.5-VL(qwen.py의 QwenVLModel), 비용 없이 바로
                                     테스트 가능. 다만 이 프로젝트가 처음에 겪은 "judge
                                     신뢰도" 문제가 task-level 판정(여러 스텝짜리 trajectory
                                     이해)에서는 더 심하게 재현될 수 있음 - 실제로 붙여보고
                                     신뢰도 확인 필요.
    make_openai_judge(model=...)  - OpenAI vision API(GPT-4V/GPT-4o 등). WebVoyager/
                                     RegionFocus 논문과 동일한 방식. openai 패키지는 judge_fn을
                                     "실행"하는 시점에만 필요(lazy import) - 이 파일 자체는
                                     openai 미설치 환경에서도 import/selftest 가능.
둘 다 이번 세션에서는 실제 모델/API 호출까지 검증하지 못했음(무거운 모델 로드/API 키가
필요해서). trajectory 수집 및 다수결 집계 로직은 mock으로 단위 테스트해뒀다
(`python eval_webvoyager.py --selftest`).

[planner/agent_loop가 아직 없음]
run_episode()는 "다음에 뭘 할지" 정하는 agent_step_fn을 파라미터로 받는다 - 이 파일은
그 함수가 뭐든 상관없이 trajectory 수집/judge 채점만 담당한다. 지금은 placeholder로
dummy_agent_step()(항상 즉시 terminate)만 있고, planner/agent_loop.py가 완성되면
거기서 만든 정책 함수를 그대로 넣으면 된다.

[WebVoyager 논문 관례 반영]
- judge에는 trajectory 전체가 아니라 마지막 15장 스크린샷만 넘김(원 논문과 동일,
  MAX_JUDGE_SCREENSHOTS).
- judge를 한 번만 믿지 않고 기본 3회 반복 후 다수결로 최종 판정(원 논문이 "GPT judge를
  3회 돌려 평균/표준편차를 보고"한 것의 실용적 버전 - 여기서는 각 태스크의 최종 success를
  다수결로 정하고, judge별 개별 응답도 다 로그에 남겨서 나중에 judge 자체의 변동성을 따로
  볼 수 있게 함).

[아직 없는 것 - 다음 단계]
--resume 지원(eval_regionfocus.py에는 있음)은 아직 없음. 태스크 수가 많아지고 오래
걸리기 시작하면 그때 같은 방식으로 추가할 것.
"""

import argparse
import base64
import io
import json
import os
import re
import time
from collections import Counter

from PIL import Image

from env_webvoyager import WebVoyagerEnv, load_webvoyager_tasks

MAX_JUDGE_SCREENSHOTS = 15
DEFAULT_MAX_STEPS = 15
DEFAULT_JUDGE_REPEATS = 3


# ---------------------------------------------------------------------------
# placeholder agent (planner/agent_loop.py 완성 전까지 스모크 테스트용)
# ---------------------------------------------------------------------------
def dummy_agent_step(screenshot, task_info, history):
    """
    실제 planner가 아직 없을 때 run_episode()를 테스트하기 위한 최소 placeholder.
    항상 바로 종료(terminate, status=failure)한다 - 파이프라인 배선 확인용일 뿐,
    실제 태스크를 풀지는 않는다.
    """
    return {"action": "terminate", "status": "failure"}


# ---------------------------------------------------------------------------
# trajectory 수집
# ---------------------------------------------------------------------------
def run_episode(env: WebVoyagerEnv, task, agent_step_fn, max_steps=DEFAULT_MAX_STEPS):
    """
    task를 env에 reset하고, agent_step_fn이 "terminate"를 낼 때까지(또는 max_steps
    도달까지) 액션을 실행한다.

    agent_step_fn(screenshot, task_info, history) -> action dict
        (gui_grounding.ComputerUseTool 스키마). "terminate" 액션이 나오면 그 자리에서
        멈춘다 - env.execute_action()에는 안 보냄(env_webvoyager.py가 terminate를
        거부하도록 만들어져 있으므로 여기서 걸러야 함).

    Returns: dict {
        "instruction": str, "url": str,
        "screenshots": [PIL.Image, ...]   # 스텝별 전체 - judge에는 마지막 N장만 넘길 것
        "actions": [action_dict, ...],
        "final_answer": str | None,       # terminate action의 "text" 필드(있으면)
        "n_steps": int,
        "hit_max_steps": bool,
    }
    """
    screenshot, task_info = env.reset(task)
    screenshots = [screenshot]
    actions = []
    final_answer = None
    hit_max_steps = True

    for _ in range(max_steps):
        action = agent_step_fn(screenshot, task_info, {"actions": actions, "screenshots": screenshots})
        if action.get("action") == "terminate":
            final_answer = action.get("text")
            actions.append(action)
            hit_max_steps = False
            break
        screenshot, reward, terminated, truncated, task_info = env.execute_action(action)
        screenshots.append(screenshot)
        actions.append(action)
        if terminated or truncated:
            hit_max_steps = False
            break

    return {
        "instruction": task_info["instruction"],
        "url": task_info["url"],
        "screenshots": screenshots,
        "actions": actions,
        "final_answer": final_answer,
        "n_steps": len(actions),
        "hit_max_steps": hit_max_steps,
    }


# ---------------------------------------------------------------------------
# judge 인터페이스 + 구현체 2종
# ---------------------------------------------------------------------------
def _parse_success_verdict(response: str):
    """
    judge 응답에서 {"reason": "...", "success": true/false} JSON을 파싱.
    region_focus._parse_judge_verdict()와 같은 원칙(JSON 우선, 실패시 substring 폴백,
    완전 실패면 안전하게 False)을 success 판정용으로 재구현 - 모듈 간 결합을 줄이려고
    이 파일에 로컬로 따로 둠(region_focus.py를 import하지 않음).
    """
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if "success" in obj:
                return bool(obj["success"]), obj.get("reason")
        except (json.JSONDecodeError, AttributeError):
            pass
    upper = response.upper()
    if "SUCCESS" in upper and "NOT SUCCESS" not in upper and "UNSUCCESS" not in upper:
        return True, None
    return False, None  # 파싱 완전 실패시 안전하게 실패 처리 (judge_inference와 동일한 원칙)


_JUDGE_PROMPT_TEMPLATE = (
    'Task instruction: "{instruction}"\n'
    "You are shown the last {n} screenshots of an agent's browsing session while attempting "
    "this task (in chronological order; the last one is the final state).\n"
    "{answer_line}"
    'Reply with ONLY this JSON: {{"reason": "<short reason>", "success": true/false}}\n'
    "Be strict: only mark success if the screenshots (and answer, if given) clearly show the "
    "task instruction was fulfilled."
)


def _build_judge_prompt(instruction, n_screenshots, final_answer):
    answer_line = f'The agent\'s final answer text was: "{final_answer}"\n' if final_answer else ""
    return _JUDGE_PROMPT_TEMPLATE.format(instruction=instruction, n=n_screenshots, answer_line=answer_line)


def make_qwen_judge(qwen_model, max_new_tokens=256):
    """로컬 Qwen2.5-VL(qwen.py의 QwenVLModel)을 judge로 쓰는 judge_fn을 만들어 반환."""

    def judge_fn(instruction, screenshots, final_answer):
        imgs = screenshots[-MAX_JUDGE_SCREENSHOTS:]
        prompt = _build_judge_prompt(instruction, len(imgs), final_answer)
        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        response = qwen_model.generate(messages, max_new_tokens=max_new_tokens, temperature=0.0)
        success, reason = _parse_success_verdict(response)
        return {"success": success, "raw_response": response, "reason": reason}

    return judge_fn


def make_openai_judge(model="gpt-4o", api_key=None, max_tokens=300):
    """
    OpenAI vision API(GPT-4V/GPT-4o 등)를 judge로 쓰는 judge_fn을 만들어 반환.
    WebVoyager/RegionFocus 논문과 동일한 방식. api_key=None이면 환경변수 OPENAI_API_KEY를
    사용(openai 클라이언트 기본 동작). openai 패키지는 반환된 judge_fn을 실제로 호출하는
    시점에만 필요 - make_openai_judge() 자체나 이 파일 import는 openai 미설치 환경에서도
    문제없다.
    """

    def judge_fn(instruction, screenshots, final_answer):
        from openai import OpenAI  # 실제 호출 시점에만 필요 (lazy import)

        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        imgs = screenshots[-MAX_JUDGE_SCREENSHOTS:]
        prompt = _build_judge_prompt(instruction, len(imgs), final_answer)

        content = [{"type": "text", "text": prompt}]
        for img in imgs:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
        response_text = resp.choices[0].message.content
        success, reason = _parse_success_verdict(response_text)
        return {"success": success, "raw_response": response_text, "reason": reason}

    return judge_fn


def run_judge_with_repeats(judge_fn, instruction, screenshots, final_answer, n_repeats=DEFAULT_JUDGE_REPEATS):
    """
    judge_fn을 n_repeats번 호출해서 다수결로 최종 success를 정한다(RegionFocus 논문이
    "GPT judge를 3회 돌려 평균/표준편차 보고"한 것의 실용적 버전). 개별 판정도 다 보존해서
    나중에 judge 자체의 변동성(분산)을 따로 분석할 수 있게 한다.

    Returns: {"success": bool, "votes": [bool, ...], "agreement": float, "runs": [judge_fn 결과, ...]}
    """
    runs = [judge_fn(instruction, screenshots, final_answer) for _ in range(n_repeats)]
    votes = [r["success"] for r in runs]
    success = Counter(votes).most_common(1)[0][0]
    agreement = sum(1 for v in votes if v == success) / len(votes)
    return {"success": success, "votes": votes, "agreement": agreement, "runs": runs}


# ---------------------------------------------------------------------------
# 배치 실행 (eval_regionfocus.py와 유사한 구조 - --resume은 아직 미구현)
# ---------------------------------------------------------------------------
def run_batch(tasks, env, agent_step_fn, judge_fn, max_steps=DEFAULT_MAX_STEPS,
              judge_repeats=DEFAULT_JUDGE_REPEATS, out_path=None):
    rows = []
    out_f = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for i, task in enumerate(tasks):
            t0 = time.time()
            traj = run_episode(env, task, agent_step_fn, max_steps=max_steps)
            judge_result = run_judge_with_repeats(
                judge_fn, traj["instruction"], traj["screenshots"], traj["final_answer"],
                n_repeats=judge_repeats,
            )
            row = {
                "idx": i,
                "instruction": traj["instruction"],
                "url": traj["url"],
                "n_steps": traj["n_steps"],
                "hit_max_steps": traj["hit_max_steps"],
                "final_answer": traj["final_answer"],
                "success": judge_result["success"],
                "judge_agreement": judge_result["agreement"],
                "judge_votes": judge_result["votes"],
                "elapsed_sec": round(time.time() - t0, 2),
            }
            rows.append(row)
            print(
                f"[{i + 1}/{len(tasks)}] {'O' if row['success'] else 'X'} "
                f"steps={row['n_steps']} agreement={row['judge_agreement']:.2f} "
                f"instr={row['instruction'][:50]!r}"
            )
            if out_f:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if out_f:
            out_f.close()

    n = len(rows)
    success_rate = sum(1 for r in rows if r["success"]) / n if n else 0.0
    print(f"\n성공률: {success_rate:.3f} ({sum(1 for r in rows if r['success'])}/{n})")
    return rows, success_rate


# ---------------------------------------------------------------------------
# mock 기반 단위 테스트 (실제 브라우저/모델/API 없이 제어 흐름만 검증)
# ---------------------------------------------------------------------------
def _run_mock_selftest():
    """
    `python eval_webvoyager.py --selftest`
    (1) run_episode의 종료 조건(즉시 terminate / max_steps 도달)
    (2) run_judge_with_repeats의 다수결 로직
    (3) _parse_success_verdict의 JSON/폴백 파싱
    (4) run_batch의 집계 + jsonl 저장
    를 mock으로 검증한다. 실제 브라우저/모델/API는 필요 없음.
    """
    from unittest.mock import MagicMock

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    fake_img = Image.new("RGB", (4, 4))

    # --- run_episode: 즉시 terminate ---
    fake_env = MagicMock()
    fake_env.reset.return_value = (fake_img, {"instruction": "do X", "url": "http://x"})
    traj = run_episode(fake_env, {"web": "http://x", "ques": "do X"}, dummy_agent_step, max_steps=5)
    check("즉시 terminate -> n_steps=1", traj["n_steps"] == 1)
    check("즉시 terminate -> hit_max_steps=False", traj["hit_max_steps"] is False)
    check("즉시 terminate -> execute_action 안 불림", not fake_env.execute_action.called)

    # --- run_episode: max_steps까지 계속 진행하는 agent ---
    def never_stop_agent(screenshot, task_info, history):
        return {"action": "wait", "time": 0.0}

    fake_env2 = MagicMock()
    fake_env2.reset.return_value = (fake_img, {"instruction": "do Y", "url": "http://y"})
    fake_env2.execute_action.return_value = (fake_img, None, False, False, {"instruction": "do Y", "url": "http://y"})
    traj2 = run_episode(fake_env2, {"web": "http://y", "ques": "do Y"}, never_stop_agent, max_steps=4)
    check("계속 진행 -> max_steps만큼 실행", traj2["n_steps"] == 4)
    check("계속 진행 -> hit_max_steps=True", traj2["hit_max_steps"] is True)
    check("계속 진행 -> screenshots 개수 = n_steps+1(초기 포함)", len(traj2["screenshots"]) == 5)

    # --- run_judge_with_repeats: 다수결 ---
    seq = iter([True, False, True])  # 2:1 -> True로 다수결
    judge_fn = lambda instruction, screenshots, final_answer: {"success": next(seq), "raw_response": "r"}
    result = run_judge_with_repeats(judge_fn, "do X", [fake_img], None, n_repeats=3)
    check("다수결 2:1 -> success=True", result["success"] is True)
    check("agreement = 2/3", abs(result["agreement"] - 2 / 3) < 1e-6)
    check("votes 개수=3", len(result["votes"]) == 3)

    # --- _parse_success_verdict ---
    ok, reason = _parse_success_verdict('{"reason": "did it", "success": true}')
    check("JSON success=true 파싱", ok is True and reason == "did it")
    ok2, _ = _parse_success_verdict('{"reason": "nope", "success": false}')
    check("JSON success=false 파싱", ok2 is False)
    ok3, _ = _parse_success_verdict("The task was completed with SUCCESS.")
    check("substring 폴백 SUCCESS", ok3 is True)
    ok4, _ = _parse_success_verdict("garbage response with no verdict")
    check("파싱 완전 실패 -> False로 안전 처리", ok4 is False)

    # --- run_batch: 집계 + jsonl 저장 ---
    import tempfile

    fake_env3 = MagicMock()
    fake_env3.reset.return_value = (fake_img, {"instruction": "do Z", "url": "http://z"})
    always_success_judge = lambda instruction, screenshots, final_answer: {"success": True, "raw_response": "ok"}
    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "out.jsonl")
        rows, rate = run_batch(
            [{"web": "http://z", "ques": "do Z"}] * 3,
            fake_env3, dummy_agent_step, always_success_judge,
            judge_repeats=1, out_path=out_path,
        )
        check("run_batch -> 3개 row", len(rows) == 3)
        check("run_batch -> success_rate=1.0", rate == 1.0)
        with open(out_path, encoding="utf-8") as f:
            saved = [json.loads(line) for line in f]
        check("run_batch -> jsonl 저장 개수 일치", len(saved) == 3)

    n_fail = sum(1 for _, ok in checks if not ok)
    for name, ok in checks:
        print(("[OK]  " if ok else "[FAIL]") + " " + name)
    print(f"\n{len(checks) - n_fail}/{len(checks)} passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="실제 브라우저/모델/API 없이 제어 흐름만 mock으로 검증")
    ap.add_argument("--tasks_jsonl", default=None, help="WebVoyager 태스크 jsonl 경로")
    ap.add_argument("--web_name", default=None)
    ap.add_argument("--judge", choices=["qwen", "openai"], default="qwen")
    ap.add_argument("--openai_model", default="gpt-4o")
    ap.add_argument("--adapter_dir", default=None, help="Qwen judge용 LoRA 어댑터 (선택)")
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--judge_repeats", type=int, default=DEFAULT_JUDGE_REPEATS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.selftest:
        _run_mock_selftest()
    else:
        if not args.tasks_jsonl:
            raise SystemExit("--tasks_jsonl 필요 (또는 --selftest로 로직만 검증)")
        tasks = load_webvoyager_tasks(args.tasks_jsonl, web_name=args.web_name)
        if args.limit:
            tasks = tasks[: args.limit]

        env = WebVoyagerEnv()

        if args.judge == "qwen":
            from qwen import QwenVLModel

            qwen_model = QwenVLModel(adapter_dir=args.adapter_dir)
            judge_fn = make_qwen_judge(qwen_model)
        else:
            judge_fn = make_openai_judge(model=args.openai_model)

        print(
            "[eval_webvoyager.py] 주의: agent_step_fn이 아직 dummy_agent_step()(항상 즉시 "
            "종료)임 - planner/agent_loop.py가 완성되면 이 자리에 실제 정책을 넣을 것."
        )
        run_batch(
            tasks, env, dummy_agent_step, judge_fn,
            max_steps=args.max_steps, judge_repeats=args.judge_repeats, out_path=args.out,
        )
        env.close()