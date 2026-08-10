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

# (2026-08-09 추가) vlm_agent(qwen.py/gui_grounding.py가 있는 폴더)를 sys.path에 넣는다 -
# planner.py/agent_loop.py와 동일한 패턴. 이 파일은 gui_grounding.ground()와 qwen.QwenVLModel을
# 함수 안에서 lazy import하는데(build_planner_grounding_agent_step/CLI), agent_loop보다 먼저
# import될 수도 있으니 여기서도 직접 부트스트랩해서 실행 cwd/순서에 의존하지 않게 한다.
import sys as _sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (os.path.join(_HERE, ".."), os.path.join(_HERE, "..", "vlm_agent")):
    _candidate = os.path.abspath(_candidate)
    if os.path.isfile(os.path.join(_candidate, "qwen.py")):
        if _candidate not in _sys.path:
            _sys.path.insert(0, _candidate)
        break

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
# (2026-08-09 추가) 실제 정책: planner LoRA(plan) + grounding LoRA(좌표) 연결
# ---------------------------------------------------------------------------
def _convert_planner_action_to_env(plan: dict, grounding_model, screenshot, ground_kwargs: dict):
    """
    agent/planner.py의 출력 스키마(자연어 target_description/action/text)를
    env_webvoyager.WebVoyagerEnv.execute_action()이 기대하는 스키마(픽셀 coordinate)로
    변환한다. click류는 gui_grounding.ground()를 호출해서 target_description을 실제
    좌표로 바꾼다(이 시점에 grounding_model은 이미 grounding LoRA가 활성 상태라고 가정 -
    agent_loop._AdapterSwitchView.generate()가 planning 호출 뒤 자동으로 default(grounding)로
    복원해주므로, plan_next_action()/plan_with_reflection() 호출 직후 여기로 넘어올 때는
    항상 그 상태다).

    (2026-08-11 추가 - 버그 수정) grounding 실패/drag 미구현으로 이 함수가 액션을 no-op(wait)
    으로 "다운그레이드"할 때는, 반환 dict에 내부용 마커 "_downgrade_reason"을 같이 실어 보낸다.
    호출부(build_planner_grounding_agent_step.agent_step_fn)는 이 함수를 부르기 전에 이미
    planner_history에 원래 액션(예: left_click)을 "정상적으로 실행됐다"는 전제로 append해둔
    상태라, 이 마커가 없으면 "클릭을 시도했지만 실제로는 화면에서 아무 일도 안 일어났다"는
    사실이 history에서 사라져 다음 스텝의 planner가 "내가 방금 그 액션을 실행했다"고 착각하게
    된다 - reflection이 반려한 액션을 history에서 완전히 지웠다가 같은 액션을 계속 재제안하던
    (Allrecipes CAPTCHA 10스텝 반복) 것과 완전히 같은 실패 패턴이 grounding 실패/drag 경로에서도
    똑같이 재현될 수 있는 코드 경로라, 그때 쓴 해법(_rejected/_rejection_reason 마커)을 여기도
    동일하게 적용한다. 이 마커는 env로 그대로 넘어가면 안 되므로 agent_step_fn이 planner_history를
    보정한 뒤 반드시 pop해서 지운다(env_webvoyager.execute_action()은 이 키를 모름 - 다만 몰라도
    무시되긴 하니 안 지워도 즉시 깨지진 않는다, 다만 불필요한 키를 env로 흘려보내지 않기 위해 지운다).
    """
    from gui_grounding import ground

    act = plan.get("action")

    if act == "terminate":
        # run_episode()는 final_answer를 action.get("text")에서 읽는데, planner.py의
        # terminate 스키마는 "answer" 필드를 쓴다 - 여기서 다리를 놓아준다.
        return {"action": "terminate", "status": plan.get("status", "failure"), "text": plan.get("answer")}

    if act in ("left_click", "double_click", "right_click"):
        target = plan.get("target_description") or ""
        g = ground(grounding_model, target, screenshot, **ground_kwargs)
        if g["result"] != "positive":
            print(f"[agent_step] grounding 실패(target={target!r}) -> 이번 스텝은 no-op으로 스킵")
            return {
                "action": "wait",
                "time": 0.5,
                "_downgrade_reason": f"grounding failed for target_description={target!r} (result={g.get('result')!r})",
            }
        w, h = screenshot.size
        x, y = g["point"][0] * w, g["point"][1] * h
        return {"action": act, "coordinate": [x, y]}

    if act == "drag":
        # env_webvoyager.py의 _drag()는 아직 NotImplementedError(스키마에 시작점이 없어서) -
        # planner LoRA는 drag를 낼 수 있지만 실행부가 못 받으니, 에피소드를 죽이는 대신
        # no-op으로 다운그레이드하고 로그만 남긴다. drag 실제 실행은 다음 작업 범위.
        print("[agent_step] drag 액션은 env_webvoyager.py에 아직 미구현 -> no-op으로 스킵")
        return {
            "action": "wait",
            "time": 0.0,
            "_downgrade_reason": "drag is not implemented in env_webvoyager.py yet",
        }

    if act == "type":
        return {"action": "type", "text": plan.get("text", "")}

    if act == "key":
        return {"action": "key", "text": plan.get("text", "")}

    if act == "scroll":
        return {"action": "scroll", "text": plan.get("text", "down")}

    if act == "wait":
        return {"action": "wait", "time": 1.0}

    # 알 수 없는 action(이론상 _parse_planner_action이 이미 걸러서 여기 안 와야 하지만,
    # 방어적으로) -> 안전하게 종료
    print(f"[agent_step] 알 수 없는 action={act!r} -> terminate/failure로 안전 종료")
    return {"action": "terminate", "status": "failure", "text": None}


_ANSWER_EXTRACTION_PROMPT_TEMPLATE = (
    'Task: "{instruction}"\n\n'
    "The attached image is the final screenshot after an agent finished this task. Answer the "
    "task's question as concisely as possible based only on what is visible in this screenshot. "
    "If the screenshot does not contain enough information to answer, reply with exactly: unknown."
)


def _extract_final_answer(grounding_model, instruction: str, screenshot, max_new_tokens: int = 100):
    """
    (2026-08-10 추가) planner LoRA가 terminate 시점에 "answer"를 채우지 못하는 문제(실측:
    질문형 태스크 3개 중 3개 전부 final_answer=null)의 원인을 찾아보니, 학습 데이터
    (prepare_planner_dataset.py의 terminate 변환)가 AgentNet에 없는 "최종 답변 텍스트"를
    아예 채운 적이 없어서였다 - AgentNet 자체가 이 정보를 갖고 있지 않아 재학습으로 고칠
    수 있는 문제가 아니다. 그래서 "행동 결정"과 "최종 답변 추출"을 분리해서, terminate인데
    answer가 비어 있을 때만 이 함수로 별도 QA 호출을 한 번 더 한다(WebVoyager 등 여러
    에이전트 시스템이 실제로 쓰는 분리 방식).

    grounding_model.model이 peft.PeftModel이면(어댑터가 얹혀 있으면) disable_adapter()로
    순수 base 상태에서 물어본다 - planner/grounding LoRA 둘 다 이 자유형 QA 포맷을 학습에서
    본 적이 없어서, 얹은 채로 물으면 오히려 이상한 포맷으로 답할 위험이 있다(이 세션 내내
    반복된 원칙: LoRA는 자기가 학습받은 입력 구조로만 물어야 함).
    """
    prompt = _ANSWER_EXTRACTION_PROMPT_TEMPLATE.format(instruction=instruction)
    messages = [
        {"role": "user", "content": [{"type": "image", "image": screenshot}, {"type": "text", "text": prompt}]}
    ]

    def _generate():
        return grounding_model.generate(messages, max_new_tokens=max_new_tokens, temperature=0.0).strip()

    if hasattr(grounding_model.model, "disable_adapter"):
        with grounding_model.model.disable_adapter():
            response = _generate()
    else:
        response = _generate()

    if not response or response.strip().lower() == "unknown":
        return None
    return response


def build_planner_grounding_agent_step(
    grounding_model,
    planning_view,
    use_reflection: bool = True,
    max_iterations: int = 2,
    planner_max_new_tokens: int = 300,
    ground_max_new_tokens: int = 128,
    ground_min_pixels: int | None = None,
    ground_max_pixels: int | None = None,
    verbose: bool = True,
):
    """
    agent_loop.load_shared_model()이 반환한 (model, planning_view)로 실제 정책 함수를
    만든다. dummy_agent_step 대신 run_batch()/run_episode()에 넘기면 된다.

        model, planning_view = load_shared_model(adapter_dir, planner_adapter_dir=...)
        agent_step_fn = build_planner_grounding_agent_step(model.model 아님 - model 자체, planning_view)
        run_batch(tasks, env, agent_step_fn, judge_fn, ...)

    각 스텝: planning_view로 plan_with_reflection()(또는 plan_next_action())을 돌려 planner
    스키마의 plan을 얻고 -> _convert_planner_action_to_env()로 env 스키마 액션으로 변환.
    planner의 history_actions는 run_episode()가 넘겨주는 history(env 스키마)와 별개로,
    이 함수 내부 클로저에서 planner 스키마 그대로 따로 누적한다(두 스키마가 달라서 그대로
    재사용할 수 없음 - _format_history()는 planner 스키마를 기대함).

    [2026-08-10 추가 - reflection은 항상 base 모델로]
    use_reflection=True(기본, --no_reflect로 끌 수 있는 선택 옵션)일 때, 실제로 돌려보니
    reflection 비평 호출을 planning_view(=planner LoRA 어댑터)로 그대로 흘려보내면 헛소리가
    나왔다 - reflection이 요구하는 출력 포맷({"observation","approved","critique"})을 planner
    LoRA는 학습에서 한 번도 못 봤고(액션 JSON 포맷만 SFT됨), 그 어댑터를 켠 채로 다른 포맷을
    강제해봐야 자기가 아는 포맷으로 답할 뿐이다. 그래서 reflection 호출만 별도로
    agent_loop._BaseModelView(disable_adapter)로 감싸서 base 모델로 돌린다 - planner.py의
    plan_with_reflection(reflection_model=...) 파라미터가 정확히 이 용도로 만들어져 있었는데
    이 함수가 그동안 안 넘기고 있었다(그래서 propose/reflect 둘 다 조용히 같은 planner
    어댑터로 돌고 있었음).
    """
    from planner import plan_next_action, plan_with_reflection

    reflection_view = None
    if use_reflection:
        from agent_loop import _BaseModelView

        if hasattr(grounding_model.model, "disable_adapter"):
            reflection_view = _BaseModelView(grounding_model)
        else:
            # grounding_model 자체가 애초에 어댑터 없이 로드된 경우(이 eval 경로에서는 거의
            # 안 일어나지만) - 이미 base이니 그대로 씀.
            reflection_view = grounding_model

    planner_history: list = []

    ground_kwargs = {"max_new_tokens": ground_max_new_tokens}
    if ground_min_pixels is not None:
        ground_kwargs["min_pixels"] = ground_min_pixels
    if ground_max_pixels is not None:
        ground_kwargs["max_pixels"] = ground_max_pixels

    def agent_step_fn(screenshot, task_info, history):
        instruction = task_info["instruction"]
        if use_reflection:
            plan = plan_with_reflection(
                planning_view, instruction, screenshot,
                history_actions=planner_history,
                max_new_tokens=planner_max_new_tokens,
                max_iterations=max_iterations,
                reflection_model=reflection_view,
            )
        else:
            plan = plan_next_action(
                planning_view, instruction, screenshot,
                history_actions=planner_history,
                max_new_tokens=planner_max_new_tokens,
            )

        if verbose:
            print(
                f"[agent_step] action={plan.get('action')!r} "
                f"target={plan.get('target_description')!r} text={plan.get('text')!r} "
                f"status={plan.get('status')!r} answer={plan.get('answer')!r} "
                f"parse_failed={plan.get('_parse_failed')!r}"
            )
            # (2026-08-10 추가) plan_with_reflection()이 실제로 반려를 하고 있는지 눈으로 보려고
            # _reflection_approved/_reflection_log를 찍는다 - 실측으로 같은 target을 여러 번
            # 반복 클릭하는 게 관찰됐는데, reflection이 그걸 걸러주고 있는지(반려했는데 planner가
            # 계속 비슷하게 재제안하는 것인지) reflection 자체가 그냥 다 승인해버리는지 이 로그
            # 없이는 구분이 안 됐다.
            if "_reflection_approved" in plan:
                print(f"[agent_step]   reflection_approved={plan.get('_reflection_approved')!r}")
                for entry in plan.get("_reflection_log", []):
                    verdict = entry.get("verdict", {})
                    print(
                        f"[agent_step]   iter={entry.get('iteration')} "
                        f"approved={verdict.get('approved')!r} "
                        f"possible_failure_reason={verdict.get('possible_failure_reason')!r} "
                        f"critique={verdict.get('critique')!r}"
                    )

        # (2026-08-10 추가) 실측으로 확인된 진짜 버그: plan_with_reflection()이 max_iterations를
        # 다 써도 승인 안 나면 마지막(반려당한) 후보를 `_reflection_approved: False`로 표시만 하고
        # 그대로 반환한다(planner.py 설계상 "실행할지 말지는 호출부가 판단"하도록 일부러 강제
        # 차단하지 않음) - 근데 이 함수가 그 플래그를 여태 안 보고 무조건 실행해버리고 있었다.
        # 그 결과 reflection이 반려한("Lakers 뉴스랑 무관한 화면인데 terminate하려 함" 등) 액션이
        # 그대로 실행되면서 점점 더 엉뚱한 페이지로 새어나가는 게 실측으로 확인됨. 여기서 명시적으로
        # 막는다 - 반려된 채로 끝난 액션은 실행하지 않고 안전하게 한 스텝 쉬며(no-op) 다음 스텝에서
        # 갱신된 history를 보고 다시 판단하게 한다.
        #
        # [2026-08-10 추가 수정, x2] 처음엔 반려된 plan을 history에서 아예 뺐다("이미 했다"처럼
        # 보이면 안 되니까) - 근데 그러면 다음 스텝(새 agent_step_fn 호출)에서 모델이 방금
        # 반려당했다는 사실 자체를 완전히 잊어버려서, 똑같은 화면을 보고 똑같은 액션을 계속
        # 재제안하는 문제가 실측으로 확인됨(Allrecipes CAPTCHA 화면에서 "CAPTCHA 체크박스 클릭"을
        # 10스텝 내내 반복 제안). reflection의 critique가 스텝 경계를 못 넘어가고 사라지는 게
        # 원인이라, planner.py의 _format_history()에 반려 표시(`_rejected`/`_rejection_reason`)를
        # 추가해서 "시도했지만 반려당했다(+반려 이유)"를 history에 명확히 남기는 쪽으로 바꿨다 -
        # "이미 했다"도 아니고 "기록에 아예 없다"도 아닌 절충안.
        if plan.get("_reflection_approved") is False:
            log = plan.get("_reflection_log") or []
            last_critique = log[-1].get("verdict", {}).get("critique") if log else None
            planner_history.append({
                "action": plan.get("action"),
                "target_description": plan.get("target_description"),
                "text": plan.get("text"),
                "_rejected": True,
                "_rejection_reason": last_critique,
            })
            print(
                f"[agent_step] reflection이 끝까지 승인 안 함(max_iterations={max_iterations} 소진) -> "
                f"실행하지 않고 스킵(history엔 '반려됨'으로 기록). 반려된 후보: action={plan.get('action')!r} "
                f"target={plan.get('target_description')!r}"
            )
            return {"action": "wait", "time": 0.5}

        planner_history.append(plan)

        # (2026-08-10 추가) terminate인데 answer가 비어 있으면 별도 QA 호출로 채운다.
        # _extract_final_answer() docstring 참고 - planner LoRA가 애초에 이 필드를 학습에서
        # 못 본 문제라 재시도/reflection으로는 안 고쳐짐. reflection이 반려한 terminate는 위에서
        # 이미 걸러졌으니, 여기 도달하는 terminate는 (reflection이 껐거나) 승인된 것만 남는다.
        if plan.get("action") == "terminate" and not plan.get("answer"):
            try:
                extracted = _extract_final_answer(grounding_model, instruction, screenshot)
            except Exception as e:  # noqa: BLE001 - 최종 답변 추출 실패로 에피소드 전체를 죽이지 않음
                print(f"[agent_step] answer 추출 실패(무시하고 진행): {e}")
                extracted = None
            if extracted:
                plan = dict(plan)
                plan["answer"] = extracted

        env_action = _convert_planner_action_to_env(plan, grounding_model, screenshot, ground_kwargs)

        # (2026-08-11 추가 - 버그 수정) grounding 실패/drag 미구현으로 위에서 no-op(wait)으로
        # 다운그레이드된 경우, 바로 위 planner_history.append(plan)이 "실제로는 실행되지 않은"
        # 액션(예: left_click)을 마치 실행된 것처럼 남겨버린 상태다 - reflection 최종 반려 케이스와
        # 똑같은 문제라 똑같은 해법(_rejected/_rejection_reason)으로 그 마지막 항목을 덮어써서
        # 바로잡는다. _downgrade_reason은 env로 넘길 필요 없는 내부 마커이므로 pop해서 제거한다.
        downgrade_reason = env_action.pop("_downgrade_reason", None)
        if downgrade_reason is not None:
            planner_history[-1] = {
                "action": plan.get("action"),
                "target_description": plan.get("target_description"),
                "text": plan.get("text"),
                "_rejected": True,
                "_rejection_reason": downgrade_reason,
            }
            print(
                f"[agent_step] 액션이 실행되지 못하고 no-op으로 다운그레이드됨 -> history에 "
                f"'반려됨'으로 기록. reason={downgrade_reason}"
            )

        return env_action

    return agent_step_fn


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

    # --- _convert_planner_action_to_env / build_planner_grounding_agent_step ---
    import sys
    import types

    fake_ground_calls = []

    def _fake_ground(model, instruction, screenshot, **kwargs):
        fake_ground_calls.append((instruction, kwargs))
        if instruction == "fail me":
            return {"result": "wrong_format", "point": None, "raw_response": "??"}
        return {"result": "positive", "point": [0.25, 0.75], "raw_response": "(250,750)"}

    fake_gui_grounding_module = types.ModuleType("gui_grounding")
    fake_gui_grounding_module.ground = _fake_ground
    sys.modules["gui_grounding"] = fake_gui_grounding_module
    try:
        wide_img = Image.new("RGB", (200, 100))

        # terminate: status/answer -> text 다리 놓기
        env_act = _convert_planner_action_to_env(
            {"action": "terminate", "status": "success", "answer": "42"}, None, wide_img, {}
        )
        check("terminate -> text에 answer가 들어감", env_act == {"action": "terminate", "status": "success", "text": "42"})

        # left_click: grounding 성공 -> 정규화 좌표*screenshot 크기로 변환
        env_act2 = _convert_planner_action_to_env(
            {"action": "left_click", "target_description": "the X button"}, None, wide_img, {}
        )
        check(
            "left_click -> point[0,1]*[w,h]로 픽셀 좌표 변환",
            env_act2 == {"action": "left_click", "coordinate": [50.0, 75.0]},
        )

        # left_click: grounding 실패 -> no-op(wait)으로 다운그레이드
        env_act3 = _convert_planner_action_to_env(
            {"action": "left_click", "target_description": "fail me"}, None, wide_img, {}
        )
        check("grounding 실패 -> wait no-op", env_act3["action"] == "wait")

        # drag -> 아직 env 미구현이라 wait no-op으로 다운그레이드
        env_act4 = _convert_planner_action_to_env(
            {"action": "drag", "target_description": "a", "text": "b"}, None, wide_img, {}
        )
        check("drag -> env 미구현이라 wait no-op", env_act4["action"] == "wait")

        # type/key/scroll/wait 패스스루
        check(
            "type 패스스루",
            _convert_planner_action_to_env({"action": "type", "text": "hi"}, None, wide_img, {})
            == {"action": "type", "text": "hi"},
        )
        check(
            "key 패스스루",
            _convert_planner_action_to_env({"action": "key", "text": "Enter"}, None, wide_img, {})
            == {"action": "key", "text": "Enter"},
        )
        check(
            "scroll 패스스루(direction=text)",
            _convert_planner_action_to_env({"action": "scroll", "text": "up"}, None, wide_img, {})
            == {"action": "scroll", "text": "up"},
        )
        check(
            "wait -> 기본 1.0초",
            _convert_planner_action_to_env({"action": "wait"}, None, wide_img, {}) == {"action": "wait", "time": 1.0},
        )

        # 알 수 없는 action -> 안전하게 terminate/failure
        unknown = _convert_planner_action_to_env({"action": "fly"}, None, wide_img, {})
        check("알 수 없는 action -> terminate/failure", unknown["action"] == "terminate" and unknown["status"] == "failure")

        # --- _extract_final_answer: terminate에서 answer가 비었을 때 별도 QA로 채우는 로직 ---
        def _make_fake_grounding_model(response_text, has_adapter=True):
            m = MagicMock()
            m.generate.return_value = response_text
            if not has_adapter:
                del m.model.disable_adapter  # hasattr()이 False가 되도록
            return m

        fake_gm = _make_fake_grounding_model("  The cheapest flight is $210.  ")
        answer = _extract_final_answer(fake_gm, "find the cheapest flight", wide_img)
        check("_extract_final_answer -> 응답 텍스트를 strip해서 반환", answer == "The cheapest flight is $210.")
        check(
            "_extract_final_answer -> disable_adapter() 컨텍스트 안에서 generate 호출됨(어댑터 있으면)",
            fake_gm.model.disable_adapter.called and fake_gm.model.disable_adapter.return_value.__enter__.called,
        )

        fake_gm_unknown = _make_fake_grounding_model("unknown")
        check(
            "_extract_final_answer -> 'unknown' 응답이면 None 반환",
            _extract_final_answer(fake_gm_unknown, "q", wide_img) is None,
        )

        fake_gm_no_adapter = _make_fake_grounding_model("some answer", has_adapter=False)
        check(
            "_extract_final_answer -> 어댑터 없는 모델도 그냥 generate() 직접 호출로 동작",
            _extract_final_answer(fake_gm_no_adapter, "q", wide_img) == "some answer",
        )

        # --- build_planner_grounding_agent_step: planner/gui_grounding을 fake로 갈아끼우고 연동 확인 ---
        plan_calls = []

        def _fake_plan_with_reflection(planning_view, instruction, screenshot, history_actions=None, **kw):
            plan_calls.append(
                {"instruction": instruction, "history_len": len(history_actions or []), "kwargs": kw}
            )
            return {"reasoning": "r", "action": "left_click", "target_description": "the X button"}

        def _fake_plan_next_action(planning_view, instruction, screenshot, history_actions=None, **kw):
            plan_calls.append({"instruction": instruction, "history_len": len(history_actions or [])})
            return {"reasoning": "r", "action": "wait"}

        fake_planner_module = types.ModuleType("planner")
        fake_planner_module.plan_with_reflection = _fake_plan_with_reflection
        fake_planner_module.plan_next_action = _fake_plan_next_action
        sys.modules["planner"] = fake_planner_module
        try:
            fake_model = MagicMock()
            fake_planning_view = MagicMock()

            agent_step_fn = build_planner_grounding_agent_step(
                fake_model, fake_planning_view, use_reflection=True, verbose=False
            )
            step1 = agent_step_fn(wide_img, {"instruction": "close the window"}, {"actions": [], "screenshots": []})
            check(
                "agent_step_fn(reflection) -> planner plan을 env 액션으로 변환",
                step1 == {"action": "left_click", "coordinate": [50.0, 75.0]},
            )
            step2 = agent_step_fn(wide_img, {"instruction": "close the window"}, {"actions": [], "screenshots": []})
            check("agent_step_fn -> 두 번째 호출에서 history_actions에 이전 plan이 누적됨", plan_calls[1]["history_len"] == 1)

            # (2026-08-10 추가) reflection_model이 plan_with_reflection에 넘어가는지, 그리고 그게
            # planning_view(planner 어댑터)가 아니라 grounding_model을 감싼 base view인지 확인 -
            # reflection이 planner LoRA 포맷 헛소리를 냈던 버그의 재발 방지 검증.
            import agent_loop as _agent_loop_module

            reflection_model_passed = plan_calls[0]["kwargs"].get("reflection_model")
            check(
                "reflection_model이 agent_loop._BaseModelView(base)로 전달됨(planner 어댑터 아님)",
                isinstance(reflection_model_passed, _agent_loop_module._BaseModelView)
                and reflection_model_passed._qwen_model is fake_model
                and reflection_model_passed is not fake_planning_view,
            )

            agent_step_fn_no_reflect = build_planner_grounding_agent_step(
                fake_model, fake_planning_view, use_reflection=False, verbose=False
            )
            plan_calls.clear()
            agent_step_fn_no_reflect(wide_img, {"instruction": "x"}, {"actions": [], "screenshots": []})
            check("--no_reflect -> plan_next_action(reflection 없는 쪽) 사용", len(plan_calls) == 1)

            # terminate인데 answer가 없으면 agent_step_fn이 answer 추출을 자동으로 태우는지
            def _fake_plan_terminate_no_answer(planning_view, instruction, screenshot, history_actions=None, **kw):
                return {"reasoning": "r", "action": "terminate", "status": "success"}

            fake_planner_module.plan_with_reflection = _fake_plan_terminate_no_answer
            fake_model_for_answer = _make_fake_grounding_model("42")
            agent_step_fn2 = build_planner_grounding_agent_step(
                fake_model_for_answer, fake_planning_view, use_reflection=True, verbose=False
            )
            term_action = agent_step_fn2(wide_img, {"instruction": "what is the answer?"}, {"actions": [], "screenshots": []})
            check(
                "terminate + answer 없음 -> _extract_final_answer로 채워서 env action의 text에 들어감",
                term_action == {"action": "terminate", "status": "success", "text": "42"},
            )

            # terminate인데 answer가 이미 있으면 추출 호출 자체를 안 해야 함(중복 호출 낭비 방지)
            def _fake_plan_terminate_with_answer(planning_view, instruction, screenshot, history_actions=None, **kw):
                return {"reasoning": "r", "action": "terminate", "status": "success", "answer": "already have it"}

            fake_planner_module.plan_with_reflection = _fake_plan_terminate_with_answer
            fake_model_should_not_be_called = _make_fake_grounding_model("SHOULD NOT APPEAR")
            agent_step_fn3 = build_planner_grounding_agent_step(
                fake_model_should_not_be_called, fake_planning_view, use_reflection=True, verbose=False
            )
            term_action2 = agent_step_fn3(wide_img, {"instruction": "x"}, {"actions": [], "screenshots": []})
            check(
                "terminate + answer 이미 있음 -> 추출 재호출 안 하고 기존 answer 그대로 사용",
                term_action2 == {"action": "terminate", "status": "success", "text": "already have it"}
                and not fake_model_should_not_be_called.generate.called,
            )

            # (2026-08-10 추가) reflection이 끝까지 반려(_reflection_approved=False)한 액션은
            # 실행하지 않고 wait no-op으로 스킵해야 함 - 실측으로 "반려된 액션이 그대로 실행돼서
            # 엉뚱한 페이지로 새는" 버그가 나왔던 부분의 회귀 방지 테스트.
            rejected_plan_calls = []

            def _fake_plan_rejected(planning_view, instruction, screenshot, history_actions=None, **kw):
                rejected_plan_calls.append(
                    {"history_len": len(history_actions or []), "history": list(history_actions or [])}
                )
                return {
                    "reasoning": "r",
                    "action": "left_click",
                    "target_description": "the Sort dropdown",
                    "_reflection_approved": False,
                    "_reflection_log": [{"iteration": 1, "verdict": {"approved": False, "critique": "off-task"}}],
                }

            fake_planner_module.plan_with_reflection = _fake_plan_rejected
            fake_model_should_not_ground = _make_fake_grounding_model("SHOULD NOT APPEAR")
            agent_step_fn4 = build_planner_grounding_agent_step(
                fake_model_should_not_ground, fake_planning_view, use_reflection=True, verbose=False
            )
            rejected_action = agent_step_fn4(wide_img, {"instruction": "x"}, {"actions": [], "screenshots": []})
            check(
                "reflection 최종 반려 -> wait no-op으로 스킵(실행 안 함)",
                rejected_action == {"action": "wait", "time": 0.5},
            )
            check(
                "reflection 최종 반려 -> grounding(_convert_planner_action_to_env) 자체를 안 탐",
                not fake_model_should_not_ground.generate.called,
            )
            # (2026-08-10 추가, x2) 반려된 plan은 "_rejected: True" 마커와 함께 planner_history에
            # 기록되어야 함(완전히 빼버리면 다음 스텝에서 모델이 방금 반려당한 걸 까먹고 똑같은
            # 걸 또 제안하는 문제가 실측으로 나왔음 - planner.py._format_history()가 이 마커를
            # 보고 "시도했지만 반려됨"으로 명확히 구분해서 보여줌).
            agent_step_fn4(wide_img, {"instruction": "x"}, {"actions": [], "screenshots": []})
            check(
                "reflection 최종 반려 -> planner_history에 '반려됨' 마커와 함께 기록됨(완전히 빠지지 않음)",
                rejected_plan_calls[1]["history_len"] == 1,
            )

            # approved=True(정상 승인)면 여전히 정상 실행돼야 함(과잉 차단 아닌지 확인)
            def _fake_plan_approved(planning_view, instruction, screenshot, history_actions=None, **kw):
                return {
                    "reasoning": "r",
                    "action": "left_click",
                    "target_description": "the X button",
                    "_reflection_approved": True,
                    "_reflection_log": [],
                }

            fake_planner_module.plan_with_reflection = _fake_plan_approved
            agent_step_fn5 = build_planner_grounding_agent_step(
                fake_model, fake_planning_view, use_reflection=True, verbose=False
            )
            approved_action = agent_step_fn5(wide_img, {"instruction": "x"}, {"actions": [], "screenshots": []})
            check(
                "reflection 승인(True) -> 정상적으로 grounding까지 이어져서 실행됨",
                approved_action == {"action": "left_click", "coordinate": [50.0, 75.0]},
            )

            # (2026-08-11 추가 - 회귀 테스트) 버그: grounding 실패로 액션이 no-op(wait)으로
            # 다운그레이드될 때, planner_history에는 이미 "left_click을 정상 실행했다"는 전제로
            # plan이 append돼 있어서 다음 스텝의 planner가 실제로는 안 일어난 일을 "일어난 일"로
            # 착각했었다(reflection 최종 반려 케이스와 동일한 실패 패턴). 이제는 grounding 실패시
            # 해당 history 항목이 _rejected 마커로 정정되어야 한다.
            def _fake_plan_ground_fail(planning_view, instruction, screenshot, history_actions=None, **kw):
                plan_calls.append(
                    {"history_len": len(history_actions or []), "history": list(history_actions or [])}
                )
                return {"reasoning": "r", "action": "left_click", "target_description": "fail me"}

            fake_planner_module.plan_with_reflection = _fake_plan_ground_fail
            plan_calls.clear()
            agent_step_fn6 = build_planner_grounding_agent_step(
                fake_model, fake_planning_view, use_reflection=True, verbose=False
            )
            ground_fail_action = agent_step_fn6(wide_img, {"instruction": "x"}, {"actions": [], "screenshots": []})
            check(
                "grounding 실패 다운그레이드 -> env로 나가는 액션엔 내부 마커(_downgrade_reason)가 안 남음",
                "_downgrade_reason" not in ground_fail_action,
            )
            check(
                "grounding 실패 다운그레이드 -> env 액션 자체는 여전히 wait no-op",
                ground_fail_action["action"] == "wait",
            )
            agent_step_fn6(wide_img, {"instruction": "x"}, {"actions": [], "screenshots": []})
            check(
                "grounding 실패 다운그레이드 -> 다음 스텝 history에 원래 액션이 '반려됨'으로 정정되어 남음"
                "(실행된 것처럼 남지 않음)",
                plan_calls[1]["history"][-1].get("_rejected") is True
                and plan_calls[1]["history"][-1].get("action") == "left_click"
                and plan_calls[1]["history"][-1].get("target_description") == "fail me"
                and "grounding failed" in (plan_calls[1]["history"][-1].get("_rejection_reason") or ""),
            )
        finally:
            del sys.modules["planner"]
    finally:
        del sys.modules["gui_grounding"]

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
    ap.add_argument("--adapter_dir", default=None,
                     help="Qwen judge용 LoRA 어댑터 (선택). --reuse_agent_model_for_judge를 켜면 무시됨.")
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--judge_repeats", type=int, default=DEFAULT_JUDGE_REPEATS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    # (2026-08-09 추가) 실제 planner+grounding 정책. --agent_grounding_adapter_dir를 안 주면
    # 기존처럼 dummy_agent_step(즉시 실패)로 동작 - 파이프라인 배선만 확인하고 싶을 때는 그대로 둘 것.
    ap.add_argument("--agent_grounding_adapter_dir", default=None,
                     help="grounding LoRA 체크포인트(예: checkpoints/qwen2.5vl-3b-gui-lora-stage2/"
                          "checkpoint-4130). 지정해야 dummy_agent_step 대신 실제 정책(planner+grounding)이 돈다.")
    ap.add_argument("--agent_planner_adapter_dir", default=None,
                     help="planner LoRA 체크포인트(예: checkpoints/qwen2.5vl-3b-planner-lora). 지정 안 하면 "
                          "planning은 base 모델(disable_adapter)로 돈다 - agent_loop.py의 load_shared_model 참고.")
    ap.add_argument("--no_reflect", dest="use_reflection", action="store_false", default=True,
                     help="plan_with_reflection 대신 plan_next_action만 사용(비평 루프 생략, 스텝당 더 빠름)")
    ap.add_argument("--max_iterations", type=int, default=2, help="--no_reflect가 아닐 때 reflection 최대 재시도")
    ap.add_argument("--min_pixels", type=int, default=None)
    ap.add_argument("--max_pixels", type=int, default=None)
    ap.add_argument(
        "--reuse_agent_model_for_judge", action="store_true",
        help="(2026-08-09 추가) --judge qwen일 때, 별도 모델을 또 로드하지 않고 이미 로드된 agent "
             "모델을 disable_adapter() 상태(base)로 재사용해서 judge로 쓴다. RTX 5070 Ti 16GB에서 "
             "agent 모델(2개 LoRA 동시 로드) + judge 모델(별도 인스턴스)을 같이 올리면 VRAM이 빠듯하니, "
             "--agent_grounding_adapter_dir를 쓸 때는 기본적으로 이 옵션을 켜는 걸 권장.",
    )
    args = ap.parse_args()

    if args.selftest:
        _run_mock_selftest()
    else:
        if not args.tasks_jsonl:
            raise SystemExit("--tasks_jsonl 필요 (또는 --selftest로 로직만 검증)")
        tasks = load_webvoyager_tasks(args.tasks_jsonl, web_name=args.web_name)
        if args.web_name and not tasks:
            # (2026-08-10 추가) --web_name이 jsonl 안의 실제 web_name과 안 맞으면 load_webvoyager_tasks가
            # 그냥 빈 리스트를 조용히 돌려주고, 그 뒤로도 아무 검증 없이 run_batch까지 흘러가서
            # "성공률: 0.000 (0/0)"만 찍히고 끝나버린다(실측 - 원인 파악하기 어려운 침묵 실패였음).
            # 여기서 바로 걸러서 실제 존재하는 web_name 목록을 보여준다.
            available = sorted({rec.get("web_name") for rec in load_webvoyager_tasks(args.tasks_jsonl)})
            raise SystemExit(
                f"--web_name={args.web_name!r}에 해당하는 태스크가 없음. {args.tasks_jsonl} 안의 실제 "
                f"web_name 목록: {available}"
            )
        if args.limit:
            tasks = tasks[: args.limit]
        if not tasks:
            raise SystemExit(f"{args.tasks_jsonl}에서 태스크를 하나도 못 찾음 (파일이 비었거나 경로 확인 필요)")

        env = WebVoyagerEnv()

        agent_model = None
        if args.agent_grounding_adapter_dir:
            from agent_loop import load_shared_model

            model_kwargs = {}
            if args.min_pixels is not None:
                model_kwargs["min_pixels"] = args.min_pixels
            if args.max_pixels is not None:
                model_kwargs["max_pixels"] = args.max_pixels

            agent_model, planning_view = load_shared_model(
                args.agent_grounding_adapter_dir,
                planner_adapter_dir=args.agent_planner_adapter_dir,
                **model_kwargs,
            )
            agent_step_fn = build_planner_grounding_agent_step(
                agent_model, planning_view,
                use_reflection=args.use_reflection, max_iterations=args.max_iterations,
                ground_min_pixels=args.min_pixels, ground_max_pixels=args.max_pixels,
            )
        else:
            print(
                "[eval_webvoyager.py] 주의: --agent_grounding_adapter_dir 미지정 -> agent_step_fn이 "
                "dummy_agent_step()(항상 즉시 종료)임. 실제 정책을 돌리려면 --agent_grounding_adapter_dir"
                "(+ 선택적으로 --agent_planner_adapter_dir)를 지정할 것."
            )
            agent_step_fn = dummy_agent_step

        if args.judge == "qwen":
            if args.reuse_agent_model_for_judge:
                if agent_model is None:
                    raise SystemExit(
                        "--reuse_agent_model_for_judge는 --agent_grounding_adapter_dir로 agent 모델이 "
                        "이미 로드돼 있어야 함"
                    )
                from agent_loop import _BaseModelView

                judge_model = _BaseModelView(agent_model)
            elif agent_model is not None and args.adapter_dir is None:
                # agent 모델이 이미 로드돼 있는데 judge용으로 또 하나 로드하는 건 VRAM 낭비 -
                # --adapter_dir를 따로 지정하지 않았으면(=judge에 특정 LoRA가 필요한 게 아니면)
                # 그냥 agent 모델을 base로 재사용하도록 자동으로 권장 경로를 태움.
                print(
                    "[eval_webvoyager.py] agent 모델이 이미 로드돼 있어서 judge용 별도 모델 로드를 "
                    "건너뛰고 agent 모델을 base로 재사용함(VRAM 절약). 원치 않으면 --adapter_dir를 "
                    "명시하거나 코드에서 이 자동 재사용 분기를 끌 것."
                )
                from agent_loop import _BaseModelView

                judge_model = _BaseModelView(agent_model)
            else:
                from qwen import QwenVLModel

                judge_model = QwenVLModel(adapter_dir=args.adapter_dir)
            judge_fn = make_qwen_judge(judge_model)
        else:
            judge_fn = make_openai_judge(model=args.openai_model)

        run_batch(
            tasks, env, agent_step_fn, judge_fn,
            max_steps=args.max_steps, judge_repeats=args.judge_repeats, out_path=args.out,
        )
        env.close()