"""
gui_actor_eval_webvoyager.py

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
(`python gui_actor_eval_webvoyager.py --selftest`).

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

[--resume - 2026-08-11 추가]
API 비용(GPT-4o planner/judge) 때문에 중간에 죽으면(개인 PC 장시간 무중단 실행 리스크 -
드라이버 행/절전/네트워크 끊김 등) 이미 끝낸 태스크까지 다시 도는 걸 막아야 해서 추가.
--out 파일에 매 태스크 결과가 즉시 flush되므로(run_batch 참고) 죽어도 그때까지 결과는
안 날아간다 - --resume은 그 파일을 읽어서 이미 있는 task_id는 건너뛰고 이어서 append하는
것만 담당. 태스크 식별은 WebVoyager jsonl의 "id" 필드(예: "Amazon--16") 기준(_task_key()
참고) - 없으면 url+instruction 조합으로 폴백.
"""

import argparse
import base64
import io
import json
import os
import re
import sys as _sys
import time
from collections import Counter

from PIL import Image

# (v3 이동 - agent-actor3b/ 폴더로 옮기면서 위치 변경) 이 파일은 이제 vlm_agent/agent/가 아니라
# vlm_agent/agent-actor3b/에 있다. env_webvoyager.py/planner.py/agent_loop.py/api_planner.py는
# 여전히 vlm_agent/agent/에 그대로 있고(옮기지 않음 - --grounding_backend lora 경로가 그대로
# 참조), qwen.py/gui_grounding.py/region_focus.py/coord_utils.py는 vlm_agent/ 루트에,
# gui_actor_grounding.py/gui_actor_region_focus.py는 이 파일과 같은 폴더에 있다.
#
# 아래에서 vlm_agent/(qwen.py 등)와 vlm_agent/agent/(env_webvoyager.py 등)를 sys.path에
# 명시적으로 추가하는 걸 바로 다음 줄의 `from env_webvoyager import ...`(top-level import라
# 부트스트랩이 그 전에 실행돼야 함)보다 먼저 해야 한다 - 원래 이 파일이 vlm_agent/agent/ 안에
# 있었을 때는 파이썬이 스크립트 자기 자신의 디렉토리를 자동으로 sys.path에 넣어줘서 이 import가
# 별도 부트스트랩 없이도 됐는데, 폴더를 옮기면서 그 암묵적 동작을 잃었다.
_HERE = os.path.dirname(os.path.abspath(__file__))
_VLM_AGENT_DIR = os.path.abspath(os.path.join(_HERE, ".."))   # vlm_agent/ (qwen.py 등)
_LEGACY_AGENT_DIR = os.path.join(_VLM_AGENT_DIR, "agent")     # vlm_agent/agent/ (env_webvoyager.py 등)
for _candidate in (_VLM_AGENT_DIR, _LEGACY_AGENT_DIR):
    if os.path.isdir(_candidate) and _candidate not in _sys.path:
        _sys.path.insert(0, _candidate)

from env_webvoyager import WebVoyagerEnv, load_webvoyager_tasks

MAX_JUDGE_SCREENSHOTS = 15
# (2026-08-14 수정) 15 -> 17. 실측(Booking--0): destination/날짜/검색까지 정상 진행하고도
# 필터/정렬 등 마무리 조작 몇 스텝이 모자라서 15스텝 한도에 걸려 중간에 끊기는 게 확인됨
# (로케일 재적용 레이스 컨디션/새 탭 처리 버그 수정 이후로는 스텝이 헛돌지 않고 실제
# 진행에 쓰이므로, 조금 늘려도 낭비가 아니라 실제로 필요한 여유일 가능성이 높음).
DEFAULT_MAX_STEPS = 17
DEFAULT_JUDGE_REPEATS = 3


# (2026-08-11 추가) 같은 액션이 이만큼 반복되면 "막혔다"고 보고 조기 종료한다.
# CAPTCHA에 걸려서 planner가 같은 걸 계속 재시도하는 경우(Allrecipes 사례, planner.py
# docstring 참고)의 일반화된 안전장치 - CAPTCHA뿐 아니라 grounding이 계속 같은 지점을
# 잘못 찍는 등 "어떤 이유로든 진행이 안 되는" 상황을 폭넓게 잡는다.
DEFAULT_STUCK_REPEAT_THRESHOLD = 5

# (2026-08-11 수정 - 뺑뺑이/오실레이션 탐지) 원래는 "바로 직전과 연속으로 같은가"만 봤는데,
# 이러면 검색 버튼을 못 찾고 두세 개 다른 액션을 번갈아 시도하는(A -> B -> A -> B) 패턴은
# 못 잡는다(매 스텝 직전 액션과 다르므로 연속 카운트가 계속 리셋됨) - planner.py의
# REPEAT_WARNING_WINDOW와 동일한 문제/동일한 해법. "연속" 대신 "최근
# STUCK_REPEAT_WINDOW개 안에서의 빈도"로 조기종료를 판단하도록 바꿨다. 윈도우 크기는
# 임계값의 2배 - 2-사이클(A/B가 절반씩 번갈아 나오는 최악의 경우)도 윈도우가 다 차면
# 반드시 걸리도록 하기 위함.
DEFAULT_STUCK_REPEAT_WINDOW = DEFAULT_STUCK_REPEAT_THRESHOLD * 2


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
# (2026-08-11 추가 - region focus 재연결) click grounding 백엔드 선택
# ---------------------------------------------------------------------------
def _build_click_ground_fn(
    grounding_backend: str = "lora",
    use_regionfocus: bool = False,
    regionfocus_debug_image: bool = False,
    regionfocus_debug_text: bool = False,
    regionfocus_step1_format: str = "point_text",
    regionfocus_step4_format: str = "point_text",
):
    """
    click 계열 액션(left_click/double_click/right_click)의 grounding을 무엇으로 할지 정하는
    factory. eval_webvoyager.py(Track A, v2)는 원래 _convert_planner_action_to_env()가
    gui_grounding.ground()(초기 grounding 1회)만 직접 import해서 썼는데, 이 프로젝트가 실제로
    검증/적용한 RegionFocus 재탐색 파이프라인(region_focus.py의 ground_with_regionfocus() -
    초기 grounding -> judge 판단 -> 오답이면 재탐색 -> crop/zoom 4비율 정밀화 -> 후보 종합,
    module docstring 참고)이 이 배치 평가 경로에는 연결이 안 되어 있었다. 여기서 다시 연결한다.

    반환하는 ground_fn은 어느 쪽이든 gui_grounding.ground()와 동일한 시그니처
    ground_fn(model, instruction, screenshot, **kwargs) -> {"result", "point", "raw_response", ...}
    로 통일되어 있어(region_focus.ground_with_regionfocus() docstring 참고), 호출부
    (_convert_planner_action_to_env)는 어느 백엔드가 실제로 도는지 몰라도 된다. 두 백엔드가
    서로 못 알아듣는 kwargs(gui_grounding.ground()는 task_id 개념이 없고, ground_with_regionfocus()는
    max_new_tokens를 안 받음 - 내부 각 단계가 자체 max_new_tokens를 하드코딩해서 씀)는 여기서
    조용히 걸러낸다.

    [v3 추가] grounding_backend="gui_actor"면 LoRA 대신 GUI-Actor(microsoft/GUI-Actor-3B-
    Qwen2.5-VL, coordinate-free pointer 방식)를 쓴다. use_regionfocus는 이 경우에도 그대로
    존중된다 - True면 gui_actor_region_focus.ground_with_regionfocus_gui_actor()(judge/재탐색/
    crop-zoom/aggregation을 GUI-Actor 위에서 재현한 버전), False면 gui_actor_grounding.ground()
    (초기 grounding 1회)로 간다. LoRA 전용 region_focus.ground_with_regionfocus()는 여기서
    안 탄다 - 그쪽 judge/재탐색 파이프라인이 LoRA의 학습 포맷(coord_utils.PROMPT_TEMPLATE)에
    강하게 결합돼 있어 GUI-Actor에는 그대로 못 얹기 때문에, gui_actor_region_focus.py로 따로
    재구현했다(모듈 docstring 참고). 두 gui_actor 경로 모두 min_pixels/max_pixels/max_new_tokens
    등 낯선 kwargs를 알아서 무시하므로(**_ignored) 여기서 따로 걸러낼 필요가 없다.
    """
    if grounding_backend == "gui_actor":
        # [v3 수정 - RF 결합] 처음엔 gui_actor 백엔드에서 RegionFocus를 아예 안 쓰고
        # 초기 grounding 1회(gui_actor_grounding.ground())만 쓰도록 고정했었는데,
        # gui_actor_region_focus.py로 RegionFocus 알고리즘(judge/재탐색/crop-zoom/aggregation)을
        # GUI-Actor 위에서 재현했으므로 이제 use_regionfocus를 그대로 존중한다 - LoRA 경로와
        # 동일하게 켜져 있으면 RegionFocus, 꺼져 있으면(--no_regionfocus) 초기 grounding 1회.
        if use_regionfocus:
            from gui_actor_region_focus import ground_with_regionfocus_gui_actor

            def _ground_fn(model, instruction, screenshot, **kwargs):
                task_id = kwargs.pop("task_id", None)
                kwargs.pop("max_new_tokens", None)
                return ground_with_regionfocus_gui_actor(
                    model, instruction, screenshot,
                    debug_image=regionfocus_debug_image,
                    debug_text=regionfocus_debug_text,
                    task_id=task_id,
                    **kwargs,  # min_pixels/max_pixels가 남아있어도 **_ignored로 조용히 무시됨
                )

            return _ground_fn

        from gui_actor_grounding import ground as _gui_actor_ground

        def _ground_fn(model, instruction, screenshot, **kwargs):
            return _gui_actor_ground(model, instruction, screenshot, **kwargs)

        return _ground_fn

    if not use_regionfocus:
        from gui_grounding import ground

        def _ground_fn(model, instruction, screenshot, **kwargs):
            kwargs.pop("task_id", None)
            return ground(model, instruction, screenshot, **kwargs)

        return _ground_fn

    from region_focus import ground_with_regionfocus

    def _ground_fn(model, instruction, screenshot, **kwargs):
        task_id = kwargs.pop("task_id", None)
        kwargs.pop("max_new_tokens", None)
        return ground_with_regionfocus(
            model, instruction, screenshot,
            debug_image=regionfocus_debug_image,
            debug_text=regionfocus_debug_text,
            task_id=task_id,
            step1_format=regionfocus_step1_format,
            step4_format=regionfocus_step4_format,
            **kwargs,  # min_pixels/max_pixels만 남아있으면 그대로 통과
        )

    return _ground_fn


# ---------------------------------------------------------------------------
# (2026-08-09 추가) 실제 정책: planner LoRA(plan) + grounding LoRA(좌표) 연결
# ---------------------------------------------------------------------------
def _convert_planner_action_to_env(
    plan: dict, grounding_model, screenshot, ground_kwargs: dict, ground_fn=None, task_id=None,
):
    """
    agent/planner.py의 출력 스키마(자연어 target_description/action/text)를
    env_webvoyager.WebVoyagerEnv.execute_action()이 기대하는 스키마(픽셀 coordinate)로
    변환한다. click류는 ground_fn(기본은 gui_grounding.ground(), build_planner_grounding_agent_step이
    use_regionfocus=True로 만들었으면 region_focus.ground_with_regionfocus() - 위
    _build_click_ground_fn() 참고)을 호출해서 target_description을 실제 좌표로 바꾼다(이 시점에
    grounding_model은 이미 grounding LoRA가 활성 상태라고 가정 - agent_loop._AdapterSwitchView.
    generate()가 planning 호출 뒤 자동으로 default(grounding)로 복원해주므로,
    plan_next_action()/plan_with_reflection() 호출 직후 여기로 넘어올 때는 항상 그 상태다).

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
    if ground_fn is None:
        ground_fn = _build_click_ground_fn(use_regionfocus=False)

    act = plan.get("action")

    if act == "terminate":
        # run_episode()는 final_answer를 action.get("text")에서 읽는데, planner.py의
        # terminate 스키마는 "answer" 필드를 쓴다 - 여기서 다리를 놓아준다.
        return {"action": "terminate", "status": plan.get("status", "failure"), "text": plan.get("answer")}

    if act in ("left_click", "double_click", "right_click"):
        target = plan.get("target_description") or ""
        g = ground_fn(grounding_model, target, screenshot, task_id=task_id, **ground_kwargs)
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
        # (2026-08 추가 - 구현) env_webvoyager.py의 _drag()가 이제 start_coordinate/coordinate
        # 두 좌표를 받아 실제로 실행한다(모듈 docstring 참고). planner.py의 drag 스키마는
        # target_description=시작점 설명, text=끝점 설명이라(기존 필드 재사용, 새 필드 없음),
        # 여기서 두 지점을 각각 ground_fn으로 grounding해서 픽셀 좌표로 변환한다 - click류와
        # 똑같은 로직을 두 번 반복하는 셈. 둘 중 하나라도 설명이 비어있거나 grounding이
        # 실패하면(원인 불문) 에피소드를 죽이는 대신 no-op으로 다운그레이드한다(click 실패
        # 처리와 동일한 원칙 - _downgrade_reason 마커로 history 보정도 동일하게 적용됨).
        start_target = plan.get("target_description") or ""
        end_target = plan.get("text") or ""
        if not start_target or not end_target:
            print(
                f"[agent_step] drag 액션에 시작점/끝점 설명이 부족함"
                f"(target_description={start_target!r}, text={end_target!r}) -> no-op으로 스킵"
            )
            return {
                "action": "wait",
                "time": 0.5,
                "_downgrade_reason": (
                    f"drag missing target_description or text "
                    f"(target_description={start_target!r}, text={end_target!r})"
                ),
            }

        g_start = ground_fn(grounding_model, start_target, screenshot, task_id=task_id, **ground_kwargs)
        if g_start["result"] != "positive":
            print(f"[agent_step] drag 시작점 grounding 실패(target={start_target!r}) -> no-op으로 스킵")
            return {
                "action": "wait",
                "time": 0.5,
                "_downgrade_reason": (
                    f"grounding failed for drag start target_description={start_target!r} "
                    f"(result={g_start.get('result')!r})"
                ),
            }

        g_end = ground_fn(grounding_model, end_target, screenshot, task_id=task_id, **ground_kwargs)
        if g_end["result"] != "positive":
            print(f"[agent_step] drag 끝점 grounding 실패(target={end_target!r}) -> no-op으로 스킵")
            return {
                "action": "wait",
                "time": 0.5,
                "_downgrade_reason": (
                    f"grounding failed for drag end text={end_target!r} (result={g_end.get('result')!r})"
                ),
            }

        w, h = screenshot.size
        x1, y1 = g_start["point"][0] * w, g_start["point"][1] * h
        x2, y2 = g_end["point"][0] * w, g_end["point"][1] * h
        return {"action": "left_click_drag", "start_coordinate": [x1, y1], "coordinate": [x2, y2]}

    if act == "type":
        return {"action": "type", "text": plan.get("text", "")}

    if act == "key":
        return {"action": "key", "text": plan.get("text", "")}

    if act == "scroll":
        return {"action": "scroll", "text": plan.get("text", "down")}

    if act == "wait":
        return {"action": "wait", "time": 1.0}

    if act == "back":
        # (2026-08-11 추가) 브라우저 히스토리 뒤로가기 - grounding이 필요 없는 액션이라
        # 그대로 통과. env_webvoyager.WebVoyagerEnv._back()이 실제 driver.back()을 호출.
        return {"action": "back"}

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


# (2026-08-15 추가 - _extract_final_answer가 헛소리를 걸러내기 위한 안전장치) 일부 모델(특히
# GUI-Actor처럼 자유형 QA를 학습에서 못 본 모델)이 답변 뒤에 챗 템플릿/액션 포맷 토큰을 이어서
# 계속 생성해버리는 게 실측으로 확인됨(예: "unknown.\nassistantos\npyautogui.click([1] )") -
# 이런 패턴이 나오면 진짜 답변이 아니라 생성 이탈이므로, 이 마커가 나오는 지점 이후는 잘라낸다.
_ANSWER_LEAKAGE_CUT_MARKERS = ("\nassistant", "\nuser:", "\n<|")


def _extract_final_answer(qa_model, instruction: str, screenshot, max_new_tokens: int = 100):
    """
    (2026-08-10 추가) planner LoRA가 terminate 시점에 "answer"를 채우지 못하는 문제(실측:
    질문형 태스크 3개 중 3개 전부 final_answer=null)의 원인을 찾아보니, 학습 데이터
    (prepare_planner_dataset.py의 terminate 변환)가 AgentNet에 없는 "최종 답변 텍스트"를
    아예 채운 적이 없어서였다 - AgentNet 자체가 이 정보를 갖고 있지 않아 재학습으로 고칠
    수 있는 문제가 아니다. 그래서 "행동 결정"과 "최종 답변 추출"을 분리해서, terminate인데
    answer가 비어 있을 때만 이 함수로 별도 QA 호출을 한 번 더 한다(WebVoyager 등 여러
    에이전트 시스템이 실제로 쓰는 분리 방식).

    qa_model.model이 peft.PeftModel이면(어댑터가 얹혀 있으면) disable_adapter()로 순수 base
    상태에서 물어본다 - LoRA가 이 자유형 QA 포맷을 학습에서 본 적이 없어서, 얹은 채로 물으면
    오히려 이상한 포맷으로 답할 위험이 있다(이 세션 내내 반복된 원칙: LoRA는 자기가 학습받은
    입력 구조로만 물어야 함).

    [2026-08-15 수정 - 호출부가 grounding_model 대신 planner 모델을 넘기도록 변경] 원래
    build_planner_grounding_agent_step()이 이 함수에 grounding_model(그라운딩 전용 모델)을
    넘겼는데, grounding_backend가 GUI-Actor(좌표 찍기 전문, 자유형 질의응답 학습 안 됨)로
    바뀐 뒤로 실측에서 "unknown.\nassistantos\npyautogui.click([1] )" 같은 생성 이탈이
    확인됨 - GUI-Actor한테 범용 화면 판독 질문을 던지는 게 애초에 안 맞는 조합이었다. 이제는
    이미 사용 중인 planner 모델(대부분 GPT-4o 등 OpenAI API - 비전 QA를 훨씬 잘함)을 이
    함수에 넘기도록 호출부를 바꿨다 - 새 API 호출이 추가되는 게 아니라 자리만 바뀐 것.
    로컬 planner LoRA를 쓰는 경우에도 위 disable_adapter() 로직이 그대로 적용되므로 이전과
    동일하게 안전하다(오히려 grounding LoRA보다 planner 쪽이 애초에 이 함수가 상정한
    "LoRA를 끈 순수 base"에 더 가까운 자리였음).

    [2026-08-15 추가 - 헛소리 감지 강화] "unknown" 감지가 정확히 일치할 때만 걸러졌는데,
    실제로는 "unknown." 처럼 구두점이 붙거나 뒤에 챗 템플릿 이탈 텍스트가 더 붙는 경우가
    있어서 못 걸렀다(위 실측 사례) - 마침표/느낌표를 떼고 비교하고, 알려진 이탈 마커
    이후는 아예 잘라내는 전처리를 추가했다.
    """
    prompt = _ANSWER_EXTRACTION_PROMPT_TEMPLATE.format(instruction=instruction)
    messages = [
        {"role": "user", "content": [{"type": "image", "image": screenshot}, {"type": "text", "text": prompt}]}
    ]

    def _generate():
        return qa_model.generate(messages, max_new_tokens=max_new_tokens, temperature=0.0).strip()

    if hasattr(qa_model.model, "disable_adapter"):
        with qa_model.model.disable_adapter():
            response = _generate()
    else:
        response = _generate()

    if not response:
        return None

    for marker in _ANSWER_LEAKAGE_CUT_MARKERS:
        idx = response.lower().find(marker)
        if idx != -1:
            response = response[:idx].strip()

    if not response or response.strip().lower().rstrip(".!") == "unknown":
        return None
    return response


# ---------------------------------------------------------------------------
# (2026-08-11 추가) 태스크별 프롬프트/응답 덤프
# ---------------------------------------------------------------------------
# 지금까지는 콘솔에 찍힌 요약 로그(action/target/status 등)만 봤는데, 실제로 모델에 뭐가
# 들어갔는지(시스템 프롬프트 전문, history 렌더링 결과, reflection critique 원문 등)를
# 봐야 진단이 되는 경우가 많았다. build_planner_grounding_agent_step()이 쓰는
# planning_view/reflection_view/grounding_model을 이 얇은 proxy로 감싸서, .generate()가
# 호출될 때마다 프롬프트(텍스트 부분)와 응답을 <debug_dir>/<태스크>/stepNN_<태그>_NN.txt로
# 저장한다 - 기존 코드/테스트는 debug_dir=None(기본값)이면 이 경로를 아예 안 타서 안 건드림.
def _render_messages_for_debug(messages: list, image_filenames: dict | None = None) -> str:
    """.generate()에 넘어간 messages(Qwen 챗 포맷)를 사람이 읽을 텍스트로 풀어준다.
    image_filenames는 {id(part): "저장된파일명.png"} 매핑(_PromptRecorder._save_images가
    만듦) - 주어지면 그 파일명을 같이 적어주고, 없으면(이미지 저장을 껐거나 저장 실패)
    크기 정보만 남긴다."""
    image_filenames = image_filenames or {}
    lines = []
    for m in messages or []:
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, str):
            lines.append(f"[{role}]\n{content}")
            continue
        for part in content or []:
            ptype = part.get("type")
            if ptype == "text":
                lines.append(f"[{role} text]\n{part.get('text', '')}")
            elif ptype == "image":
                img = part.get("image")
                size = getattr(img, "size", None)
                fname = image_filenames.get(id(part))
                if fname:
                    lines.append(f"[{role} image] size={size} -> 저장됨: {fname}")
                else:
                    lines.append(f"[{role} image] <PIL.Image size={size}, 저장 안 함>")
            elif ptype == "image_url":
                lines.append(f"[{role} image_url] <생략>")
            else:
                lines.append(f"[{role} {ptype}] {part!r}")
    return "\n\n".join(lines)


class _PromptRecorder:
    """태스크별 폴더(<base_dir>/<태스크 키>/)를 만들어서 각 스텝에서 모델에 실제로 들어간
    프롬프트(+ 프롬프트에 포함된 스크린샷 이미지)/응답을 남긴다. save_images=True(기본)면
    프롬프트에 포함된 각 이미지를 stepNN_<태그>_NN_imgK.png로 같이 저장한다 - 나중에 "이
    스텝에서 모델이 정확히 뭘 보고 이 판단을 했는지" 프롬프트 텍스트와 같이 바로 확인할 수
    있게 하기 위함."""

    def __init__(self, base_dir: str, save_images: bool = True):
        self.base_dir = base_dir
        self.save_images = save_images
        self.task_dir = None
        self.step_idx = -1
        self._counts: dict = {}
        self._need_new_task = True

    def mark_new_task(self):
        # agent_step_fn.reset_episode()에서 호출됨 - 실제 폴더 생성은 다음 begin_step()에서
        # task_info(태스크 id 등)를 받을 때 한다(reset_episode 시점엔 아직 다음 태스크의
        # task_info를 모름 - run_episode가 reset_episode() 다음에야 env.reset(task)를 부름).
        self._need_new_task = True

    def begin_step(self, task_info: dict):
        if self._need_new_task:
            key = (
                (task_info or {}).get("id")
                or (task_info or {}).get("web_name")
                or (task_info or {}).get("instruction")
                or "task"
            )
            self._start_task(key)
            self._need_new_task = False
        else:
            self.step_idx += 1

    def _start_task(self, key) -> None:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(key))[:80] or "task"
        self.task_dir = os.path.join(self.base_dir, safe)
        os.makedirs(self.task_dir, exist_ok=True)
        self.step_idx = 0
        self._counts = {}

    def _save_images(self, messages: list, base_name: str) -> dict:
        """messages 안의 PIL.Image 파트들을 <base_name>_imgK.png로 저장하고,
        {id(part): 파일명} 매핑을 돌려준다. 저장 실패(이미지가 아니거나 I/O 에러)는 그냥
        건너뛴다 - 프롬프트 텍스트 로그 자체는 그것 때문에 실패하면 안 되니까."""
        image_filenames: dict = {}
        if not self.save_images:
            return image_filenames
        img_idx = 0
        for m in messages or []:
            content = m.get("content")
            if isinstance(content, str):
                continue
            for part in content or []:
                if part.get("type") != "image":
                    continue
                img = part.get("image")
                if img is None or not hasattr(img, "save"):
                    continue
                img_fname = f"{base_name}_img{img_idx}.png"
                try:
                    img.save(os.path.join(self.task_dir, img_fname))
                    image_filenames[id(part)] = img_fname
                except Exception as e:  # noqa: BLE001 - 이미지 저장 실패로 로그 전체를 죽이지 않음
                    print(f"[gui_actor_eval_webvoyager.py] 디버그 이미지 저장 실패(무시하고 진행): {e}")
                img_idx += 1
        return image_filenames

    def record(self, tag: str, messages: list, response) -> str | None:
        if self.task_dir is None:
            return None
        count_key = (self.step_idx, tag)
        n = self._counts.get(count_key, 0)
        self._counts[count_key] = n + 1
        base_name = f"step{self.step_idx:02d}_{tag}_{n:02d}"
        image_filenames = self._save_images(messages, base_name)
        fname = os.path.join(self.task_dir, base_name + ".txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write("=== PROMPT ===\n")
            f.write(_render_messages_for_debug(messages, image_filenames))
            f.write("\n\n=== RESPONSE ===\n")
            f.write(response if response is not None else "")
        return fname


class _DebugModelView:
    """.generate() 호출을 그대로 통과시키면서 recorder에 프롬프트/응답을 기록하는 얇은
    proxy. agent_loop._BaseModelView/_AdapterSwitchView와 같은 duck-typing 원칙으로,
    .model/.processor 등 내부 객체의 속성을 그대로 통과시켜서(disable_adapter() 등 내부에서
    .model을 쓰는 코드, region_focus.py처럼 .generate()를 거치지 않고 .model/.processor를
    직접 꺼내 쓰는 코드 모두) 다른 코드는 이게 debug wrapper인지 전혀 모른다.

    (2026-08-11 추가 - 버그 수정) 처음엔 .model만 명시적으로 통과시켰는데, RegionFocus
    재연결 이후 region_focus.py의 judge_inference()가 .generate()가 아니라
    qwen_model.model / qwen_model.processor를 직접 꺼내서 쓰는 걸 실제로 돌려보고서야
    발견했다(AttributeError: '_DebugModelView' object has no attribute 'processor').
    앞으로 또 이런 식으로 직접 접근하는 속성이 나올 수 있으니, 개별 프로퍼티를 하나씩
    추가하는 대신 __getattr__로 알려지지 않은 속성 전부를 내부 객체에 위임한다 - generate()
    처럼 가로채서 기록해야 하는 것만 이 클래스에 명시적으로 정의하고, 나머지는 전부
    통과시키는 게 이런 종류의 버그를 구조적으로 막는다.
    """

    def __init__(self, inner, recorder: _PromptRecorder, tag: str):
        self._inner = inner
        self._recorder = recorder
        self._tag = tag

    def __getattr__(self, name):
        # generate()는 아래 명시적으로 정의돼 있어서 여기로 안 옴 - 그 외 모든 속성/메서드
        # 접근(.model, .processor, .device, 앞으로 생길 수 있는 것들)은 내부 객체로 위임.
        return getattr(self._inner, name)

    def generate(self, messages, **kwargs):
        response = self._inner.generate(messages, **kwargs)
        self._recorder.record(self._tag, messages, response)
        return response


def build_planner_grounding_agent_step(
    grounding_model,
    planning_view,
    use_reflection: bool = False,
    max_iterations: int = 2,
    planner_max_new_tokens: int = 300,
    ground_max_new_tokens: int = 128,
    ground_min_pixels: int | None = None,
    ground_max_pixels: int | None = None,
    verbose: bool = True,
    debug_dir: str | None = None,
    debug_save_images: bool = True,
    use_regionfocus: bool = True,
    regionfocus_debug_image: bool = False,
    regionfocus_debug_text: bool = False,
    regionfocus_step1_format: str = "point_text",
    regionfocus_step4_format: str = "point_text",
    grounding_backend: str = "lora",
    # [v3 추가] "gui_actor"면 click grounding이 _build_click_ground_fn(grounding_backend="gui_actor")를
    # 타서, use_regionfocus 값에 따라 gui_actor_region_focus.ground_with_regionfocus_gui_actor()
    # (True) 또는 gui_actor_grounding.ground()(False, 초기 grounding 1회)로 간다 - 둘 다 이제
    # gui_actor 백엔드에서 지원된다(gui_actor_region_focus.py 모듈 docstring 참고). reflection은
    # __main__ CLI 쪽에서 gui_actor 백엔드일 때 기본으로 꺼둔다(GUI-Actor가 학습에서 못 본
    # 구조화된 비평 포맷이라 출력 품질 미검증 - 이 함수를 CLI 밖에서 직접 호출할 거면 호출부가
    # 그 판단을 대신 해야 한다).
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

    [2026-08-11 추가 - region focus 재연결]
    use_regionfocus=True(기본)면 click 계열 액션의 grounding을 gui_grounding.ground()(초기
    grounding 1회) 대신 region_focus.ground_with_regionfocus()(재탐색+crop/zoom 정밀화, 이
    프로젝트가 실제로 grounding 정확도를 끌어올린 것으로 확인한 파이프라인 - region_focus.py
    module docstring 참고)로 돌린다. click 1회당 모델 호출이 5~9회로 늘어나서 스텝당 훨씬
    느려지므로, 빠르게 배선만 확인하고 싶으면 use_regionfocus=False(CLI는 --no_regionfocus)로
    끌 것. _build_click_ground_fn() 참고.
    """
    from planner import plan_next_action, plan_with_reflection

    click_ground_fn = _build_click_ground_fn(
        grounding_backend=grounding_backend,
        use_regionfocus=use_regionfocus,
        regionfocus_debug_image=regionfocus_debug_image,
        regionfocus_debug_text=regionfocus_debug_text,
        regionfocus_step1_format=regionfocus_step1_format,
        regionfocus_step4_format=regionfocus_step4_format,
    )

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

    # (2026-08-11 추가) debug_dir가 주어지면 planning/reflection/grounding 각각의 .generate()를
    # _DebugModelView로 감싼다 - 실제로 뭘 호출하는지는 그대로, 프롬프트/응답만 옆에서 기록.
    recorder = _PromptRecorder(debug_dir, save_images=debug_save_images) if debug_dir else None
    if recorder is not None:
        debug_planning_view = _DebugModelView(planning_view, recorder, "planner")
        debug_reflection_view = (
            _DebugModelView(reflection_view, recorder, "reflection") if reflection_view is not None else None
        )
        debug_grounding_click_view = _DebugModelView(grounding_model, recorder, "grounding")
        # (2026-08-15 수정) grounding_model 대신 planning_view를 answer 추출에 씀 -
        # _extract_final_answer() docstring의 "2026-08-15 수정" 항목 참고. debug_planning_view를
        # 그대로 재사용하면 "planner" 라벨로 기록되어 버려서(진짜 planning 콜과 구분이 안 됨),
        # 같은 planning_view를 별도 라벨("answer_extraction")로 한 번 더 감싼다.
        debug_answer_extraction_view = _DebugModelView(planning_view, recorder, "answer_extraction")
    else:
        debug_planning_view = planning_view
        debug_reflection_view = reflection_view
        debug_grounding_click_view = grounding_model
        debug_answer_extraction_view = planning_view

    ground_kwargs = {"max_new_tokens": ground_max_new_tokens}
    if ground_min_pixels is not None:
        ground_kwargs["min_pixels"] = ground_min_pixels
    if ground_max_pixels is not None:
        ground_kwargs["max_pixels"] = ground_max_pixels

    def agent_step_fn(screenshot, task_info, history):
        if recorder is not None:
            recorder.begin_step(task_info)

        instruction = task_info["instruction"]
        if use_reflection:
            plan = plan_with_reflection(
                debug_planning_view, instruction, screenshot,
                history_actions=planner_history,
                max_new_tokens=planner_max_new_tokens,
                max_iterations=max_iterations,
                reflection_model=debug_reflection_view,
            )
        else:
            plan = plan_next_action(
                debug_planning_view, instruction, screenshot,
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

        # (2026-08-10 추가, 2026-08-15 수정) terminate인데 answer가 비어 있으면 별도 QA 호출로
        # 채운다. _extract_final_answer() docstring 참고 - planner가 애초에 이 필드를 못 채우는
        # 문제라 재시도/reflection으로는 안 고쳐짐. reflection이 반려한 terminate는 위에서 이미
        # 걸러졌으니, 여기 도달하는 terminate는 (reflection이 껐거나) 승인된 것만 남는다.
        # [2026-08-15 수정] grounding_model(GUI-Actor 등 좌표 전문 모델) 대신 planning_view를
        # 넘기도록 바꿈 - GUI-Actor한테 자유형 QA를 물었더니 "unknown.\nassistantos\n
        # pyautogui.click([1] )" 같은 생성 이탈이 실측으로 확인됨(_extract_final_answer 함수
        # docstring의 "2026-08-15 수정" 항목 참고).
        if plan.get("action") == "terminate" and not plan.get("answer"):
            try:
                extracted = _extract_final_answer(debug_answer_extraction_view, instruction, screenshot)
            except Exception as e:  # noqa: BLE001 - 최종 답변 추출 실패로 에피소드 전체를 죽이지 않음
                print(f"[agent_step] answer 추출 실패(무시하고 진행): {e}")
                extracted = None
            if extracted:
                plan = dict(plan)
                plan["answer"] = extracted

        env_action = _convert_planner_action_to_env(
            plan, debug_grounding_click_view, screenshot, ground_kwargs,
            ground_fn=click_ground_fn, task_id=(task_info or {}).get("id"),
        )

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

    # (2026-08-11 추가 - 버그 수정) planner_history는 이 클로저가 빌드될 때 딱 한 번만 만들어지는데,
    # agent_step_fn 자체는 run_batch()가 여러 태스크에 걸쳐 재사용한다 - 그래서 task 1이 끝날 때
    # 남긴 마지막 기록(예: "terminate: success")이 지워지지 않고 task 2의 첫 스텝 컨텍스트로 그대로
    # 새어 들어갔다(실측: task 1은 6스텝 정상 진행 후 성공, task 2~10은 전부 1스텝만에 바로
    # terminate/success -> "나 방금 이미 끝냈잖아"로 착각한 것과 정확히 일치하는 패턴).
    # agent_step_fn에 넘어오는 history 인자(actions/screenshots)를 보고 "비어있으면 새 에피소드"로
    # 추론하는 방법도 있지만, 이 함수의 unit test들이 전부 매 호출마다 history를 {"actions": [],
    # "screenshots": []}로 단순화해서 넘기고 있어서(실제 run_episode처럼 스텝마다 채워 넣지 않음)
    # 그 추론 방식은 기존 테스트들과 의미가 충돌한다. 대신 명시적인 reset_episode() 훅을 붙여서
    # run_batch()/run_episode()가 매 태스크 시작 시점에 직접 부르게 한다 - 더 명확하고, 기존
    # history 인자의 의미도 안 건드린다.
    def _reset_episode():
        planner_history.clear()
        if recorder is not None:
            recorder.mark_new_task()

    agent_step_fn.reset_episode = _reset_episode

    return agent_step_fn


# ---------------------------------------------------------------------------
# trajectory 수집
# ---------------------------------------------------------------------------
def _action_fingerprint(action: dict) -> tuple:
    """
    (2026-08-11 추가) 두 액션이 "사실상 같은 시도"인지 비교하기 위한 단순화된 키.
    좌표는 10px 단위로 뭉개서 비교한다 - RegionFocus가 매번 정확히 똑같은 픽셀을 찍으리라는
    보장이 없어서(약간의 흔들림 정도는 "같은 시도"로 취급하고 싶음), 완전 일치 대신 근접
    일치로 반복을 감지한다.
    """
    coord = action.get("coordinate")
    coord_key = tuple(round(c / 10) * 10 for c in coord) if coord else None
    return (action.get("action"), coord_key, action.get("text"))


def run_episode(
    env: WebVoyagerEnv, task, agent_step_fn, max_steps=DEFAULT_MAX_STEPS,
    stuck_repeat_threshold=DEFAULT_STUCK_REPEAT_THRESHOLD,
    stuck_repeat_window=None,
):
    """
    task를 env에 reset하고, agent_step_fn이 "terminate"를 낼 때까지(또는 max_steps
    도달까지) 액션을 실행한다.

    agent_step_fn(screenshot, task_info, history) -> action dict
        (gui_grounding.ComputerUseTool 스키마). "terminate" 액션이 나오면 그 자리에서
        멈춘다 - env.execute_action()에는 안 보냄(env_webvoyager.py가 terminate를
        거부하도록 만들어져 있으므로 여기서 걸러야 함).

    [2026-08-11 추가 - CAPTCHA/bot-check 대응]
    실제 사이트를 자동화로 돌리면 CAPTCHA/봇 감지 페이지에 막히는 경우가 실제로 있다
    (env_webvoyager.py 상단 주석의 Allrecipes 사례 참고). 이 함수는 CAPTCHA를 풀거나
    우회하지 않는다(그건 이 프로젝트가 할 일이 아님) - 대신 두 가지 신호로 "이 태스크는
    막혔다"를 최대한 빨리, 정직하게 알아채고 max_steps를 낭비하지 않도록 조기 종료한다.

    1) env.detect_bot_check() (env_webvoyager.WebVoyagerEnv에 있으면) - title/URL/iframe
       기준으로 CAPTCHA/bot-check 페이지인지 저렴하게 확인. reset() 직후와 매 스텝 이후
       모두 확인한다(reset 시점엔 없다가 특정 페이지로 이동하면서 나타나는 경우도 있어서).
    2) 최근 stuck_repeat_window개 액션 안에서 같은 액션이 stuck_repeat_threshold회 이상
       등장하면 원인(CAPTCHA든 다른 이유든)과 무관하게 "멈췄다"고 보고 종료 -
       detect_bot_check()가 못 잡는 케이스(알려지지 않은 문구를 쓰는 봇 차단 페이지 등)까지
       잡아내는 일반화된 안전장치. (2026-08-11 수정) 원래는 "연속"만 봤는데, 이러면 두세 개
       액션을 번갈아 반복하는 뺑뺑이(A -> B -> A -> B)를 못 잡아서 윈도우 내 빈도 기반으로
       바꿨다 - planner.py의 REPETITION WARNING과 동일한 문제의식/해법.

    env가 detect_bot_check()를 제공하지 않아도(구버전 env, 또는 테스트용 mock) 그냥
    건너뛰고 정상 동작한다 - 이 함수가 env_webvoyager.WebVoyagerEnv 전용으로 굳어지지
    않도록 duck-typing으로 처리.

    Returns: dict {
        "instruction": str, "url": str,
        "screenshots": [PIL.Image, ...]   # 스텝별 전체 - judge에는 마지막 N장만 넘길 것
        "actions": [action_dict, ...],
        "final_answer": str | None,       # terminate action의 "text" 필드(있으면)
        "n_steps": int,
        "hit_max_steps": bool,
        "blocked": bool,                  # CAPTCHA/bot-check 또는 반복-정체로 조기 종료됐는지
        "blocked_reason": str | None,
    }
    """
    # (2026-08-11 추가 - 버그 수정) agent_step_fn이 build_planner_grounding_agent_step()으로
    # 만들어진 경우, 그 안의 planner_history는 여러 태스크에 걸쳐 재사용되는 클로저 변수라 매
    # 에피소드 시작 시점에 명시적으로 비워줘야 한다(자세한 설명은 build_planner_grounding_
    # agent_step()의 reset_episode 주석 참고) - dummy_agent_step처럼 이 훅이 없는 함수는
    # hasattr로 걸러서 그냥 넘어간다.
    reset_episode = getattr(agent_step_fn, "reset_episode", None)
    if callable(reset_episode):
        reset_episode()

    screenshot, task_info = env.reset(task)
    screenshots = [screenshot]
    actions = []
    final_answer = None
    hit_max_steps = True
    blocked = False
    blocked_reason = None

    initial_bot_check = task_info.get("_bot_check_at_reset") if hasattr(task_info, "get") else None
    if initial_bot_check:
        # env가 자체적으로 재시도(예: env_webvoyager.WebVoyagerEnv의 captcha_reset_retries)를
        # 다 써보고도 여전히 감지된 상태로 넘어온 것 - agent가 뭘 하든 CAPTCHA는 못 뚫으니
        # 첫 스텝도 진행하지 않고 바로 blocked로 끝낸다.
        blocked = True
        blocked_reason = f"reset 시점부터 bot-check 감지: {initial_bot_check.get('reason')}"
        hit_max_steps = False

    if stuck_repeat_window is None:
        stuck_repeat_window = stuck_repeat_threshold * 2
    fingerprint_history: list = []
    consecutive_wait = 0
    if not blocked:
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

            detect_fn = getattr(env, "detect_bot_check", None)
            bot_check = detect_fn() if callable(detect_fn) else None
            if bot_check:
                # (2026-08-11 추가 - 수동 CAPTCHA 통과) env가 wait_for_manual_captcha()를
                # 제공하면(env_webvoyager.WebVoyagerEnv, manual_captcha_wait=True일 때만
                # 실제로 멈춤) 사람이 직접 풀 기회를 준다 - duck-typing이라 이 메서드가 없는
                # 구버전 env/mock은 그냥 기존처럼 즉시 blocked 처리된다. "is True"로 엄격하게
                # 비교하는 이유: 테스트에서 env를 MagicMock()으로만 만들면 존재하지 않는
                # 속성 접근도 전부 자동으로 또 다른 MagicMock을 만들어내서(auto-speccing),
                # wait_for_manual_captcha()도 "그냥 호출 가능하고 뭔가를 반환하는" 것처럼
                # 보여 truthy 체크로는 실수로 "풀렸다"고 오판할 수 있다 - 실제 True를
                # 명시적으로 반환하는 경우(env_webvoyager.WebVoyagerEnv.wait_for_manual_captcha()
                # 또는 그렇게 설계된 테스트용 mock)만 인정한다.
                manual_wait_fn = getattr(env, "wait_for_manual_captcha", None)
                resolved_manually = callable(manual_wait_fn) and manual_wait_fn() is True
                if not resolved_manually:
                    blocked = True
                    blocked_reason = f"스텝 진행 중 bot-check 감지: {bot_check.get('reason')}"
                    hit_max_steps = False
                    break

            # (2026-08-15 수정 - 사용자 요청) scroll은 뺑뺑이/정체 판단에서 아예 뺀다. planner.py
            # 쪽 REPETITION WARNING도 같은 이유로 scroll을 뺐음(_is_similar_action 참고) -
            # scroll은 매번 화면의 새로운 부분을 보여주는 정상 탐색이라 반복 자체가 "제자리걸음"
            # 신호가 아니고, "개수가 명시된 태스크는 목표를 채울 때까지 계속 스크롤하라"는 최근
            # 프롬프트 지침과 하드 조기종료가 서로 충돌하면 안 되므로(정상적으로 스크롤하며
            # 목록을 훑던 에피소드가 스크롤 반복만으로 blocked 처리되는 걸 방지) fingerprint
            # 히스토리에 아예 안 쌓는다 - 진짜로 스크롤만 하며 끝까지 진전이 없는 경우는 max_steps로
            # 자연스럽게 막힌다.
            # (2026-08-15 추가 - 사용자 요청) wait도 같은 이유로 뺀다 - "최근 윈도우 안 빈도"로
            # 세는 방식이라, 연속 반복이 아니라 서로 다른 정상 액션들 사이사이에 한 번씩만 끼어
            # 나와도(예: click→wait→type→wait→...) 개수가 금방 threshold에 도달해서 실제로는
            # 전혀 안 막혔는데 뺑뺑이로 오판될 수 있다 - wait은 다른 진짜 액션의 부수적 대기일
            # 뿐이라 그 자체로 "제자리걸음"의 신호가 아니다.
            if action.get("action") not in ("scroll", "wait"):
                fp = _action_fingerprint(action)
                fingerprint_history.append(fp)
                window = fingerprint_history[-stuck_repeat_window:]
                fp_count = window.count(fp)
                if fp_count >= stuck_repeat_threshold:
                    blocked = True
                    blocked_reason = (
                        f"최근 {len(window)}개 액션 중 같은 액션이 {fp_count}회 반복 등장(뺑뺑이/정체로 판단): {fp}"
                    )
                    hit_max_steps = False
                    break

            # (2026-08-16 추가 - wait 제외의 부작용 보완) 위에서 wait을 윈도우 빈도 카운트에서
            # 뺐더니, 다른 액션과 섞인 산발적 wait은 더 이상 오탐하지 않게 됐지만(의도한 수정),
            # 그 대가로 "다른 액션 없이 wait만 계속" 찍는 순수 연속 스팸까지 못 잡게 되는 실측
            # 회귀가 나왔다(WolframAlpha: 메인 페이지에서 못 넘어가고 wait만 반복하다 max_steps
            # 소진). 이건 윈도우 빈도와는 별개로 "직전부터 연속으로 wait이 몇 번인가"만 추가로
            # 세서 잡는다 - 중간에 다른 액션이 하나라도 끼면 0으로 리셋되므로 산발적 사용에는
            # 여전히 안 걸리고, 진짜 wait 연타만 걸린다.
            if action.get("action") == "wait":
                consecutive_wait += 1
            else:
                consecutive_wait = 0
            if consecutive_wait >= stuck_repeat_threshold:
                blocked = True
                blocked_reason = f"wait이 연속 {consecutive_wait}회 반복 등장(정체로 판단)"
                hit_max_steps = False
                break

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
        "blocked": blocked,
        "blocked_reason": blocked_reason,
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


def make_openai_judge(model="gpt-4o", api_key=None, max_tokens=300, max_retries=5, retry_base_delay=1.0):
    """
    OpenAI vision API(GPT-4V/GPT-4o 등)를 judge로 쓰는 judge_fn을 만들어 반환.
    WebVoyager/RegionFocus 논문과 동일한 방식. api_key=None이면 환경변수 OPENAI_API_KEY를
    사용(openai 클라이언트 기본 동작). openai 패키지는 반환된 judge_fn을 실제로 호출하는
    시점에만 필요 - make_openai_judge() 자체나 이 파일 import는 openai 미설치 환경에서도
    문제없다.

    (2026-08-11 추가 - 실측 크래시 대응) 실제 실행에서 TPM(분당 토큰) rate limit(HTTP 429)에
    걸려서 배치 전체가 죽는 걸 확인했다(judge는 이미지를 최대 MAX_JUDGE_SCREENSHOTS장씩
    보내서 planner보다도 토큰을 더 많이 씀 - rate limit에 걸리기 더 쉽다). api_planner.py의
    _call_with_retry/_is_rate_limit_error를 그대로 재사용해서, 429만 지수 백오프로 자동
    재시도하고 다른 에러/재시도 소진시엔 그대로 예외를 올린다.
    """

    def judge_fn(instruction, screenshots, final_answer):
        from openai import OpenAI  # 실제 호출 시점에만 필요 (lazy import)
        from api_planner import _call_with_retry

        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        imgs = screenshots[-MAX_JUDGE_SCREENSHOTS:]
        prompt = _build_judge_prompt(instruction, len(imgs), final_answer)

        content = [{"type": "text", "text": prompt}]
        for img in imgs:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        def _on_retry(attempt, delay, exc):
            print(f"[gui_actor_eval_webvoyager.py] judge rate limit(429) 감지 - {delay:.1f}초 대기 후 재시도 ({attempt}/{max_retries}): {exc}")

        resp = _call_with_retry(
            lambda: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
            ),
            max_retries=max_retries, base_delay=retry_base_delay, on_retry=_on_retry,
        )
        response_text = resp.choices[0].message.content
        success, reason = _parse_success_verdict(response_text)
        return {"success": success, "raw_response": response_text, "reason": reason}

    return judge_fn


def run_judge_with_repeats(
    judge_fn, instruction, screenshots, final_answer, n_repeats=DEFAULT_JUDGE_REPEATS, max_workers=1,
):
    """
    judge_fn을 n_repeats번 호출해서 다수결로 최종 success를 정한다(RegionFocus 논문이
    "GPT judge를 3회 돌려 평균/표준편차 보고"한 것의 실용적 버전). 개별 판정도 다 보존해서
    나중에 judge 자체의 변동성(분산)을 따로 분석할 수 있게 한다.

    (2026-08-11 추가 - 태스크 간 지연 단축) 실측 지적: 태스크가 끝나고 다음 태스크로
    넘어가기까지 텀이 길다. 원인은 run_episode() 자체가 아니라(그건 이미 driver 재사용으로
    줄임), 에피소드가 끝난 "직후" run_batch()가 이 함수를 부르는 시점 - n_repeats(기본 3)
    번의 judge_fn 호출이 완전히 순차(파이썬 리스트 컴프리헨션)로 돌아서, OpenAI API judge처럼
    호출 하나가 몇 초씩 걸리는 경우 태스크당 3배로 그 시간이 그대로 늘어난다.
    max_workers>1이면 ThreadPoolExecutor로 n_repeats번 호출을 동시에 쏴서 벽시계 시간을
    거의 1회분으로 줄인다 - OpenAI 호출은 서로 완전히 독립된 새 HTTP 요청이라 병렬화해도
    안전하다. 기본값은 max_workers=1(순차, 기존과 100% 동일 동작)로 둔 이유: 로컬 Qwen
    judge(같은 모델 인스턴스를 재사용)는 GPU 모델 하나를 여러 스레드에서 동시에 generate()
    호출하는 게 안전하다는 보장이 없어서(경쟁 상태/VRAM 문제 위험) - "이 judge_fn이 병렬
    호출을 견딜 수 있는가"는 호출부가 알고 명시적으로 올려야 한다(__main__에서
    --judge_backend openai일 때만 자동으로 올림).

    Returns: {"success": bool, "votes": [bool, ...], "agreement": float, "runs": [judge_fn 결과, ...]}
    """
    if max_workers <= 1:
        runs = [judge_fn(instruction, screenshots, final_answer) for _ in range(n_repeats)]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(judge_fn, instruction, screenshots, final_answer) for _ in range(n_repeats)]
            runs = [f.result() for f in futures]
    votes = [r["success"] for r in runs]
    success = Counter(votes).most_common(1)[0][0]
    agreement = sum(1 for v in votes if v == success) / len(votes)
    return {"success": success, "votes": votes, "agreement": agreement, "runs": runs}


# ---------------------------------------------------------------------------
# 배치 실행 (eval_regionfocus.py와 유사한 구조 - --resume 지원, 2026-08-11 추가)
# ---------------------------------------------------------------------------
def _task_key(task) -> str:
    """
    (2026-08-11 추가 - --resume) 태스크를 고유하게 식별하는 키. WebVoyager jsonl 레코드는
    보통 "id"(예: "Amazon--16")를 갖고 있어서 있으면 그걸 그대로 쓴다 - 이미 끝난 태스크인지
    판단하는 기준이라, 실행할 때마다 안정적으로 같은 값이 나와야 한다(instruction 텍스트만
    쓰면 같은 문구의 태스크가 우연히 여러 개일 때 잘못 스킵될 수 있어서 id를 우선함). "id"가
    없는 구버전 jsonl이나 (url, instruction) 튜플 태스크는 url+instruction 조합으로 폴백.
    """
    if isinstance(task, dict):
        tid = task.get("id")
        if tid:
            return str(tid)
        url = task.get("web") or task.get("url") or ""
        instruction = task.get("ques") or task.get("instruction") or ""
        return f"{url}|{instruction}"
    if isinstance(task, (list, tuple)) and len(task) == 2:
        return f"{task[0]}|{task[1]}"
    return str(task)


def run_batch(tasks, env, agent_step_fn, judge_fn, max_steps=DEFAULT_MAX_STEPS,
              judge_repeats=DEFAULT_JUDGE_REPEATS, out_path=None,
              stuck_repeat_threshold=DEFAULT_STUCK_REPEAT_THRESHOLD,
              stuck_repeat_window=None, resume=False, judge_max_workers=1,
              task_indices=None):
    """
    (2026-08-11 추가 - resume 파라미터) resume=True면 out_path에 이미 있는 결과(이전
    실행이 중간에 죽었거나 사용자가 일부러 끊은 경우)를 읽어서, task_id가 이미 있는 태스크는
    건너뛰고 파일에 이어서(append) 쓴다. rows(최종 요약에 쓰이는 리스트)에는 이전 실행
    결과도 합쳐 넣어서, 성공률 등 요약이 "이번 실행분만"이 아니라 "전체 누적"을 반영하게
    한다. resume=False(기본)면 예전과 동일하게 매번 out_path를 덮어쓰고 처음부터 전부
    돈다 - 하위 호환.

    task_indices: (2026-08-15 추가 - 특정 태스크만 재실행, --idx CLI 옵션용) tasks와 길이가
        같은 리스트를 주면, 결과 row의 "idx" 필드에 enumerate() 위치 대신 이 값을 쓴다.
        예: scroll/wait 뺑뺑이 오탐 버그를 고친 뒤, 원래 전체실행에서 그 사유로 blocked됐던
        태스크 몇 개(예: idx 4, 9, 28)만 골라 다시 돌리고 싶을 때 - tasks 자체는 3개짜리
        부분집합이라 plain enumerate면 idx가 0,1,2로 재번호 매겨지는데, 그러면 이전 전체실행
        결과와 대조가 안 된다. 원래 idx를 그대로 넘겨주면 결과 jsonl에도 4, 9, 28로 찍혀서
        "그 태스크가 이제 정상 처리되는지" 바로 비교 가능. None(기본)이면 예전처럼
        enumerate(tasks) 위치를 그대로 씀 - 하위호환.

        [2026-08-16 추가 - --idx + --resume 조합 버그 수정] task_indices가 주어졌다는 건
        "--idx로 이 태스크들은 무조건 다시 돌려라"라는 명시적 요청이다. 그런데 같은 out_path에
        --resume까지 같이 켜면(예: 기존 전체 결과 jsonl을 유지한 채로 특정 idx만 갱신하고
        싶은 경우), 아래 일반 resume 로직이 "out_path에 이미 그 task_id가 있다"는 이유만으로
        건너뛰어버려서 --idx가 아예 먹통이 되는 문제가 실측으로 확인됐다(--idx의 "무조건
        재실행" 의도와 --resume의 "이미 있으면 건너뛰기" 의도가 충돌하는데 지금까지는
        --resume이 이겼음). 그래서 task_indices에 해당하는 태스크는 --resume이 켜져 있어도
        절대 건너뛰지 않고, 그 태스크들의 예전 결과 줄은 파일/rows에서 버려서(교체) 같은
        task_id가 파일에 중복으로 남지 않게 한다. task_indices가 없는(순수 --resume만 쓰는)
        기존 흐름은 전혀 안 건드림 - 하위호환.
    """
    rows = []
    done_keys = set()
    file_exists = bool(out_path and os.path.exists(out_path))
    forced_keys = {_task_key(t) for t in tasks} if task_indices is not None else set()
    if resume and file_exists:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prev_row = json.loads(line)
                except json.JSONDecodeError:
                    # (2026-08-11) 직전 실행이 write 도중(예: flush 전 프로세스 강제 종료)
                    # 죽었으면 마지막 줄이 반쯤 쓰였을 수 있다 - 그런 손상된 줄은 무시하고
                    # 계속 진행(그 태스크는 done_keys에 안 들어가므로 이번 실행에서 다시 돈다 -
                    # 데이터 손실보다 중복 재실행 쪽이 안전한 폴백).
                    continue
                key = prev_row.get("task_id")
                if key in forced_keys:
                    # (2026-08-16 추가) --idx로 강제 재실행이 지정된 태스크의 예전 결과 줄은
                    # 버린다 - 밑에서 새로 돈 결과로 교체될 것이므로 done_keys에도 안 넣는다
                    # (건너뛰지 않고 반드시 다시 돌게).
                    continue
                rows.append(prev_row)
                if key:
                    done_keys.add(key)
        print(
            f"[run_batch] --resume: {out_path}에서 기존 결과 {len(rows)}개 발견 "
            f"(이미 끝난 태스크 {len(done_keys)}개는 건너뜀"
            + (f", --idx로 강제 재실행 지정된 {len(forced_keys)}개는 예전 결과를 버리고 재실행)" if forced_keys else ")")
        )

    # (2026-08-16 추가) forced_keys가 있는 재실행(--idx + --resume 조합)이면, 방금 위에서
    # 예전 파일 중 그 태스크들 줄만 뺀 나머지(rows)를 디스크에도 반영해야 한다 - 그냥
    # "a"(append)로 열면 디스크엔 예전 줄이 그대로 남은 채로 새 결과가 뒤에 또 붙어서 같은
    # task_id가 파일에 중복으로 남는다. 그래서 이 경우엔 "w"로 열어서 kept rows를 먼저
    # 다시 써두고(디스크를 rows와 일치시킴), 이후 새로 도는 태스크 결과는 같은 파일 핸들에
    # 이어서 쓴다. task_indices가 없거나(순수 --resume) forced_keys가 비어있으면(--idx가
    # 예전 결과와 안 겹침) 기존 동작(있으면 "a", 없으면 "w") 그대로.
    out_mode = "a" if (resume and file_exists and not forced_keys) else "w"
    out_f = open(out_path, out_mode, encoding="utf-8") if out_path else None
    if out_f and resume and file_exists and forced_keys:
        for r in rows:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_f.flush()
    try:
        for pos, task in enumerate(tasks):
            i = task_indices[pos] if task_indices is not None else pos
            task_id = _task_key(task)
            if resume and task_id in done_keys:
                continue
            t0 = time.time()
            try:
                traj = run_episode(
                    env, task, agent_step_fn, max_steps=max_steps, stuck_repeat_threshold=stuck_repeat_threshold,
                    stuck_repeat_window=stuck_repeat_window,
                )
            except Exception as e:  # noqa: BLE001 - (2026-08-15 추가) 렌더러 hang/드라이버 크래시 대응
                # 실측: Billboard처럼 광고/비디오가 무거운 사이트에서 CDP 명령이 "Timed out
                # receiving message from renderer" (selenium.TimeoutException)로 죽는 경우가
                # 확인됨 - env.execute_action()이 예외를 그대로 던지고, run_episode()도 이걸
                # 안 잡아서 이전엔 배치 전체(run_batch)가 죽었다. 여기서 잡아서 이 태스크만
                # blocked로 기록하고 다음 태스크로 넘어간다. 죽은 드라이버 세션을 그대로 재사용하면
                # 다음 태스크의 env.reset()도 같이 죽을 수 있으므로, driver를 강제로 정리하고
                # None으로 되돌려서 다음 reset()이 _make_driver()로 완전히 새 세션을 만들게 한다
                # (env_webvoyager.WebVoyagerEnv.reset()의 "driver=None -> 새로 생성" 경로 재사용).
                print(f"[run_batch] 태스크 {task_id!r} 실행 중 예외 발생(건너뛰고 계속 진행): {e}")
                try:
                    if getattr(env, "driver", None) is not None:
                        env.driver.quit()
                except Exception as quit_exc:  # noqa: BLE001 - 정리 실패해도 계속 진행
                    print(f"[run_batch] 죽은 드라이버 정리 중 추가 예외(무시): {quit_exc}")
                finally:
                    env.driver = None
                try:
                    task_url, task_instr, _ = env._parse_task(task)
                except Exception:  # noqa: BLE001 - task 포맷이 예상과 달라도 최소한 기록은 남김
                    task_url, task_instr = None, None
                traj = {
                    "instruction": task_instr,
                    "url": task_url,
                    "final_answer": None,
                    "n_steps": 0,
                    "hit_max_steps": False,
                    "blocked": True,
                    "blocked_reason": f"env/driver 예외로 태스크 중단: {e}",
                }

            if traj["blocked"]:
                # (2026-08-11 추가) CAPTCHA/bot-check로 막혔거나 반복-정체로 조기 종료된
                # 태스크는 judge에게 물어볼 이유가 없다 - 성공일 리 없고(화면이 캡차/멈춘
                # 상태라 judge를 혼란스럽게 할 뿐이며), judge 호출(모델/API) 비용도 아낄 수
                # 있다. success는 명시적으로 False, votes/agreement는 "판단 자체를 안 했다"는
                # 뜻으로 빈 값/None으로 남겨서 "agent가 추론에 실패해서 실패"와 "애초에 막혀서
                # 실패"를 결과에서 구분할 수 있게 한다.
                judge_result = {"success": False, "votes": [], "agreement": None, "runs": []}
            else:
                judge_result = run_judge_with_repeats(
                    judge_fn, traj["instruction"], traj["screenshots"], traj["final_answer"],
                    n_repeats=judge_repeats, max_workers=judge_max_workers,
                )
            row = {
                "idx": i,
                "task_id": task_id,
                "instruction": traj["instruction"],
                "url": traj["url"],
                "n_steps": traj["n_steps"],
                "hit_max_steps": traj["hit_max_steps"],
                "blocked": traj["blocked"],
                "blocked_reason": traj["blocked_reason"],
                "final_answer": traj["final_answer"],
                "success": judge_result["success"],
                "judge_agreement": judge_result["agreement"],
                "judge_votes": judge_result["votes"],
                "elapsed_sec": round(time.time() - t0, 2),
            }
            rows.append(row)
            status_tag = "BLOCKED" if row["blocked"] else ("O" if row["success"] else "X")
            agreement_str = f"{row['judge_agreement']:.2f}" if row["judge_agreement"] is not None else "n/a"
            print(
                f"[{pos + 1}/{len(tasks)}] (idx={i}) {status_tag} "
                f"steps={row['n_steps']} agreement={agreement_str} "
                f"instr={row['instruction'][:50]!r}"
            )
            if out_f:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if out_f:
            out_f.close()

    n = len(rows)
    n_blocked = sum(1 for r in rows if r["blocked"])
    n_success = sum(1 for r in rows if r["success"])
    success_rate = n_success / n if n else 0.0
    print(
        f"\n성공률: {success_rate:.3f} ({n_success}/{n}) | "
        f"bot-check/정체로 막힌 태스크: {n_blocked}/{n}"
        + (f" (제외하면 {n_success}/{n - n_blocked} = {n_success / (n - n_blocked):.3f})" if n_blocked and n_blocked < n else "")
    )
    return rows, success_rate


# ---------------------------------------------------------------------------
# mock 기반 단위 테스트 (실제 브라우저/모델/API 없이 제어 흐름만 검증)
# ---------------------------------------------------------------------------
def _run_mock_selftest():
    """
    `python gui_actor_eval_webvoyager.py --selftest`
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
    fake_env.detect_bot_check.return_value = None  # bot-check 없음(정상) - MagicMock 기본값(자동
    # 생성된 truthy MagicMock)을 그대로 두면 매 스텝 "감지됨"으로 오판되니 명시적으로 None 지정.
    traj = run_episode(fake_env, {"web": "http://x", "ques": "do X"}, dummy_agent_step, max_steps=5)
    check("즉시 terminate -> n_steps=1", traj["n_steps"] == 1)
    check("즉시 terminate -> hit_max_steps=False", traj["hit_max_steps"] is False)
    check("즉시 terminate -> execute_action 안 불림", not fake_env.execute_action.called)
    check("즉시 terminate -> blocked 아님", traj["blocked"] is False)

    # --- run_episode: max_steps까지 계속 진행하는 agent (bot-check 없음, 반복도 없음) ---
    def never_stop_agent(screenshot, task_info, history):
        # (2026-08-11 수정) 매번 다른 액션(스텝 번호로 구분)을 내야 한다 - 안 그러면 새로 추가된
        # "같은 액션 연속 반복 -> stuck" 감지에 걸려서 이 테스트의 원래 의도(max_steps 소진까지
        # 정상 진행되는지 확인)와 다른 경로를 타게 된다.
        n = len(history["actions"])
        return {"action": "wait", "time": 0.0} if n % 2 == 0 else {"action": "scroll", "text": "down"}

    fake_env2 = MagicMock()
    fake_env2.reset.return_value = (fake_img, {"instruction": "do Y", "url": "http://y"})
    fake_env2.execute_action.return_value = (fake_img, None, False, False, {"instruction": "do Y", "url": "http://y"})
    fake_env2.detect_bot_check.return_value = None
    traj2 = run_episode(fake_env2, {"web": "http://y", "ques": "do Y"}, never_stop_agent, max_steps=4)
    check("계속 진행 -> max_steps만큼 실행", traj2["n_steps"] == 4)
    check("계속 진행 -> hit_max_steps=True", traj2["hit_max_steps"] is True)
    check("계속 진행 -> screenshots 개수 = n_steps+1(초기 포함)", len(traj2["screenshots"]) == 5)
    check("계속 진행 -> blocked 아님", traj2["blocked"] is False)

    # --- (2026-08-11 추가) run_episode가 agent_step_fn.reset_episode()를 매 에피소드마다 호출하는지 ---
    fake_env_reset_hook = MagicMock()
    fake_env_reset_hook.reset.return_value = (fake_img, {"instruction": "do RH", "url": "http://rh"})
    fake_env_reset_hook.detect_bot_check.return_value = None
    agent_with_reset_hook = MagicMock(return_value={"action": "terminate", "status": "success"})
    agent_with_reset_hook.reset_episode = MagicMock()
    run_episode(fake_env_reset_hook, {"web": "http://rh", "ques": "do RH"}, agent_with_reset_hook, max_steps=5)
    check(
        "run_episode -> agent_step_fn.reset_episode()가 에피소드 시작 시점에 호출됨",
        agent_with_reset_hook.reset_episode.called,
    )

    # reset_episode 훅이 없는(dummy_agent_step 같은) 함수는 에러 없이 그냥 넘어가야 함
    fake_env_no_hook = MagicMock()
    fake_env_no_hook.reset.return_value = (fake_img, {"instruction": "do NH", "url": "http://nh"})
    fake_env_no_hook.detect_bot_check.return_value = None
    try:
        run_episode(fake_env_no_hook, {"web": "http://nh", "ques": "do NH"}, dummy_agent_step, max_steps=1)
        check("run_episode -> reset_episode 훅 없어도 에러 없음", True)
    except Exception:
        check("run_episode -> reset_episode 훅 없어도 에러 없음", False)

    # --- (2026-08-11 추가) run_episode: reset 시점부터 bot-check 감지된 경우 ---
    fake_env_blocked_at_reset = MagicMock()
    fake_env_blocked_at_reset.reset.return_value = (
        fake_img,
        {"instruction": "do W", "url": "http://w", "_bot_check_at_reset": {"reason": "title contains 'captcha'"}},
    )
    traj_blocked_reset = run_episode(
        fake_env_blocked_at_reset, {"web": "http://w", "ques": "do W"}, dummy_agent_step, max_steps=5,
    )
    check("reset 시점 bot-check -> blocked=True", traj_blocked_reset["blocked"] is True)
    check("reset 시점 bot-check -> reason에 근거가 담김", "captcha" in traj_blocked_reset["blocked_reason"])
    check("reset 시점 bot-check -> 액션을 하나도 안 냄(n_steps=0)", traj_blocked_reset["n_steps"] == 0)
    check(
        "reset 시점 bot-check -> agent_step_fn/execute_action 둘 다 안 불림(첫 스텝도 낭비 안 함)",
        not fake_env_blocked_at_reset.execute_action.called,
    )

    # --- (2026-08-11 추가) run_episode: 스텝 진행 중 bot-check가 감지되는 경우 ---
    fake_env_blocked_mid = MagicMock()
    fake_env_blocked_mid.reset.return_value = (fake_img, {"instruction": "do V", "url": "http://v"})
    fake_env_blocked_mid.execute_action.return_value = (fake_img, None, False, False, {"instruction": "do V", "url": "http://v"})
    # 2번째 스텝에서 처음 감지되도록: 1번째 호출은 None, 그 이후는 감지됨.
    fake_env_blocked_mid.detect_bot_check.side_effect = [None, {"reason": "url contains 'recaptcha'"}]

    def alternating_agent(screenshot, task_info, history):
        n = len(history["actions"])
        return {"action": "wait", "time": 0.0} if n % 2 == 0 else {"action": "scroll", "text": "down"}

    traj_blocked_mid = run_episode(fake_env_blocked_mid, {"web": "http://v", "ques": "do V"}, alternating_agent, max_steps=10)
    check("스텝 중 bot-check -> blocked=True", traj_blocked_mid["blocked"] is True)
    check("스텝 중 bot-check -> reason에 근거가 담김", "recaptcha" in traj_blocked_mid["blocked_reason"])
    check("스텝 중 bot-check -> 2스텝만에 조기 종료(더 안 돎)", traj_blocked_mid["n_steps"] == 2)

    # --- (2026-08-11 추가 - 수동 CAPTCHA 통과) run_episode: wait_for_manual_captcha()가
    # True를 반환하면(사람이 직접 풀었다는 뜻) blocked 처리 없이 계속 진행돼야 함 ---
    fake_env_manual_resolved = MagicMock()
    fake_env_manual_resolved.reset.return_value = (fake_img, {"instruction": "do W", "url": "http://w"})
    fake_env_manual_resolved.execute_action.return_value = (fake_img, None, False, False, {"instruction": "do W", "url": "http://w"})
    # 1번째 스텝에서 bot-check 뜨지만 wait_for_manual_captcha()가 True -> 안 막히고 계속 진행.
    # detect_bot_check는 그 뒤로도 계속 뭔가 감지된 것처럼 두되(True 처리 자체는
    # wait_for_manual_captcha 쪽 책임이라 run_episode는 그 반환값만 믿음), stuck 임계값에
    # 걸리지 않도록 매 스텝 다른 액션을 내는 alternating_agent를 재사용.
    fake_env_manual_resolved.detect_bot_check.return_value = {"reason": "title contains 'captcha'"}
    fake_env_manual_resolved.wait_for_manual_captcha.return_value = True

    traj_manual_resolved = run_episode(
        fake_env_manual_resolved, {"web": "http://w", "ques": "do W"}, alternating_agent, max_steps=3,
    )
    check("wait_for_manual_captcha=True -> blocked 안 됨(사람이 풀었으므로 계속 진행)", traj_manual_resolved["blocked"] is False)
    check("wait_for_manual_captcha=True -> max_steps까지 정상 진행", traj_manual_resolved["n_steps"] == 3)
    check(
        "wait_for_manual_captcha -> bot-check 감지될 때마다 호출됨",
        fake_env_manual_resolved.wait_for_manual_captcha.call_count == 3,
    )

    # --- (2026-08-11 추가) run_episode: 같은 액션이 연속 반복되면(bot-check 신호 없이도) stuck으로 조기 종료 ---
    fake_env_stuck = MagicMock()
    fake_env_stuck.reset.return_value = (fake_img, {"instruction": "do U", "url": "http://u"})
    fake_env_stuck.execute_action.return_value = (fake_img, None, False, False, {"instruction": "do U", "url": "http://u"})
    fake_env_stuck.detect_bot_check.return_value = None  # bot-check는 계속 정상으로 보임 - 그래도 반복이면 잡혀야 함

    def repeating_agent(screenshot, task_info, history):
        return {"action": "left_click", "coordinate": [123, 456]}  # 항상 같은 좌표를 반복 클릭

    traj_stuck = run_episode(
        fake_env_stuck, {"web": "http://u", "ques": "do U"}, repeating_agent,
        max_steps=10, stuck_repeat_threshold=3,
    )
    check("반복 액션 -> blocked=True", traj_stuck["blocked"] is True)
    check("반복 액션 -> reason에 '반복' 언급", "반복" in traj_stuck["blocked_reason"])
    check("반복 액션 -> stuck_repeat_threshold(3)만큼만 실행하고 종료", traj_stuck["n_steps"] == 3)

    # 좌표가 10px 이내로만 흔들리는 경우도 "같은 시도"로 취급되는지(완전히 똑같은 좌표만 잡으면
    # RegionFocus의 미세한 흔들림에 stuck 감지가 무력화됨)
    import itertools

    # 121.10px 단위 버킷 경계를 넘지 않는 범위(120±2)에서만 흔들리게 해서 "10px 단위로는
    # 같은 버킷"이라는 전제를 확실히 만족시킨다 - 버킷 경계에 걸치는 값(예: 118~122처럼
    # round(11.x)/round(12.x) 경계를 넘나드는 값)을 쓰면 반올림 버킷이 갈려서 "같은 시도"로
    # 안 잡히는 것 자체는 _action_fingerprint()의 정상 동작(완전한 jitter-불변은 아님, 10px
    # 버킷 내부에서만 흡수)이라 테스트 의도(작은 흔들림은 흡수됨)에 맞게 흔들림 폭을 좁혔다.
    jitter = itertools.cycle([0, 2, -2, 1])

    def jittering_agent(screenshot, task_info, history):
        return {"action": "left_click", "coordinate": [120 + next(jitter), 456]}

    traj_jitter = run_episode(
        fake_env_stuck, {"web": "http://u", "ques": "do U"}, jittering_agent,
        max_steps=10, stuck_repeat_threshold=3,
    )
    check("10px 이내로 흔들리는 좌표도 '같은 시도'로 잡혀서 stuck 처리됨", traj_jitter["blocked"] is True)

    # --- (2026-08-11 추가 - 뺑뺑이/오실레이션 탐지) 두 액션을 번갈아 반복해도(연속 아님) 잡혀야 함 ---
    # 예전 "연속 반복" 방식은 이 패턴(A -> B -> A -> B...)을 절대 못 잡았음(매 스텝 직전과
    # 다르므로 연속 카운트가 계속 리셋됨) - 검색 버튼을 못 찾고 이것저것 번갈아 시도하는
    # 실제 관찰 사례를 재현.
    def oscillating_agent(screenshot, task_info, history):
        n = len(history["actions"])
        return (
            {"action": "left_click", "coordinate": [100, 200]}
            if n % 2 == 0
            else {"action": "left_click", "coordinate": [300, 400]}
        )

    traj_osc = run_episode(
        fake_env_stuck, {"web": "http://u", "ques": "do U"}, oscillating_agent,
        max_steps=10, stuck_repeat_threshold=3,
    )
    check("A-B 번갈아 반복(뺑뺑이) -> blocked=True(예전엔 절대 못 잡던 케이스)", traj_osc["blocked"] is True)
    check("뺑뺑이 -> reason에 '반복' 언급", "반복" in traj_osc["blocked_reason"])
    # threshold=3 -> A(짝수 스텝마다)가 세 번째로 나오는 5번째 스텝(A,B,A,B,A)에서 바로 종료
    check("뺑뺑이 -> A가 3번째 등장하는 5스텝만에 종료", traj_osc["n_steps"] == 5)

    # stuck_repeat_window를 명시적으로 좁히면(예: 2) 애초에 직전 액션 1개랑만 비교하는 셈이라
    # 서로 다른 A/B가 매번 갈아치워져서 threshold(3)에 절대 못 도달 -> 안 잡힘(max_steps까지 감)
    traj_osc_narrow_window = run_episode(
        fake_env_stuck, {"web": "http://u", "ques": "do U"}, oscillating_agent,
        max_steps=6, stuck_repeat_threshold=3, stuck_repeat_window=2,
    )
    check(
        "stuck_repeat_window을 좁히면 뺑뺑이를 못 잡음(윈도우가 threshold보다 작아서)",
        traj_osc_narrow_window["blocked"] is False,
    )

    # --- (2026-08-15 추가 - 사용자 요청) scroll은 stuck-repeat(뺑뺑이) 조기종료 판단에서 아예
    # 빠져야 함. "개수가 명시된 태스크는 목표를 채울 때까지 계속 스크롤/탐색하라"는 최근 프롬프트
    # 지침과, 반복된 스크롤을 뺑뺑이로 보고 하드 조기종료시키는 이 안전장치가 서로 충돌하던
    # 문제(Booking 배치 실측: 정상적으로 같은 방향 스크롤을 몇 번 했을 뿐인데 조기종료 위험에
    # 노출) 재발 방지. ---
    fake_env_scroll_only = MagicMock()
    fake_env_scroll_only.reset.return_value = (fake_img, {"instruction": "do S", "url": "http://s"})
    fake_env_scroll_only.execute_action.return_value = (fake_img, None, False, False, {"instruction": "do S", "url": "http://s"})
    fake_env_scroll_only.detect_bot_check.return_value = None

    def scroll_only_agent(screenshot, task_info, history):
        return {"action": "scroll", "text": "down"}  # 매번 완전히 똑같은 스크롤 액션만 반복

    traj_scroll_only = run_episode(
        fake_env_scroll_only, {"web": "http://s", "ques": "do S"}, scroll_only_agent,
        max_steps=10, stuck_repeat_threshold=3,
    )
    check(
        "같은 방향 scroll만 계속 반복해도 stuck으로 안 잡힘(max_steps까지 정상 진행)",
        traj_scroll_only["blocked"] is False and traj_scroll_only["n_steps"] == 10,
    )

    # scroll이 다른 액션과 함께 나올 땐, scroll 자체는 안 세도 그 "다른 액션"의 반복은 여전히 잡혀야 함
    def scroll_plus_repeating_click(screenshot, task_info, history):
        n = len(history["actions"])
        return {"action": "scroll", "text": "down"} if n % 2 == 0 else {"action": "left_click", "coordinate": [50, 50]}

    traj_scroll_plus_click = run_episode(
        fake_env_scroll_only, {"web": "http://s", "ques": "do S"}, scroll_plus_repeating_click,
        max_steps=10, stuck_repeat_threshold=3,
    )
    check(
        "scroll을 껴서 반복해도, 사이사이의 같은 click 반복은 여전히 stuck으로 잡힘",
        traj_scroll_plus_click["blocked"] is True,
    )

    # (2026-08-15 추가 - 사용자 요청) wait도 scroll과 같은 이유로 뺑뺑이 판단에서 제외되는지
    # 검증 - "최근 윈도우 안 빈도" 방식이라 연속 반복이 아니어도(다른 정상 액션들 사이사이에
    # 한 번씩만 껴도) 오탐될 수 있었던 문제 재발 방지.
    def wait_only_agent(screenshot, task_info, history):
        return {"action": "wait"}

    # (2026-08-16 수정 - 회귀 수정) fingerprint 윈도우-빈도 카운트에서 wait을 뺀 것과는 별개로,
    # consecutive_wait 카운터가 순수 연속 wait 스팸을 잡는다(WolframAlpha 실측 회귀 수정) - 그래서
    # 이제 "wait만 계속"은 stuck_repeat_threshold(3)에 도달하는 3스텝만에 blocked=True여야 정상.
    traj_wait_only = run_episode(
        fake_env_scroll_only, {"web": "http://s", "ques": "do S"}, wait_only_agent,
        max_steps=10, stuck_repeat_threshold=3,
    )
    check(
        "wait만 계속 반복하면 연속 스팸으로 잡힘(회귀 수정: 예전엔 못 잡았음)",
        traj_wait_only["blocked"] is True and traj_wait_only["n_steps"] == 3,
    )
    check("wait 연속 스팸 -> reason에 'wait' 언급", "wait" in traj_wait_only["blocked_reason"])

    def wait_plus_repeating_click(screenshot, task_info, history):
        n = len(history["actions"])
        return {"action": "wait"} if n % 2 == 0 else {"action": "left_click", "coordinate": [50, 50]}

    traj_wait_plus_click = run_episode(
        fake_env_scroll_only, {"web": "http://s", "ques": "do S"}, wait_plus_repeating_click,
        max_steps=10, stuck_repeat_threshold=3,
    )
    check(
        "wait을 껴서 반복해도, 사이사이의 같은 click 반복은 여전히 stuck으로 잡힘",
        traj_wait_plus_click["blocked"] is True,
    )

    def wait_interspersed_agent(screenshot, task_info, history):
        # 서로 다른 실제 click 사이사이에 wait이 한 번씩만 낀다 - consecutive_wait은 매번
        # 0으로 리셋되므로 안 걸려야 한다(원래 이번 세션 wait-제외 수정이 의도했던 케이스:
        # 산발적 wait은 오탐하면 안 됨). click 좌표는 서로 멀리 떨어뜨려 fingerprint가
        # 우연히 겹치지 않게 한다.
        n = len(history["actions"])
        if n % 2 == 0:
            return {"action": "wait"}
        return {"action": "left_click", "coordinate": [50 + n * 50, 50]}

    traj_wait_interspersed = run_episode(
        fake_env_scroll_only, {"web": "http://s", "ques": "do S"}, wait_interspersed_agent,
        max_steps=10, stuck_repeat_threshold=3,
    )
    check(
        "서로 다른 click 사이사이에 wait이 한 번씩만 껴도 stuck으로 안 잡힘(산발적 wait은 여전히 제외)",
        traj_wait_interspersed["blocked"] is False and traj_wait_interspersed["n_steps"] == 10,
    )

    # --- env가 detect_bot_check()를 아예 제공하지 않는 경우(구버전 env) -> 에러 없이 정상 동작 ---
    class _EnvWithoutBotCheck:
        def __init__(self):
            self.reset_called_with = None

        def reset(self, task):
            return fake_img, {"instruction": "do T", "url": "http://t"}

        def execute_action(self, action):
            return fake_img, None, False, False, {"instruction": "do T", "url": "http://t"}

    traj_no_detect = run_episode(
        _EnvWithoutBotCheck(), {"web": "http://t", "ques": "do T"}, never_stop_agent, max_steps=2,
    )
    check("detect_bot_check 없는 env -> 에러 없이 정상 동작(duck-typing)", traj_no_detect["n_steps"] == 2)
    check("detect_bot_check 없는 env -> blocked=False", traj_no_detect["blocked"] is False)

    # --- run_judge_with_repeats: 다수결 ---
    seq = iter([True, False, True])  # 2:1 -> True로 다수결
    judge_fn = lambda instruction, screenshots, final_answer: {"success": next(seq), "raw_response": "r"}
    result = run_judge_with_repeats(judge_fn, "do X", [fake_img], None, n_repeats=3)
    check("다수결 2:1 -> success=True", result["success"] is True)
    check("agreement = 2/3", abs(result["agreement"] - 2 / 3) < 1e-6)
    check("votes 개수=3", len(result["votes"]) == 3)

    # --- (2026-08-11 추가 - 태스크 간 지연 단축) run_judge_with_repeats: max_workers 병렬 실행 ---
    import threading

    concurrent_calls = {"active": 0, "max_seen": 0, "total": 0}
    lock = threading.Lock()

    def _slow_judge(instruction, screenshots, final_answer):
        with lock:
            concurrent_calls["active"] += 1
            concurrent_calls["max_seen"] = max(concurrent_calls["max_seen"], concurrent_calls["active"])
            concurrent_calls["total"] += 1
        time.sleep(0.05)  # 실제 네트워크 호출을 흉내(짧게) - 이게 있어야 동시 실행이 겹치는 걸 관측 가능
        with lock:
            concurrent_calls["active"] -= 1
        return {"success": True, "raw_response": "ok"}

    result_parallel = run_judge_with_repeats(_slow_judge, "do X", [fake_img], None, n_repeats=3, max_workers=3)
    check("max_workers=3 -> 호출은 여전히 3번", concurrent_calls["total"] == 3)
    check("max_workers=3 -> 실제로 동시에 여러 개가 겹쳐서 실행됨(순차였다면 1)", concurrent_calls["max_seen"] > 1)
    check("max_workers=3 -> 결과 개수/다수결도 정상", len(result_parallel["votes"]) == 3 and result_parallel["success"] is True)

    # max_workers=1(기본)이면 예전처럼 완전히 순차 - 동시에 겹치는 일이 없어야 함
    concurrent_calls2 = {"active": 0, "max_seen": 0, "total": 0}

    def _slow_judge2(instruction, screenshots, final_answer):
        concurrent_calls2["active"] += 1
        concurrent_calls2["max_seen"] = max(concurrent_calls2["max_seen"], concurrent_calls2["active"])
        concurrent_calls2["total"] += 1
        concurrent_calls2["active"] -= 1
        return {"success": True, "raw_response": "ok"}

    run_judge_with_repeats(_slow_judge2, "do X", [fake_img], None, n_repeats=3, max_workers=1)
    check("max_workers=1(기본) -> 순차 실행(하위 호환, 겹치는 호출 없음)", concurrent_calls2["max_seen"] == 1)

    # --- (2026-08-11 추가) make_openai_judge -> 429 rate limit 자동 재시도(api_planner._call_with_retry 재사용) ---
    import sys as _sys_for_judge_retry
    import types as _types_for_judge_retry

    fake_judge_response = MagicMock()
    fake_judge_response.choices = [MagicMock(message=MagicMock(content='{"reason": "looks done", "success": true}'))]
    fake_judge_client = MagicMock()
    fake_judge_client.chat.completions.create.side_effect = [
        Exception("Error code: 429 - rate_limit_exceeded"),
        fake_judge_response,
    ]
    fake_openai_module_judge = _types_for_judge_retry.ModuleType("openai")
    fake_openai_module_judge.OpenAI = MagicMock(return_value=fake_judge_client)
    _sys_for_judge_retry.modules["openai"] = fake_openai_module_judge
    orig_sleep_judge = time.sleep
    time.sleep = lambda *a, **k: None
    try:
        judge_fn_retry = make_openai_judge(model="gpt-4o-test", retry_base_delay=0.01)
        judge_result_retry = judge_fn_retry("do X", [fake_img], None)
        check(
            "make_openai_judge -> 429 한 번은 자동 재시도로 흡수하고 정상 판정 반환",
            judge_result_retry["success"] is True,
        )
        check(
            "make_openai_judge -> create()가 재시도 포함 2번 호출됨",
            fake_judge_client.chat.completions.create.call_count == 2,
        )
    finally:
        time.sleep = orig_sleep_judge
        del _sys_for_judge_retry.modules["openai"]

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
    fake_env3.detect_bot_check.return_value = None
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
        check("run_batch -> blocked 필드가 각 row에 있고 전부 False", all(r["blocked"] is False for r in rows))
        with open(out_path, encoding="utf-8") as f:
            saved = [json.loads(line) for line in f]
        check("run_batch -> jsonl 저장 개수 일치", len(saved) == 3)

    # --- (2026-08-15 추가 - --idx CLI 옵션용) run_batch: task_indices를 주면 결과 row의 "idx"가
    # enumerate() 위치가 아니라 원래 인덱스로 찍혀야 함 - 원래 전체실행 결과와 대조 가능하게. ---
    fake_env_idx = MagicMock()
    fake_env_idx.reset.return_value = (fake_img, {"instruction": "do IDX", "url": "http://idx"})
    fake_env_idx.detect_bot_check.return_value = None
    with tempfile.TemporaryDirectory() as d:
        out_path_idx = os.path.join(d, "out_idx.jsonl")
        rows_idx, _ = run_batch(
            [{"web": "http://idx", "ques": "do IDX"}] * 3,  # 원래 idx 4, 9, 28인 태스크 3개를 골라온 상황을 흉내
            fake_env_idx, dummy_agent_step, always_success_judge,
            judge_repeats=1, out_path=out_path_idx, task_indices=[4, 9, 28],
        )
        check(
            "run_batch -> task_indices 지정시 row의 idx가 원래 인덱스로 찍힘(재번호 안 매김)",
            [r["idx"] for r in rows_idx] == [4, 9, 28],
        )
        with open(out_path_idx, encoding="utf-8") as f:
            saved_idx = [json.loads(line) for line in f]
        check("run_batch -> task_indices -> jsonl에도 원래 idx가 그대로 저장됨", [r["idx"] for r in saved_idx] == [4, 9, 28])

    # task_indices 미지정(기본)이면 예전처럼 enumerate() 위치를 그대로 씀 - 하위호환
    with tempfile.TemporaryDirectory() as d:
        rows_default_idx, _ = run_batch(
            [{"web": "http://idx", "ques": "do IDX"}] * 3,
            fake_env_idx, dummy_agent_step, always_success_judge,
            judge_repeats=1, out_path=os.path.join(d, "out.jsonl"),
        )
        check(
            "run_batch -> task_indices 미지정시 예전처럼 0,1,2로 매겨짐(하위호환)",
            [r["idx"] for r in rows_default_idx] == [0, 1, 2],
        )

    # --- (2026-08-11 추가) run_batch: blocked된 태스크는 judge를 아예 안 부르고 success=False로 기록 ---
    fake_env_blocked_batch = MagicMock()
    fake_env_blocked_batch.reset.return_value = (
        fake_img,
        {"instruction": "do BLOCKED", "url": "http://blocked", "_bot_check_at_reset": {"reason": "title contains 'captcha'"}},
    )
    judge_call_count = {"n": 0}

    def _counting_judge(instruction, screenshots, final_answer):
        judge_call_count["n"] += 1
        return {"success": True, "raw_response": "should not be called"}

    with tempfile.TemporaryDirectory() as d:
        out_path2 = os.path.join(d, "out2.jsonl")
        rows_blocked, rate_blocked = run_batch(
            [{"web": "http://blocked", "ques": "do BLOCKED"}] * 2,
            fake_env_blocked_batch, dummy_agent_step, _counting_judge,
            judge_repeats=3, out_path=out_path2,
        )
        check("run_batch -> blocked 태스크는 judge_fn을 한 번도 안 부름(비용 절약)", judge_call_count["n"] == 0)
        check("run_batch -> blocked 태스크의 success는 False", all(r["success"] is False for r in rows_blocked))
        check("run_batch -> blocked 태스크의 blocked=True + blocked_reason 존재", all(r["blocked"] is True and r["blocked_reason"] for r in rows_blocked))
        check("run_batch -> blocked 태스크는 judge_votes가 빈 리스트(판단 자체를 안 했다는 표시)", all(r["judge_votes"] == [] for r in rows_blocked))
        check("run_batch -> 전체 success_rate=0.0", rate_blocked == 0.0)

    # --- (2026-08-15 추가 - 렌더러 hang/드라이버 크래시 대응) run_batch: run_episode()가 예외를
    # 던져도 배치 전체가 죽지 않고, 해당 태스크만 blocked로 기록한 뒤 다음 태스크로 계속 진행 ---
    fake_env_crash = MagicMock()
    fake_env_crash.detect_bot_check.return_value = None
    fake_env_crash.reset.side_effect = [
        (fake_img, {"instruction": "task A", "url": "http://a"}),
        TimeoutError("Timed out receiving message from renderer: 20.000"),
        (fake_img, {"instruction": "task C", "url": "http://c"}),
    ]
    fake_env_crash._parse_task.side_effect = lambda t: (t["web"], t["ques"], {})
    orig_crash_driver = fake_env_crash.driver  # env.driver=None으로 리셋되기 전에 미리 참조 저장
    crash_judge_calls = {"n": 0}

    def _crash_test_judge(instruction, screenshots, final_answer):
        crash_judge_calls["n"] += 1
        return {"success": True, "raw_response": "ok"}

    with tempfile.TemporaryDirectory() as d:
        out_path_crash = os.path.join(d, "out_crash.jsonl")
        rows_crash, _ = run_batch(
            [
                {"id": "T--A", "web": "http://a", "ques": "task A"},
                {"id": "T--B", "web": "http://b", "ques": "task B"},
                {"id": "T--C", "web": "http://c", "ques": "task C"},
            ],
            fake_env_crash, dummy_agent_step, _crash_test_judge,
            judge_repeats=1, out_path=out_path_crash,
        )
        check("run_batch -> 중간 태스크가 예외로 죽어도 3개 row 전부 기록됨(배치가 안 죽음)", len(rows_crash) == 3)
        check(
            "run_batch -> 예외난 태스크만 blocked=True, 나머지는 blocked=False",
            [r["blocked"] for r in rows_crash] == [False, True, False],
        )
        check(
            "run_batch -> blocked_reason에 원본 예외 메시지가 남음",
            "Timed out receiving message from renderer" in (rows_crash[1]["blocked_reason"] or ""),
        )
        check("run_batch -> 예외난 태스크는 success=False", rows_crash[1]["success"] is False)
        check(
            "run_batch -> 정상 태스크 2개는 judge가 호출됨(예외난 태스크는 호출 안 됨)",
            crash_judge_calls["n"] == 2,
        )
        check("run_batch -> 예외 처리 중 죽은 driver.quit()이 호출됨", orig_crash_driver.quit.called)
        check("run_batch -> 예외 처리 후 driver가 None으로 리셋됨(다음 reset()이 새 세션 만들도록)", fake_env_crash.driver is None)

    # --- (2026-08-11 추가) _task_key ---
    check("_task_key -> id 필드가 있으면 그걸 그대로 씀", _task_key({"id": "Amazon--16", "web": "http://a", "ques": "q"}) == "Amazon--16")
    check(
        "_task_key -> id 없으면 url+instruction 조합으로 폴백",
        _task_key({"web": "http://a", "ques": "q"}) == "http://a|q",
    )
    check("_task_key -> 튜플 태스크도 지원", _task_key(("http://b", "do B")) == "http://b|do B")

    # --- (2026-08-11 추가) run_batch: --resume - 이미 끝난 task_id는 건너뛰고 이어서 씀 ---
    fake_env_resume = MagicMock()
    fake_env_resume.reset.return_value = (fake_img, {"instruction": "resumed", "url": "http://r"})
    fake_env_resume.detect_bot_check.return_value = None
    resume_agent_calls = {"n": 0}

    def _counting_agent(screenshot, task_info, history):
        resume_agent_calls["n"] += 1
        return {"action": "terminate", "status": "success"}

    with tempfile.TemporaryDirectory() as d:
        out_path3 = os.path.join(d, "out3.jsonl")
        # 직전 실행이 T1/T2까지 끝내고 죽었다고 가정 - 결과 파일을 직접 만들어둠.
        with open(out_path3, "w", encoding="utf-8") as f:
            f.write(json.dumps({"task_id": "T1", "success": True, "blocked": False}) + "\n")
            f.write(json.dumps({"task_id": "T2", "success": False, "blocked": False}) + "\n")

        tasks_resume = [
            {"id": "T1", "web": "http://r", "ques": "task 1"},
            {"id": "T2", "web": "http://r", "ques": "task 2"},
            {"id": "T3", "web": "http://r", "ques": "task 3"},
        ]
        rows_resume, rate_resume = run_batch(
            tasks_resume, fake_env_resume, _counting_agent, always_success_judge,
            judge_repeats=1, out_path=out_path3, resume=True,
        )
        check("--resume -> 이미 끝난 T1/T2는 다시 안 돎(agent_step_fn 1번만 호출)", resume_agent_calls["n"] == 1)
        check("--resume -> rows에 이전 2개 + 새로 돈 1개 = 3개", len(rows_resume) == 3)
        check(
            "--resume -> 새로 돈 태스크는 T3(안 끝난 것만)",
            any(r.get("task_id") == "T3" for r in rows_resume),
        )
        with open(out_path3, encoding="utf-8") as f:
            saved_resume = [json.loads(line) for line in f]
        check("--resume -> 파일에도 기존 2줄 + 새 1줄 = 3줄로 append됨(덮어쓰기 아님)", len(saved_resume) == 3)
        check(
            "--resume -> 전체 success_rate가 이전 결과까지 합쳐서 계산됨(1승1패 + 새로 1승 = 2/3)",
            abs(rate_resume - 2 / 3) < 1e-6,
        )

    # resume=False(기본)면 기존 파일이 있어도 그냥 덮어쓰고 처음부터 다 돎 - 하위 호환 확인
    resume_agent_calls["n"] = 0
    with tempfile.TemporaryDirectory() as d:
        out_path4 = os.path.join(d, "out4.jsonl")
        with open(out_path4, "w", encoding="utf-8") as f:
            f.write(json.dumps({"task_id": "T1", "success": True, "blocked": False}) + "\n")
        rows_no_resume, _ = run_batch(
            [{"id": "T1", "web": "http://r", "ques": "task 1"}],
            fake_env_resume, _counting_agent, always_success_judge,
            judge_repeats=1, out_path=out_path4, resume=False,
        )
        check("resume=False(기본) -> 이미 있던 결과 무시하고 다시 돎(하위 호환)", resume_agent_calls["n"] == 1)
        check("resume=False -> rows도 이번에 새로 돈 1개뿐(파일 덮어씀)", len(rows_no_resume) == 1)

    # --- (2026-08-16 추가 - --idx + --resume 조합 버그 수정) task_indices로 지정된 태스크는
    # --resume이 켜져 있어도 "이미 끝났다"고 건너뛰지 않고 무조건 재실행되어야 하고, 예전
    # 결과 줄은 새 결과로 교체되어야 함(중복으로 남으면 안 됨).
    resume_agent_calls["n"] = 0
    with tempfile.TemporaryDirectory() as d:
        out_path5 = os.path.join(d, "out5.jsonl")
        # 직전 "전체 실행"이 T1(실패)/T2(성공) 둘 다 이미 끝낸 상태라고 가정.
        with open(out_path5, "w", encoding="utf-8") as f:
            f.write(json.dumps({"idx": 0, "task_id": "T1", "success": False, "blocked": True}) + "\n")
            f.write(json.dumps({"idx": 1, "task_id": "T2", "success": True, "blocked": False}) + "\n")

        # --idx 0 으로 T1만 재실행하는 상황 재현 - tasks는 T1 하나뿐인 부분집합, task_indices=[0]
        rows_forced, rate_forced = run_batch(
            [{"id": "T1", "web": "http://r", "ques": "task 1"}],
            fake_env_resume, _counting_agent, always_success_judge,
            judge_repeats=1, out_path=out_path5, resume=True, task_indices=[0],
        )
        check(
            "--idx + --resume -> 이미 out_path에 있어도 task_indices로 지정된 태스크는 건너뛰지 않고 재실행됨",
            resume_agent_calls["n"] == 1,
        )
        check("--idx + --resume -> rows는 T2(유지) + T1(새로 돈 결과) = 2개, 중복 없음", len(rows_forced) == 2)
        t1_rows = [r for r in rows_forced if r.get("task_id") == "T1"]
        check("--idx + --resume -> T1은 딱 1개만 있음(예전 결과와 중복 안 됨)", len(t1_rows) == 1)
        check(
            "--idx + --resume -> T1은 예전 blocked=True가 아니라 새로 돈 결과(success=True)로 교체됨",
            t1_rows[0]["success"] is True and t1_rows[0]["blocked"] is False,
        )
        t2_rows = [r for r in rows_forced if r.get("task_id") == "T2"]
        check("--idx + --resume -> T2(강제 대상 아님)는 예전 결과 그대로 유지됨", len(t2_rows) == 1 and t2_rows[0]["success"] is True)
        check(
            "--idx + --resume -> success_rate도 중복 없이 T1(새 성공)+T2(기존 성공) 기준으로 계산됨(2/2)",
            abs(rate_forced - 1.0) < 1e-6,
        )
        with open(out_path5, encoding="utf-8") as f:
            saved_forced = [json.loads(line) for line in f]
        check("--idx + --resume -> 파일에도 딱 2줄만 남음(디스크에 중복 안 남음)", len(saved_forced) == 2)
        check(
            "--idx + --resume -> 파일 안 T1도 새 결과로 교체돼있음(디스크와 rows가 일치)",
            sum(1 for r in saved_forced if r.get("task_id") == "T1" and r.get("success") is True) == 1,
        )

    # --- _convert_planner_action_to_env / build_planner_grounding_agent_step ---
    import sys
    import types

    fake_ground_calls = []

    def _fake_ground(model, instruction, screenshot, **kwargs):
        fake_ground_calls.append((instruction, kwargs))
        if instruction in ("fail me", "fail start", "fail end"):
            return {"result": "wrong_format", "point": None, "raw_response": "??"}
        # drag 시작/끝점이 서로 다른 좌표로 grounding되는지 구분해서 검증하기 위해
        # instruction별로 다른 point를 반환한다.
        if instruction == "drag start":
            return {"result": "positive", "point": [0.1, 0.2], "raw_response": "(100,200)"}
        if instruction == "drag end":
            return {"result": "positive", "point": [0.6, 0.8], "raw_response": "(600,800)"}
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

        # drag -> 시작/끝점 각각 grounding해서 left_click_drag의 start_coordinate/coordinate로 변환
        env_act4 = _convert_planner_action_to_env(
            {"action": "drag", "target_description": "drag start", "text": "drag end"}, None, wide_img, {}
        )
        check(
            "drag -> left_click_drag 액션으로 변환됨",
            env_act4["action"] == "left_click_drag",
        )
        check(
            "drag -> 시작점이 target_description grounding 결과(0.1,0.2)*[w,h]로 변환됨",
            env_act4.get("start_coordinate") == [20.0, 20.0],
        )
        check(
            "drag -> 끝점이 text grounding 결과(0.6,0.8)*[w,h]로 변환됨",
            env_act4.get("coordinate") == [120.0, 80.0],
        )
        check(
            "drag -> ground_fn이 시작점/끝점 각각 한 번씩, 총 2번 호출됨",
            sum(1 for instr, _ in fake_ground_calls if instr in ("drag start", "drag end")) == 2,
        )

        # drag: target_description/text 중 하나라도 비어있으면 no-op
        env_act4b = _convert_planner_action_to_env(
            {"action": "drag", "target_description": "drag start", "text": ""}, None, wide_img, {}
        )
        check("drag -> 끝점 설명 없으면 wait no-op", env_act4b["action"] == "wait")

        # drag: 시작점 grounding 실패 -> wait no-op(끝점은 아예 호출 안 됨)
        fake_ground_calls.clear()
        env_act4c = _convert_planner_action_to_env(
            {"action": "drag", "target_description": "fail start", "text": "drag end"}, None, wide_img, {}
        )
        check("drag -> 시작점 grounding 실패 -> wait no-op", env_act4c["action"] == "wait")
        check(
            "drag -> 시작점 실패 시 끝점은 grounding 호출조차 안 함(불필요한 호출 방지)",
            all(instr != "drag end" for instr, _ in fake_ground_calls),
        )

        # drag: 끝점 grounding 실패 -> wait no-op
        env_act4d = _convert_planner_action_to_env(
            {"action": "drag", "target_description": "drag start", "text": "fail end"}, None, wide_img, {}
        )
        check("drag -> 끝점 grounding 실패 -> wait no-op", env_act4d["action"] == "wait")

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
        check(
            "back 패스스루(2026-08-11 추가 - 뒤로가기)",
            _convert_planner_action_to_env({"action": "back"}, None, wide_img, {}) == {"action": "back"},
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

        # (2026-08-15 추가 - 실측 버그 재현) GUI-Actor가 "unknown."이라고 답하고 나서 챗 템플릿/
        # 액션 포맷 토큰을 이어붙여 헛소리를 계속 생성하는 게 실측으로 확인됨(ArXiv--4 태스크
        # final_answer에 그대로 남음) - 정확 일치("unknown")만 걸러내던 예전 로직은 이 변형을
        # 못 잡았다. 마커 이후를 잘라내고 구두점 뗀 뒤 비교하는 강화된 로직이 이걸 잡아내는지 확인.
        fake_gm_leaked = _make_fake_grounding_model("unknown.\nassistantos\npyautogui.click([1] )")
        check(
            "_extract_final_answer -> 'unknown.' + 챗템플릿 이탈 텍스트도 None으로 걸러짐(실측 버그 재현)",
            _extract_final_answer(fake_gm_leaked, "how many papers?", wide_img) is None,
        )

        # 이탈 마커가 있어도 마커 앞부분이 진짜 답변이면(단순 "unknown"이 아니면) 그 앞부분은 살림
        fake_gm_leaked_real_answer = _make_fake_grounding_model("42 papers\nassistantos\npyautogui.click([1] )")
        check(
            "_extract_final_answer -> 이탈 마커 앞의 진짜 답변은 마커 이후만 잘라내고 살림",
            _extract_final_answer(fake_gm_leaked_real_answer, "how many papers?", wide_img) == "42 papers",
        )

        # 구두점만 붙은 변형("unknown!" 등)도 정확 일치가 아니라서 예전엔 못 걸렀음
        fake_gm_unknown_punct = _make_fake_grounding_model("Unknown!")
        check(
            "_extract_final_answer -> 'Unknown!'처럼 대소문자/구두점만 다른 변형도 None으로 걸러짐",
            _extract_final_answer(fake_gm_unknown_punct, "q", wide_img) is None,
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
                fake_model, fake_planning_view, use_reflection=True, verbose=False, use_regionfocus=False,
            )
            step1 = agent_step_fn(wide_img, {"instruction": "close the window"}, {"actions": [], "screenshots": []})
            check(
                "agent_step_fn(reflection) -> planner plan을 env 액션으로 변환",
                step1 == {"action": "left_click", "coordinate": [50.0, 75.0]},
            )
            step2 = agent_step_fn(wide_img, {"instruction": "close the window"}, {"actions": [], "screenshots": []})
            check("agent_step_fn -> 두 번째 호출에서 history_actions에 이전 plan이 누적됨", plan_calls[1]["history_len"] == 1)

            # (2026-08-11 추가 - 버그 수정 검증) 새 태스크로 넘어갈 때(reset_episode() 호출)
            # planner_history가 비워져서, 이전 태스크의 기록이 다음 태스크로 새지 않아야 함.
            check("agent_step_fn.reset_episode 훅이 존재함", callable(getattr(agent_step_fn, "reset_episode", None)))
            agent_step_fn.reset_episode()
            step3 = agent_step_fn(wide_img, {"instruction": "a new task"}, {"actions": [], "screenshots": []})
            check(
                "reset_episode() 호출 후 다음 태스크의 첫 스텝은 history_len=0으로 시작함"
                "(이전 태스크 기록이 새지 않음)",
                plan_calls[2]["history_len"] == 0,
            )

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
                fake_model, fake_planning_view, use_reflection=False, verbose=False, use_regionfocus=False,
            )
            plan_calls.clear()
            agent_step_fn_no_reflect(wide_img, {"instruction": "x"}, {"actions": [], "screenshots": []})
            check("--no_reflect -> plan_next_action(reflection 없는 쪽) 사용", len(plan_calls) == 1)

            # terminate인데 answer가 없으면 agent_step_fn이 answer 추출을 자동으로 태우는지
            # (2026-08-15 수정) _extract_final_answer가 이제 grounding_model이 아니라
            # planning_view를 받으므로, 여기서도 fake_planning_view.generate를 채워서 확인한다 -
            # grounding_model(fake_model_for_answer)은 정말로 호출 안 되는지 확인하는 용도로 남김.
            def _fake_plan_terminate_no_answer(planning_view, instruction, screenshot, history_actions=None, **kw):
                return {"reasoning": "r", "action": "terminate", "status": "success"}

            fake_planner_module.plan_with_reflection = _fake_plan_terminate_no_answer
            fake_model_for_answer = _make_fake_grounding_model("SHOULD NOT BE USED FOR ANSWER")
            fake_planning_view.generate.reset_mock()
            fake_planning_view.generate.return_value = "42"
            agent_step_fn2 = build_planner_grounding_agent_step(
                fake_model_for_answer, fake_planning_view, use_reflection=True, verbose=False, use_regionfocus=False,
            )
            term_action = agent_step_fn2(wide_img, {"instruction": "what is the answer?"}, {"actions": [], "screenshots": []})
            check(
                "terminate + answer 없음 -> _extract_final_answer로 채워서 env action의 text에 들어감",
                term_action == {"action": "terminate", "status": "success", "text": "42"},
            )
            check(
                "terminate + answer 없음 -> answer 추출은 planning_view로 감(grounding_model 아님)",
                fake_planning_view.generate.called and not fake_model_for_answer.generate.called,
            )

            # terminate인데 answer가 이미 있으면 추출 호출 자체를 안 해야 함(중복 호출 낭비 방지)
            def _fake_plan_terminate_with_answer(planning_view, instruction, screenshot, history_actions=None, **kw):
                return {"reasoning": "r", "action": "terminate", "status": "success", "answer": "already have it"}

            fake_planner_module.plan_with_reflection = _fake_plan_terminate_with_answer
            fake_planning_view.generate.reset_mock()
            agent_step_fn3 = build_planner_grounding_agent_step(
                fake_model_for_answer, fake_planning_view, use_reflection=True, verbose=False, use_regionfocus=False,
            )
            term_action2 = agent_step_fn3(wide_img, {"instruction": "x"}, {"actions": [], "screenshots": []})
            check(
                "terminate + answer 이미 있음 -> 추출 재호출 안 하고 기존 answer 그대로 사용",
                term_action2 == {"action": "terminate", "status": "success", "text": "already have it"}
                and not fake_planning_view.generate.called,
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
                fake_model_should_not_ground, fake_planning_view, use_reflection=True, verbose=False, use_regionfocus=False,
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
                fake_model, fake_planning_view, use_reflection=True, verbose=False, use_regionfocus=False,
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
                fake_model, fake_planning_view, use_reflection=True, verbose=False, use_regionfocus=False,
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

    # --- (2026-08-11 추가 - RegionFocus 재연결 배선 검증) _build_click_ground_fn 단위 테스트 ---
    # click grounding을 gui_grounding.ground()(초기 1회) 대신 region_focus.ground_with_regionfocus()
    # (재탐색+crop/zoom 정밀화)로 돌리도록 재연결한 부분 - 새로 생긴 기본 경로(use_regionfocus=True)가
    # 실제로 region_focus를 타는지, 끄면(off) 여전히 예전 경로(gui_grounding.ground())를 타는지,
    # task_id/옵션들이 올바르게 전달되고 두 백엔드가 서로 모르는 kwargs(task_id/max_new_tokens)는
    # 조용히 걸러지는지 확인한다.
    small_img = Image.new("RGB", (200, 100))
    rf_calls = []

    def _fake_ground_with_regionfocus(model, instruction, screenshot, **kw):
        rf_calls.append(kw)
        return {"result": "positive", "point": [0.5, 0.5], "raw_response": "(500,500)"}

    fake_region_focus_module = types.ModuleType("region_focus")
    fake_region_focus_module.ground_with_regionfocus = _fake_ground_with_regionfocus
    sys.modules["region_focus"] = fake_region_focus_module
    try:
        # --- off: 예전 경로(gui_grounding.ground()) 그대로 ---
        plain_calls = []

        def _fake_plain_ground(model, instruction, screenshot, **kw):
            plain_calls.append(kw)
            return {"result": "positive", "point": [0.1, 0.1], "raw_response": "(100,100)"}

        fake_gui_grounding_module_rf = types.ModuleType("gui_grounding")
        fake_gui_grounding_module_rf.ground = _fake_plain_ground
        sys.modules["gui_grounding"] = fake_gui_grounding_module_rf
        try:
            click_fn_off = _build_click_ground_fn(use_regionfocus=False)
            r_off = click_fn_off(MagicMock(), "click X", small_img, task_id="t1", max_new_tokens=128)
            check("_build_click_ground_fn(off) -> gui_grounding.ground() 호출됨", len(plain_calls) == 1)
            check(
                "_build_click_ground_fn(off) -> task_id는 gui_grounding.ground()로 안 넘어감(모르는 kwarg라 걸러짐)",
                "task_id" not in plain_calls[0],
            )
            check("_build_click_ground_fn(off) -> 결과 그대로 반환", r_off["point"] == [0.1, 0.1])
        finally:
            del sys.modules["gui_grounding"]

        # --- on(RegionFocus): region_focus.ground_with_regionfocus() 경로 ---
        click_fn_on = _build_click_ground_fn(
            use_regionfocus=True, regionfocus_debug_image=True, regionfocus_debug_text=False,
            regionfocus_step1_format="point_text", regionfocus_step4_format="toolcall_pixel",
        )
        rf_calls.clear()
        r_on = click_fn_on(MagicMock(), "click Y", small_img, task_id="task-42", max_new_tokens=128, min_pixels=100)
        check("_build_click_ground_fn(on) -> region_focus.ground_with_regionfocus() 호출됨", len(rf_calls) == 1)
        check("_build_click_ground_fn(on) -> task_id 전달됨", rf_calls[0].get("task_id") == "task-42")
        check(
            "_build_click_ground_fn(on) -> debug_image/step1_format/step4_format 옵션 전달됨",
            rf_calls[0].get("debug_image") is True
            and rf_calls[0].get("step1_format") == "point_text"
            and rf_calls[0].get("step4_format") == "toolcall_pixel",
        )
        check("_build_click_ground_fn(on) -> max_new_tokens는 안 넘어감(내부에서 자체 처리)", "max_new_tokens" not in rf_calls[0])
        check("_build_click_ground_fn(on) -> min_pixels는 그대로 통과", rf_calls[0].get("min_pixels") == 100)
        check("_build_click_ground_fn(on) -> 결과 그대로 반환", r_on["point"] == [0.5, 0.5])

        # --- build_planner_grounding_agent_step 기본값(use_regionfocus=True)이 실제로 RegionFocus를 탐 ---
        fake_planner_module_rf = types.ModuleType("planner")

        def _fake_plan_click(planning_view, instruction, screenshot, history_actions=None, **kw):
            return {"reasoning": "r", "action": "left_click", "target_description": "the X button"}

        fake_planner_module_rf.plan_next_action = _fake_plan_click
        fake_planner_module_rf.plan_with_reflection = _fake_plan_click
        sys.modules["planner"] = fake_planner_module_rf
        try:
            rf_calls.clear()
            fake_model_rf = MagicMock()
            agent_step_fn_rf = build_planner_grounding_agent_step(
                fake_model_rf, fake_model_rf, use_reflection=False, verbose=False,
            )  # use_regionfocus 인자를 안 줌 -> 기본값(True) 그대로 검증
            agent_step_fn_rf(small_img, {"instruction": "x", "id": "task-99"}, {"actions": [], "screenshots": []})
            check(
                "build_planner_grounding_agent_step 기본값 -> click 액션이 RegionFocus 경로를 탐"
                "(재연결 배선 확인)",
                len(rf_calls) == 1 and rf_calls[0].get("task_id") == "task-99",
            )
        finally:
            del sys.modules["planner"]
    finally:
        del sys.modules["region_focus"]

    # --- (2026-08-11 추가 - 버그 수정 검증) _DebugModelView가 .generate() 말고 다른 속성/메서드도
    # 내부 객체로 위임하는지 - region_focus.py의 judge_inference()가 .generate()를 거치지 않고
    # qwen_model.model / qwen_model.processor를 직접 꺼내 쓰다가 실제로 AttributeError가 났었음
    # (처음엔 .model 프로퍼티만 명시적으로 통과시켜서 .processor는 안 뚫려 있었음).
    class _FakeInnerModel:
        def __init__(self):
            self.model = "the-underlying-peft-model"
            self.processor = "the-underlying-processor"
            self.device = "cuda:0"

        def generate(self, messages, **kwargs):
            return "raw response"

    fake_inner = _FakeInnerModel()
    fake_recorder = _PromptRecorder(base_dir="/tmp/unused")  # generate() 기록 대상 - 이 테스트에선 안 씀
    debug_view = _DebugModelView(fake_inner, fake_recorder, "test-tag")
    check("_DebugModelView -> .model 속성이 내부 객체로 위임됨", debug_view.model == "the-underlying-peft-model")
    check("_DebugModelView -> .processor 속성이 내부 객체로 위임됨(region_focus.py 버그 재발 방지)", debug_view.processor == "the-underlying-processor")
    check("_DebugModelView -> .device처럼 임의의 다른 속성도 위임됨", debug_view.device == "cuda:0")
    check("_DebugModelView -> .generate()는 위임 안 되고 이 클래스가 직접 처리(기록용으로 오버라이드됨)", debug_view.generate([]) == "raw response")

    # --- (2026-08-11 추가) 태스크별 프롬프트/응답 덤프(--debug_dir) 배선 확인 ---
    # 위쪽 테스트들의 fake plan_next_action/ground()는 model.generate()를 아예 안 부르고
    # 결과만 바로 반환하는 "완전히 껍데기"라서(그래서 각 액션 변환 로직만 빠르게 테스트할 수
    # 있었음), _DebugModelView가 실제로 .generate() 호출을 가로채서 기록하는지는 그걸로 검증이
    # 안 된다 - 이 블록은 fake들이 model.generate()를 실제로 호출하게 만들어서 배선을 끝까지
    # 확인한다.
    import shutil
    import tempfile

    dbg_dir = tempfile.mkdtemp(prefix="ewv2_debug_dump_")
    try:
        fake_view = MagicMock()
        fake_view.generate.return_value = (
            '{"reasoning": "r", "action": "left_click", "target_description": "the X button"}'
        )

        def _fake_plan_calls_generate(planning_view, instruction, screenshot, history_actions=None, **kw):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": screenshot},
                        {"type": "text", "text": f"task: {instruction}"},
                    ],
                }
            ]
            planning_view.generate(messages, max_new_tokens=kw.get("max_new_tokens", 10), temperature=0.0)
            return {"reasoning": "r", "action": "left_click", "target_description": "the X button"}

        def _fake_ground_calls_generate(model, instruction, screenshot, **kwargs):
            messages = [{"role": "user", "content": [{"type": "text", "text": f"ground: {instruction}"}]}]
            model.generate(messages, max_new_tokens=kwargs.get("max_new_tokens", 10))
            return {"result": "positive", "point": [0.25, 0.75], "raw_response": "(250,750)"}

        dbg_planner_module = types.ModuleType("planner")
        dbg_planner_module.plan_next_action = _fake_plan_calls_generate
        dbg_planner_module.plan_with_reflection = _fake_plan_calls_generate
        dbg_gui_grounding_module = types.ModuleType("gui_grounding")
        dbg_gui_grounding_module.ground = _fake_ground_calls_generate
        sys.modules["planner"] = dbg_planner_module
        sys.modules["gui_grounding"] = dbg_gui_grounding_module
        try:
            agent_step_fn_dbg = build_planner_grounding_agent_step(
                fake_view, fake_view, use_reflection=False, verbose=False, debug_dir=dbg_dir,
                use_regionfocus=False,
            )
            agent_step_fn_dbg(
                Image.new("RGB", (200, 100)), {"instruction": "find the button", "id": "task-007"},
                {"actions": [], "screenshots": []},
            )

            task_dir = os.path.join(dbg_dir, "task-007")
            check("--debug_dir -> 태스크 폴더가 task id 기준으로 생성됨", os.path.isdir(task_dir))
            dumped = os.listdir(task_dir) if os.path.isdir(task_dir) else []
            check("--debug_dir -> planner 프롬프트 파일 생성됨", "step00_planner_00.txt" in dumped)
            check("--debug_dir -> grounding 프롬프트 파일 생성됨", "step00_grounding_00.txt" in dumped)
            check(
                "--debug_dir(기본) -> 프롬프트에 포함된 스크린샷이 png로도 저장됨",
                "step00_planner_00_img0.png" in dumped,
            )
            if "step00_planner_00.txt" in dumped:
                with open(os.path.join(task_dir, "step00_planner_00.txt"), encoding="utf-8") as fh:
                    dumped_content = fh.read()
                check(
                    "--debug_dir -> 저장된 파일에 PROMPT/RESPONSE 섹션과 실제 프롬프트 텍스트가 담김",
                    "=== PROMPT ===" in dumped_content
                    and "=== RESPONSE ===" in dumped_content
                    and "task: find the button" in dumped_content,
                )
                check(
                    "--debug_dir -> 프롬프트 텍스트에 저장된 이미지 파일명이 같이 적힘",
                    "step00_planner_00_img0.png" in dumped_content,
                )
        finally:
            del sys.modules["planner"]
            del sys.modules["gui_grounding"]
    finally:
        shutil.rmtree(dbg_dir, ignore_errors=True)

    # --- (2026-08-11 추가) --no_debug_images(debug_save_images=False) -> 이미지 저장 안 함 ---
    dbg_dir2 = tempfile.mkdtemp(prefix="ewv2_debug_dump_noimg_")
    try:
        fake_view2 = MagicMock()
        fake_view2.generate.return_value = (
            '{"reasoning": "r", "action": "left_click", "target_description": "the X button"}'
        )
        dbg_planner_module2 = types.ModuleType("planner")
        dbg_planner_module2.plan_next_action = _fake_plan_calls_generate
        dbg_planner_module2.plan_with_reflection = _fake_plan_calls_generate
        dbg_gui_grounding_module2 = types.ModuleType("gui_grounding")
        dbg_gui_grounding_module2.ground = _fake_ground_calls_generate
        sys.modules["planner"] = dbg_planner_module2
        sys.modules["gui_grounding"] = dbg_gui_grounding_module2
        try:
            agent_step_fn_dbg2 = build_planner_grounding_agent_step(
                fake_view2, fake_view2, use_reflection=False, verbose=False,
                debug_dir=dbg_dir2, debug_save_images=False, use_regionfocus=False,
            )
            agent_step_fn_dbg2(
                Image.new("RGB", (200, 100)), {"instruction": "find the button", "id": "task-008"},
                {"actions": [], "screenshots": []},
            )
            task_dir2 = os.path.join(dbg_dir2, "task-008")
            dumped2 = os.listdir(task_dir2) if os.path.isdir(task_dir2) else []
            check(
                "debug_save_images=False -> png 파일은 안 생기고 txt만 남음",
                "step00_planner_00.txt" in dumped2 and not any(fn.endswith(".png") for fn in dumped2),
            )
        finally:
            del sys.modules["planner"]
            del sys.modules["gui_grounding"]
    finally:
        shutil.rmtree(dbg_dir2, ignore_errors=True)

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
    # (2026-08-11 수정 - --judge -> --judge_backend, --openai_model -> --judge_api_model 개명)
    # planner 쪽 옵션 이름(--planner_backend {local,openai} / --planner_api_model)과 짝이
    # 안 맞아서 헷갈린다는 지적에 따라 동일한 네이밍 패턴으로 맞췄다. 값(choices)과 동작은
    # 그대로("qwen"/"openai"), 이름만 바꿈.
    ap.add_argument("--judge_backend", choices=["qwen", "openai"], default="qwen")
    ap.add_argument("--judge_api_model", default="gpt-4o", help="--judge_backend openai일 때 쓸 모델 이름")
    # (2026-08-11 추가 - 태스크 간 지연 단축) judge_repeats(기본 3)번의 judge 호출을 병렬로
    # 쏠지. 기본(None)이면 --judge_backend openai일 때만 자동으로 judge_repeats만큼 병렬화
    # 하고(서로 독립된 API 호출이라 안전), qwen judge는 자동으로 켜지 않는다(같은 로컬 모델
    # 인스턴스를 여러 스레드에서 동시에 generate() 호출하는 게 안전하다는 보장이 없어서).
    # 명시적으로 값을 주면 그 값을 그대로 씀(로컬 judge에서 강제로 켜고 싶은 경우 등).
    ap.add_argument("--judge_max_workers", type=int, default=None,
                     help="judge_repeats번 호출을 몇 개까지 동시에 실행할지. 미지정시 "
                          "--judge_backend openai면 judge_repeats(완전 병렬), qwen이면 1(순차).")
    ap.add_argument("--adapter_dir", default=None,
                     help="Qwen judge용 LoRA 어댑터 (선택). --reuse_agent_model_for_judge를 켜면 무시됨.")
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--judge_repeats", type=int, default=DEFAULT_JUDGE_REPEATS)
    ap.add_argument("--limit", type=int, default=None)
    # (2026-08-15 추가 - 사용자 요청) 특정 태스크만 골라서 재실행. scroll/wait 뺑뺑이 오탐
    # 버그를 고친 뒤, 원래 전체실행 결과 jsonl에서 그 사유로 blocked됐던 태스크들의 "idx"만
    # 골라 다시 돌려서 고쳐졌는지 확인하는 용도로 만듦(원인이 다른 실패까지 전부 재실행할
    # 필요 없이 딱 원하는 것만). 결과 jsonl에도 원래 idx가 그대로 찍혀서 이전 실행 결과와
    # 바로 대조 가능(run_batch()의 task_indices 참고).
    ap.add_argument("--idx", type=str, default=None,
                     help="쉼표로 구분한 태스크 인덱스만 골라서 돌린다(예: --idx 4,9,28). 인덱스는 "
                          "--web_name 필터링 후, --limit 적용 전 순서 기준 - 즉 이전 실행 결과 jsonl의 "
                          "'idx' 필드와 동일 기준. 지정하면 --limit은 무시된다.")
    ap.add_argument("--out", default=None)
    # (2026-08-11 추가 - API 비용 보호) --out 파일에 이미 있는 task_id는 건너뛰고 이어서
    # 돌린다. 장시간 무중단 실행(개인 PC로 수백 개 태스크) 중 죽었을 때 이미 낸 GPT-4o
    # planner/judge API 비용이 낭비되지 않도록 하는 게 목적 - run_batch()/_task_key() 참고.
    ap.add_argument("--resume", action="store_true",
                     help="--out에 이미 있는 결과(task_id 기준)는 건너뛰고 이어서 실행. "
                          "--out 미지정이면 의미 없음(재개할 파일이 없으므로).")
    # (2026-08-11 추가) 태스크별 폴더에 스텝별 프롬프트/응답 원문을 txt로 저장. --log_file(콘솔
    # 전체 흐름)과는 별개로, "그 스텝에서 모델에 정확히 뭐가 들어갔는지"를 태스크/스텝 단위로
    # 찾아보기 쉽게 구조화한 것 - build_planner_grounding_agent_step()의 _PromptRecorder 참고.
    ap.add_argument("--debug_dir", default="debug",
                     help="태스크별 폴더(<debug_dir>/<태스크id>/stepNN_<planner|reflection|"
                          "grounding|answer_extraction>_NN.txt)에 프롬프트/응답 저장(기본 './debug').")
    ap.add_argument("--no_debug_dump", action="store_true", help="태스크별 프롬프트/응답 덤프를 끔")
    ap.add_argument("--no_debug_images", action="store_true",
                     help="프롬프트에 포함된 스크린샷을 png로 같이 저장하지 않음(텍스트만 저장, 용량 절약)")
    # (2026-08-11 추가) CAPTCHA/bot-check 대응. env_webvoyager.WebVoyagerEnv.detect_bot_check()/
    # run_episode() docstring 참고 - CAPTCHA를 풀거나 우회하지 않고, 감지 시 정직하게 blocked로
    # 표기하고 조기 종료해서 max_steps/judge 비용을 낭비하지 않게 하는 것까지만 한다.
    ap.add_argument("--captcha_reset_retries", type=int, default=1,
                     help="reset() 직후 bot-check가 감지되면 새 브라우저 세션으로 몇 번 더 재시도할지 "
                          "(기본 1회 = 최초 시도 포함 총 2회). env_webvoyager.WebVoyagerEnv 생성자로 전달됨.")
    # (2026-08-11 추가 - 수동 CAPTCHA 통과) "직접 눌러주면 될 것 같다"는 요청 대응. 자동
    # 재시도를 다 쓰고도 bot-check가 안 풀리면, 자동으로 포기하는 대신 잠깐 멈춰서 콘솔에서
    # Enter를 누를 때까지 기다린다 - 그 사이에 headless=False로 띄운 실제 브라우저 창에서
    # 직접 CAPTCHA를 풀면 된다. --headless(기본 켜짐)인 채로 이 옵션만 켜면 사람이 볼 화면
    # 자체가 없어서 의미가 없다 - 반드시 --no_headless와 같이 쓸 것.
    ap.add_argument("--manual_captcha", action="store_true",
                     help="bot-check가 자동 재시도로도 안 풀리면 멈춰서 사람이 직접 풀 때까지 기다린다. "
                          "--no_headless와 같이 써야 실제로 풀 수 있는 화면이 보인다.")
    ap.add_argument("--no_headless", dest="headless", action="store_false", default=True,
                     help="브라우저 창을 실제로 띄운다(headless 끔). --manual_captcha와 같이 쓰면 "
                          "CAPTCHA를 직접 풀 수 있음. 기본은 headless(창 안 띄움).")
    # (2026-08-11 추가 - 태스크 간 지연 단축) 기본은 브라우저 재사용(빠름). 태스크 간 격리를
    # 더 강하게 보장하고 싶으면(드라이버가 불안정해지는 게 의심될 때 등) 끌 것 - 예전처럼
    # 매 태스크마다 Chrome을 통째로 재기동한다(느림).
    ap.add_argument("--no_reuse_driver", dest="reuse_driver", action="store_false", default=True,
                     help="태스크마다 브라우저를 재사용하지 않고 매번 새로 켠다(느려짐, 격리는 더 강함). "
                          "기본은 재사용(쿠키만 초기화).")
    # (2026-08-15 추가 - ESPN 등 광고 새탭 리다이렉트 대응) 기본은 새 탭이 열리면 자동으로 그리로
    # 포커스를 옮기는데(Booking처럼 사이트가 실제 콘텐츠를 새 탭에 여는 경우 대응), ESPN 등은
    # 광고가 새 탭으로 리다이렉트되는 경우가 있어서 이 정책이 오히려 역효과를 낼 수 있다.
    ap.add_argument("--no_auto_switch_new_tab", dest="auto_switch_new_tab", action="store_false", default=True,
                     help="새 탭이 열려도 자동으로 그리로 전환하지 않는다(원래 탭에 그대로 머무름, "
                          "새 탭은 무시). 광고가 새 탭/리다이렉트로 뜨는 사이트(ESPN 등)에서 사용. "
                          "기본은 자동 전환.")
    # (2026-08-15 추가 - Billboard 등 무거운 사이트 렌더러 hang 대응) 실측: 광고/비디오가
    # 무거운 페이지에서 CDP 명령이 "Timed out receiving message from renderer: 20.000"
    # (selenium.TimeoutException)으로 죽는 경우가 있었다 - 이 20.000이 바로 아래 기본값이다.
    # 늘리면 "느리지만 결국 응답하는" 무거운 페이지를 안 죽이고 넘어갈 수 있다. 다만 이건
    # 완전한 해결책은 아니다 - 진짜로 렌더러가 죽어버린(무한루프/크래시) 경우는 얼마나 늘려도
    # 결국 타임아웃되므로, run_batch()의 예외 안전망(태스크 하나만 blocked 처리 후 계속 진행)은
    # 이 값과 별개로 항상 켜져 있다.
    ap.add_argument("--page_load_timeout", type=float, default=20.0,
                     help="페이지 로딩 및 (렌더러가 바쁠 때) CDP 명령 응답을 기다리는 최대 시간(초). "
                          "광고/비디오가 무거운 사이트(Billboard 등)에서 렌더러 hang으로 태스크가 "
                          "자꾸 죽으면 늘려볼 것(예: 40). 기본 20초.")
    ap.add_argument("--stuck_repeat_threshold", type=int, default=DEFAULT_STUCK_REPEAT_THRESHOLD,
                     help="최근 --stuck_repeat_window개 액션 안에 같은 액션이 이만큼 등장하면 "
                          "(bot-check 신호 유무와 무관하게) 멈춘 것으로 보고 조기 종료한다(기본 5). "
                          "연속일 필요 없음 - 두세 개 액션을 번갈아 반복하는 뺑뺑이도 잡는다.")
    ap.add_argument("--stuck_repeat_window", type=int, default=None,
                     help="위 반복 감지에 사용할 윈도우 크기(최근 몇 개 액션 안에서 셀지). "
                          "생략하면 --stuck_repeat_threshold의 2배로 자동 설정된다.")
    # (2026-08-09 추가) 실제 planner+grounding 정책. --agent_grounding_adapter_dir를 안 주면
    # 기존처럼 dummy_agent_step(즉시 실패)로 동작 - 파이프라인 배선만 확인하고 싶을 때는 그대로 둘 것.
    ap.add_argument("--agent_grounding_adapter_dir", default=None,
                     help="grounding LoRA 체크포인트(예: checkpoints/qwen2.5vl-3b-gui-lora-stage2/"
                          "checkpoint-4130). 지정해야 dummy_agent_step 대신 실제 정책(planner+grounding)이 돈다. "
                          "--grounding_backend gui_actor일 때는 무시됨(LoRA 체크포인트 대신 GUI-Actor 사전학습 "
                          "가중치를 씀).")
    # [v3 추가] click grounding 백엔드를 통째로 GUI-Actor로 바꿔서 실험. gui_actor_grounding.py
    # 참고 - microsoft/GUI-Actor-3B-Qwen2.5-VL을 gui_grounding.ground()와 동일한 반환 스키마로
    # 감싼 어댑터를 쓴다. --planner_backend openai와만 같이 쓸 수 있다(GUI-Actor 모델은 우리
    # planner LoRA를 얹을 수 있는 멀티 어댑터 PeftModel이 아니라서, 로컬 planner LoRA 공유 구조와
    # 호환되지 않음). RegionFocus(judge/재탐색)와 reflection(비평 루프)은 둘 다 우리 LoRA 전용
    # 텍스트 포맷/파이프라인에 결합돼 있어 GUI-Actor에는 못 얹는다 - 아래에서 --use_regionfocus/
    # --use_reflection이 켜져 있으면 경고를 찍고 자동으로 끈다.
    ap.add_argument("--grounding_backend", choices=["lora", "gui_actor"], default="lora",
                     help="lora(기본): --agent_grounding_adapter_dir의 우리 LoRA로 grounding. "
                          "gui_actor: microsoft/GUI-Actor-3B-Qwen2.5-VL로 grounding(사전학습 가중치, "
                          "재학습 없이 그대로 씀). --planner_backend openai 필수, RegionFocus/reflection "
                          "자동으로 꺼짐.")
    ap.add_argument("--gui_actor_model_id", default="microsoft/GUI-Actor-3B-Qwen2.5-VL",
                     help="--grounding_backend gui_actor일 때 로드할 HF 모델 id.")
    ap.add_argument("--gui_actor_attn_implementation", choices=["sdpa", "eager", "flash_attention_2"],
                     default="sdpa",
                     help="--grounding_backend gui_actor일 때 attention 구현체. flash-attn을 설치했으면 "
                          "flash_attention_2로 VRAM을 더 아낄 수 있음, 안 깔았으면 기본(sdpa) 권장.")
    ap.add_argument("--agent_planner_adapter_dir", default=None,
                     help="planner LoRA 체크포인트(예: checkpoints/qwen2.5vl-3b-planner-lora). 지정 안 하면 "
                          "planning은 base 모델(disable_adapter)로 돈다 - agent_loop.py의 load_shared_model 참고. "
                          "--planner_backend openai와는 같이 못 씀.")
    # (2026-08-11 추가) planner를 로컬 LoRA 대신 OpenAI API(GPT-4o 등)로 돌리는 옵션. api_planner.py
    # 참고 - grounding은 이 옵션과 무관하게 항상 --agent_grounding_adapter_dir의 로컬 LoRA가 담당한다
    # (planning_view 자리만 api_planner.OpenAIPlannerModel로 바뀌는 것 - planner.py는 duck-typing이라
    # 수정 불필요).
    ap.add_argument("--planner_backend", choices=["local", "openai"], default="local",
                     help="local(기본): --agent_planner_adapter_dir(있으면)로 로컬 LoRA planning. "
                          "openai: api_planner.OpenAIPlannerModel로 OpenAI API 호출해서 planning "
                          "(grounding은 그대로 로컬 LoRA).")
    ap.add_argument("--planner_api_model", default="gpt-4o", help="--planner_backend openai일 때 쓸 모델 이름")
    # (2026-08-11 수정 - --planner_api_key -> --api_key로 개명 + judge에도 적용) 원래는
    # planner(OpenAIPlannerModel)에만 넘어가고 judge(make_openai_judge)는 이 값을 안 보고
    # 환경변수 OPENAI_API_KEY만 봐서, --planner_api_key로 키를 넘겨도 judge 단계에서
    # "api_key must be set" 에러가 나는 문제가 실측으로 확인됐다(에이전트 루프는 끝까지
    # 돌고 judge 호출 직전에만 죽음). planner/judge가 같은 OpenAI 계정 키를 쓰는 게 보통이라
    # 이름을 --api_key로 통일하고, judge 쪽(--judge openai)에도 그대로 전달되도록 배선했다.
    ap.add_argument("--api_key", default=None,
                     help="OpenAI API 키. --planner_backend openai(planner)와 --judge_backend openai(judge) "
                          "양쪽 다 이 값을 쓴다. 미지정시 환경변수 OPENAI_API_KEY 사용.")
    ap.add_argument("--planner_api_base_url", default=None,
                     help="OpenAI 호환 엔드포인트(vLLM 등)를 쓸 때 지정 - 미지정시 OpenAI 공식 엔드포인트. "
                          "planner 전용(judge에는 base_url 옵션이 없음).")
    # (2026-08-11 수정 - 버그) default=False로 돼 있었던 걸 True로 고침. --no_reflect는
    # action="store_false"라 "플래그를 주면 끈다"는 의미인데, default까지 False였던 탓에
    # 플래그를 주든 안 주든 reflection이 항상 꺼진 채로 돌고 있었다(실측: 실제 실행 로그에
    # _reflection_log가 한 번도 안 남음 - CLI로는 reflection을 켤 방법 자체가 없었음).
    ap.add_argument("--no_reflect", dest="use_reflection", action="store_false", default=True,
                     help="plan_with_reflection 대신 plan_next_action만 사용(비평 루프 생략, 스텝당 더 빠름)")
    ap.add_argument("--max_iterations", type=int, default=2, help="--no_reflect가 아닐 때 reflection 최대 재시도")
    # (2026-08-11 추가 - region focus 재연결) click grounding에 region_focus.ground_with_regionfocus()를
    # 쓸지 결정. --no_reflect와 같은 패턴(store_false + default=True) - 기본으로 켜져 있고, 빠르게
    # 배선만 확인하고 싶을 때(또는 VRAM/시간이 부족할 때) --no_regionfocus로 끈다.
    ap.add_argument("--no_regionfocus", dest="use_regionfocus", action="store_false", default=True,
                     help="click grounding을 region_focus.ground_with_regionfocus()(재탐색+crop/zoom "
                          "정밀화) 대신 gui_grounding.ground()(초기 grounding 1회)만 쓰도록 끈다. "
                          "click 1회당 모델 호출이 5~9회 -> 1회로 줄어 훨씬 빠르지만 grounding 정확도는 "
                          "떨어짐(module docstring의 RegionFocus uplift 실측 참고).")
    ap.add_argument("--regionfocus_debug_image", action="store_true",
                     help="RegionFocus 재탐색 중간 이미지(crop/zoom/판단)를 ./debug/<task_id>/*.png로 저장 "
                          "(region_focus.py의 --debug_image와 동일한 용도)")
    ap.add_argument("--regionfocus_debug_text", action="store_true",
                     help="RegionFocus 각 단계에 실제로 들어간 프롬프트+응답 원문을 "
                          "./debug/<task_id>/prompt_*.txt로 저장 (region_focus.py의 --debug_text와 동일)")
    ap.add_argument("--regionfocus_step1_format", choices=["point_text", "toolcall_norm1000", "toolcall_pixel"],
                     default="point_text", help="RegionFocus Step1(초기 grounding) 좌표 요청 방식 - "
                     "기본(학습 포맷 그대로) 권장, ablation 실험용으로만 바꿀 것(region_focus.py 참고)")
    ap.add_argument("--regionfocus_step4_format", choices=["point_text", "toolcall_norm1000", "toolcall_pixel"],
                     default="point_text", help="RegionFocus Step4(crop/zoom 후 좌표 재추출) 좌표 요청 방식 - "
                     "기본(학습 포맷 그대로) 권장, ablation 실험용으로만 바꿀 것(region_focus.py 참고)")
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
        task_indices = None
        if args.idx:
            try:
                wanted_idx = [int(x.strip()) for x in args.idx.split(",") if x.strip() != ""]
            except ValueError:
                raise SystemExit(f"--idx 파싱 실패: {args.idx!r} (쉼표로 구분된 정수 목록이어야 함, 예: 0,5,12)")
            out_of_range = [i for i in wanted_idx if i < 0 or i >= len(tasks)]
            if out_of_range:
                raise SystemExit(
                    f"--idx에 범위를 벗어난 인덱스가 있음: {out_of_range} (--web_name 필터링 후 총 태스크 수: {len(tasks)})"
                )
            task_indices = wanted_idx
            tasks = [tasks[i] for i in wanted_idx]
        elif args.limit:
            tasks = tasks[: args.limit]
        if not tasks:
            raise SystemExit(f"{args.tasks_jsonl}에서 태스크를 하나도 못 찾음 (파일이 비었거나 경로 확인 필요)")

        env = WebVoyagerEnv(
            captcha_reset_retries=args.captcha_reset_retries, reuse_driver=args.reuse_driver,
            headless=args.headless, manual_captcha_wait=args.manual_captcha,
            auto_switch_new_tab=args.auto_switch_new_tab,
            page_load_timeout=args.page_load_timeout,
        )

        if args.planner_backend == "openai" and args.agent_planner_adapter_dir:
            raise SystemExit(
                "--planner_backend openai와 --agent_planner_adapter_dir는 같이 못 씀 - planner가 "
                "API로 도니 로컬 planner LoRA를 로드할 이유가 없음."
            )

        # [v3 추가] --grounding_backend gui_actor는 로컬 planner LoRA를 얹을 수 있는 멀티 어댑터
        # PeftModel 구조 밖에 있는 별도 모델(microsoft/GUI-Actor-3B-Qwen2.5-VL)이라, planner도
        # 로컬 LoRA가 아니라 API(OpenAI)로 돌아야만 한다 - load_shared_model() 경로(agent_loop.py)와
        # 애초에 호환되지 않는다.
        if args.grounding_backend == "gui_actor" and args.planner_backend != "openai":
            raise SystemExit(
                "--grounding_backend gui_actor는 --planner_backend openai와만 같이 쓸 수 있음 - "
                "GUI-Actor는 우리 planner LoRA를 얹을 수 있는 멀티 어댑터 PeftModel이 아님."
            )
        if args.grounding_backend == "gui_actor" and args.reuse_agent_model_for_judge:
            raise SystemExit(
                "--grounding_backend gui_actor와 --reuse_agent_model_for_judge는 같이 못 씀 - "
                "GUI-Actor 모델은 judge가 기대하는 자유형 텍스트 생성(.generate())을 지원하지 않음. "
                "--judge_backend openai를 쓰거나, --reuse_agent_model_for_judge 없이 --judge_backend "
                "qwen(별도 로컬 judge 모델을 새로 로드)을 쓸 것."
            )

        agent_model = None
        if args.grounding_backend == "gui_actor":
            # (v3 수정 - RF 결합) 처음엔 RegionFocus가 우리 LoRA 학습 포맷 전용이라 GUI-Actor와는
            # 무조건 호환 안 된다고 보고 여기서 자동으로 껐었다. gui_actor_region_focus.py로
            # RegionFocus 알고리즘(judge/재탐색/crop-zoom/aggregation)을 GUI-Actor 위에서 재현했으므로
            # (judge/aggregation은 GUIActorModel.generate()로 하는 일반 텍스트 QA라 그대로 재사용 가능,
            # 재탐색은 GUI-Actor가 한 번의 forward pass에서 주는 topk 후보를 재사용해서 추가 호출 없이
            # 대체) 이제 --use_regionfocus를 그대로 존중한다 - 더 이상 여기서 강제로 끄지 않는다.
            #
            # reflection(비평 루프)은 여전히 기본 꺼둔다 - GUIActorModel.generate()가 기술적으로는
            # 되지만(위와 같은 근거), reflection이 요구하는 구조화된 JSON 비평 포맷은 GUI-Actor가
            # 학습에서 한 번도 못 본 형태라 출력 품질이 검증되지 않았다. planner LoRA에 tool-call
            # 포맷을 줬을 때 grounding이 깨졌던 것과 같은 종류의 리스크라, 여기서는 보수적으로
            # 자동으로 끈다 - 직접 --no_reflect 없이 켜보고 싶으면 이 if 블록을 지우면 된다.
            if args.use_reflection:
                print(
                    "[gui_actor_eval_webvoyager.py] 주의: --grounding_backend gui_actor는 reflection을 기본적으로 "
                    "끕니다(--no_reflect와 동일 효과) - GUIActorModel.generate()로 기술적으로는 가능하지만, "
                    "reflection의 구조화된 JSON 비평 포맷은 GUI-Actor가 학습에서 본 적 없어 출력 품질이 "
                    "검증되지 않았음."
                )
                args.use_reflection = False

            from gui_actor_grounding import GUIActorModel
            from api_planner import OpenAIPlannerModel

            agent_model = GUIActorModel(
                model_id=args.gui_actor_model_id,
                attn_implementation=args.gui_actor_attn_implementation,
                # (v3 수정 - 버그) 처음엔 여기서 min_pixels/max_pixels를 안 넘겨서, 기존 LoRA
                # 경로가 쓰는 --min_pixels/--max_pixels가 GUI-Actor 경로에서는 조용히 무시되고
                # GUI-Actor 자체 프로세서 기본 해상도로만 돌고 있었다. gui_actor_grounding.
                # GUIActorModel이 이제 이 값을 프로세서 로드 시점에 받아서 실제로 해상도를
                # 통제한다(gui_actor_grounding.py의 GUIActorModel.__init__ 주석 참고).
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
            planning_view = OpenAIPlannerModel(
                model=args.planner_api_model,
                api_key=args.api_key,
                base_url=args.planner_api_base_url,
            )
            print(
                f"[gui_actor_eval_webvoyager.py] grounding backend = GUI-Actor ({args.gui_actor_model_id!r}), "
                f"planner backend = OpenAI API (model={args.planner_api_model!r})"
            )

            agent_step_fn = build_planner_grounding_agent_step(
                agent_model, planning_view,
                use_reflection=args.use_reflection, max_iterations=args.max_iterations,
                ground_min_pixels=args.min_pixels, ground_max_pixels=args.max_pixels,
                debug_dir=None if args.no_debug_dump else args.debug_dir,
                debug_save_images=not args.no_debug_images,
                use_regionfocus=args.use_regionfocus,
                regionfocus_debug_image=args.regionfocus_debug_image,
                regionfocus_debug_text=args.regionfocus_debug_text,
                regionfocus_step1_format=args.regionfocus_step1_format,
                regionfocus_step4_format=args.regionfocus_step4_format,
                grounding_backend="gui_actor",
            )
            if not args.no_debug_dump:
                print(f"[gui_actor_eval_webvoyager.py] 태스크별 프롬프트/응답 덤프 경로: {os.path.abspath(args.debug_dir)}")
            print(
                "[gui_actor_eval_webvoyager.py] click grounding = GUI-Actor(coordinate-free pointer) + "
                + (
                    "RegionFocus(gui_actor_region_focus.py 재현판: judge/재탐색/crop-zoom/aggregation)"
                    if args.use_regionfocus
                    else "초기 grounding 1회만(--no_regionfocus)"
                )
            )
        elif args.agent_grounding_adapter_dir:
            model_kwargs = {}
            if args.min_pixels is not None:
                model_kwargs["min_pixels"] = args.min_pixels
            if args.max_pixels is not None:
                model_kwargs["max_pixels"] = args.max_pixels

            if args.planner_backend == "openai":
                # (2026-08-11 추가) grounding은 여전히 로컬 QwenVLModel(+grounding LoRA)이 담당 -
                # load_shared_model()을 안 쓰고 QwenVLModel을 직접 로드하는 이유는, 이 모델엔
                # planner LoRA를 얹을 필요가 없어서다(멀티 어댑터 스왑 자체가 필요 없음).
                from qwen import QwenVLModel
                from api_planner import OpenAIPlannerModel

                agent_model = QwenVLModel(adapter_dir=args.agent_grounding_adapter_dir, **model_kwargs)
                planning_view = OpenAIPlannerModel(
                    model=args.planner_api_model,
                    api_key=args.api_key,
                    base_url=args.planner_api_base_url,
                )
                print(
                    f"[gui_actor_eval_webvoyager.py] planner backend = OpenAI API (model={args.planner_api_model!r}) "
                    "- grounding은 그대로 로컬 LoRA(--agent_grounding_adapter_dir). reflection은 "
                    "기존과 동일하게 로컬 base 모델(disable_adapter)로 돈다(build_planner_grounding_"
                    "agent_step이 grounding_model 기준으로 reflection_view를 만들기 때문에 이 분기와 "
                    "무관하게 그대로 동작함)."
                )
            else:
                from agent_loop import load_shared_model

                agent_model, planning_view = load_shared_model(
                    args.agent_grounding_adapter_dir,
                    planner_adapter_dir=args.agent_planner_adapter_dir,
                    **model_kwargs,
                )

            agent_step_fn = build_planner_grounding_agent_step(
                agent_model, planning_view,
                use_reflection=args.use_reflection, max_iterations=args.max_iterations,
                ground_min_pixels=args.min_pixels, ground_max_pixels=args.max_pixels,
                debug_dir=None if args.no_debug_dump else args.debug_dir,
                debug_save_images=not args.no_debug_images,
                use_regionfocus=args.use_regionfocus,
                regionfocus_debug_image=args.regionfocus_debug_image,
                regionfocus_debug_text=args.regionfocus_debug_text,
                regionfocus_step1_format=args.regionfocus_step1_format,
                regionfocus_step4_format=args.regionfocus_step4_format,
                grounding_backend="lora",
            )
            if not args.no_debug_dump:
                print(f"[gui_actor_eval_webvoyager.py] 태스크별 프롬프트/응답 덤프 경로: {os.path.abspath(args.debug_dir)}")
            print(
                f"[gui_actor_eval_webvoyager.py] click grounding = "
                f"{'RegionFocus(재탐색+crop/zoom 정밀화)' if args.use_regionfocus else 'plain gui_grounding.ground()(초기 grounding 1회)'}"
                + (" - 꺼져 있음(--no_regionfocus)" if not args.use_regionfocus else "")
            )
        else:
            print(
                "[gui_actor_eval_webvoyager.py] 주의: --agent_grounding_adapter_dir 미지정, --grounding_backend gui_actor도 "
                "아님 -> agent_step_fn이 dummy_agent_step()(항상 즉시 종료)임. 실제 정책을 돌리려면 "
                "--agent_grounding_adapter_dir(+ 선택적으로 --agent_planner_adapter_dir) 또는 "
                "--grounding_backend gui_actor를 지정할 것."
            )
            agent_step_fn = dummy_agent_step

        if args.judge_backend == "qwen":
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
                    "[gui_actor_eval_webvoyager.py] agent 모델이 이미 로드돼 있어서 judge용 별도 모델 로드를 "
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
            judge_fn = make_openai_judge(model=args.judge_api_model, api_key=args.api_key)

        judge_max_workers = args.judge_max_workers
        if judge_max_workers is None:
            judge_max_workers = args.judge_repeats if args.judge_backend == "openai" else 1

        try:
            run_batch(
                tasks, env, agent_step_fn, judge_fn,
                max_steps=args.max_steps, judge_repeats=args.judge_repeats, out_path=args.out,
                stuck_repeat_threshold=args.stuck_repeat_threshold,
                stuck_repeat_window=args.stuck_repeat_window,
                resume=args.resume,
                judge_max_workers=judge_max_workers,
                task_indices=task_indices,
            )
        finally:
            env.close()