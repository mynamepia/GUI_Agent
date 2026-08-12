"""
planner.py

에이전트가 "다음에 뭘 할지" 결정하는 planner. base Qwen2.5-VL(grounding LoRA 없음)을
ReAct 스타일로 프롬프팅해서, 태스크 지시문 + 현재 스크린샷 + 지금까지의 행동 히스토리를
보고 다음 액션을 JSON으로 뽑는다.

[grounding LoRA와 이 planner의 관계 - 중요]
이 파일은 "어디를 클릭할지 좌표"를 내놓지 않는다. 클릭류 액션은 target_description(자연어로
뭘 클릭할지 설명)만 내놓고, 그 설명을 실제 픽셀 좌표로 바꾸는 건
gui_grounding.ground_with_regionfocus()(grounding LoRA 담당)의 몫이다. 이 파일 혼자로는
바로 실행 가능한 액션이 안 나온다 - target_description을 grounding에 넘겨서 좌표를 받아온
다음, env_webvoyager.WebVoyagerEnv.execute_action()이 기대하는 형태(coordinate 필드)로
합치는 건 agent_loop.py의 몫(다음 단계, 아직 없음).

[모델 로딩 방식]
기본은 base 모델(LoRA 없음)로 planning한다 - grounding LoRA는 이 planning 프롬프트
포맷(JSON, target_description 등)을 학습에서 본 적이 없어서 오히려 방해가 될 수 있다는
가설 때문. agent_loop.py의 실제 파이프라인에서도 planning은 항상 어댑터를 끈 채로
돈다(_BaseModelView/disable_adapter()). 다만 이 파일의 CLI(`_cli()`)는 2026-08-07부터
`--adapter_dir`를 선택적으로 받아서, "grounding LoRA를 켠 채로 planning하면 정말
방해가 되는지"를 직접 base/adapter 두 조건으로 비교 실행해볼 수 있게 열어뒀다(수동
비교/디버깅용 - agent_loop.py가 쓰는 정식 파이프라인의 기본 동작을 바꾸는 건 아님).
LoRA 어댑터 스왑(planning ↔ grounding)으로 하나의 backbone만 로드해서 쓰는 통합은
agent_loop.py가 처리한다(verifier/model.py 문서에 적어둔 멀티 어댑터 스왑 계획과 같은
방향).

[테스트 대상: WebVoyager]
MiniWob(utterance/fields로 태스크가 구조화돼 나옴)이 아니라 WebVoyager(자유 텍스트
지시문, 실제 사이트)를 기준으로 프롬프트를 짰다 - few-shot 예시도 웹 탐색 시나리오로
만들었다. 소형 모델(3B) planning은 순수 프롬프팅만으론 약할 수 있다는 게 문헌에서
반복적으로 확인되는 부분이라(UI-TARS/Lumos/Agent Distillation 등 참고), 실제로 돌려보고
실패 패턴을 구체적으로 기록해두는 게 중요함 - 안 되면 다음 단계는 Lumos/Agent
Distillation처럼 별도 planner LoRA를 소규모 trajectory로 파인튜닝하는 것.

[출력 스키마 - gui_grounding.ComputerUseTool과 다름, "좌표" 대신 "자연어 타겟 설명"]
{
  "reasoning": "<왜 이 액션을 골랐는지 - action보다 먼저 쓰게 강제해서, 결론부터 내리고
                사후정당화하는 대신 근거를 먼저 풀게 유도함. region_focus.judge_inference()의
                reason-then-ans 트릭과 동일한 원칙>",
  "action": "left_click" | "double_click" | "right_click" | "drag" | "type" | "key" | "scroll" | "wait" | "back" | "terminate",
  "target_description": "<click/drag류일 때만 - click은 뭘 클릭할지, drag는 어디서 드래그를
                          시작할지에 대한 자연어 설명. 좌표 아님!>",
  "text": "<type/key/scroll/drag일 때만 - 입력할 텍스트 / 키 이름 / 스크롤 방향("up"|"down") /
          drag의 경우 어디까지 드래그해서 놓을지에 대한 자연어 설명>",
  "status": "success" | "failure",   # terminate일 때만
  "answer": "<질문형 태스크의 최종 답변, 있으면>"  # terminate일 때만
}

[drag 액션 - 2026-08-07 추가]
AgentNet 기반 planner LoRA 학습 데이터를 만들다가(prepare_planner_dataset.py) 궤적의
3.9%가 pyautogui.moveTo()+dragTo() 조합(드래그)이었는데 이 파일의 액션 스키마에 drag가
없어서 전부 스킵되고 있었다. 새 필드를 만들지 않고 기존 target_description(드래그 시작
지점 설명)/text(드래그 끝 지점 설명) 필드를 재사용하는 쪽으로 넣었다 - _format_candidate_plan()이
이미 두 필드를 액션 종류와 무관하게 있으면 그대로 보여주도록 짜여 있어서 별도 수정 없이도
동작하고, agent_loop.py/gui_grounding 쪽에서 "drag는 두 지점을 grounding해서 좌표로
바꿔야 한다"는 처리만 나중에 추가하면 된다(아직 agent_loop.py의 실행 매핑 자체가 없는
단계라 이 파일에서는 스키마/파싱만 준비해둠).

[self-consistency 관련 메모]
plan_next_action()은 temperature 파라미터를 받지만 내부에서 여러 번 샘플링하지는 않는다
(region_focus()가 temperature를 받되 재시도는 호출부가 담당하는 것과 같은 설계). 나중에
Agent Distillation 논문에서 언급된 "여러 번 샘플링 + 다수결"로 소형 모델의 노이즈를
완화하고 싶으면, agent_loop.py 쪽에서 이 함수를 여러 번 호출해서 다수결을 취하면 된다.

[plan_with_reflection() - 실행 전 planner<->reflection 비평 루프]
관찰된 실패(할루시네이션으로 인한 조기 termination)에 대응하려고 추가. WorldGUI-Agent
논문의 Planner-Critic/Pre-Action Validation 구조, Self-Refine/Reflexion류의 반복적
자기비판 원리를 실행 "전" 단계에 적용한 것. 설계 원칙 두 가지:
  1. reflection도 반드시 이미지를 봐야 함 - 이번 실패가 "화면 상태에 대한 잘못된 주장"이라서,
     텍스트(reasoning)만 보는 reflection은 애초에 그 주장이 맞는지 검증할 방법이 없음.
  2. 다만 같은 모델/같은 vision encoder가 같은 이미지를 다시 봐도 똑같이 잘못 읽을 위험
     (Self-Correction Mirage)이 있어서, 두 가지로 완화함: (a) reflection이 planner의 주장을
     그대로 확인하는 게 아니라 이미지를 보고 독립적으로 "지금 뭐가 보이는지"부터 먼저 서술하게
     강제(_REFLECTION_SYSTEM_PROMPT의 observation 필드) 한 뒤에 그걸 candidate plan과
     대조시킴, (b) reflection 호출의 temperature를 planner보다 높게 줘서 동일한 샘플링
     경로를 그대로 반복할 확률을 낮춤.
승인 안 되면 critique를 planner에게 다시 넣어서(revision_context) 재시도, max_iterations
번 반복해도 승인이 안 나면 마지막 후보를 그대로 반환하되 `_reflection_approved: False`로
플래그만 남김(무한루프 방지 + 조용히 막지 않고 호출부가 판단하게 하는 기존 폴백 원칙과 동일).
max_iterations 기본값은 2("1차 시도 + 반려시 1번만 수정 재시도") - agent_loop.py에서 스텝마다
이 루프를 그대로 쓸 예정이라 여기서 기본값을 정해두되 인자로 override 가능하게 열어둠.

[reflection_model 파라미터 - 제안과 비평을 각자 다른 모델로]
plan_with_reflection()은 내부에서 plan_next_action()(제안)과 _reflect_on_plan()(비평)을
호출하는데, reflection_model을 통해 이 둘을 서로 다른 모델 객체로 돌릴 수 있다 - 예를
들어 제안은 planner 전용 LoRA를 얹은 모델로, 비평은 reflector 전용 LoRA를 얹은(또는
LoRA 없는 base) 모델로 나누는 식. reflection_model을 지정하지 않으면(기본값 None) 예전
방식 그대로 qwen_model 하나로 제안/비평을 둘 다 돌린다(하위 호환).

이 파일은 qwen_model/reflection_model이 실제로 어댑터를 얹었는지 base인지 전혀 모르고
신경 쓰지도 않는다 - duck-typing으로 .generate(messages, max_new_tokens=..., temperature=...)
만 있으면 그대로 동작한다. 즉 "제안/비평 각각 자기 어댑터를 쓸 수도, 어댑터 없이 base로
돌 수도 있다"는 요구사항은 이 파일이 직접 구현하는 게 아니라, 호출부(agent_loop.py)가
qwen_model/reflection_model에 뭘 넘기느냐로 결정된다 - 이 파일은 그 결정과 완전히
무관하게 항상 똑같이 동작한다(이 파일이 어댑터 유무를 분기 처리할 필요가 없다는 뜻).

[reflection 프롬프트 강화 - 1차 버전에서 빠졌던 것들]
1차 버전은 "모순이 있으면 반려"라는 수동적 비교 위주였고, terminate/click에만 명시적
반려 기준이 있었음. 강화하면서 추가한 것: (1) type/key/scroll에도 액션별 반려 기준 추가,
(2) click 기준을 "화면에 그럴듯하게 존재"에서 "정확히 하나로 식별 가능"으로 강화,
(3) history를 보고 "방금 실패한 것과 같은 액션 반복"을 반려 신호로 체크하도록 추가,
(4) "이 액션이 틀렸을 이유를 최소 하나 찾아보라"(possible_failure_reason 필드)는 능동적
devil's-advocate 프레이밍 추가 - "reasoning이 그럴듯해 보이면 통과"가 아니라 "스크린샷만이
증거"라는 원칙을 시스템 프롬프트 서두에 명시.
"""

from __future__ import annotations  # 아래 QwenVLModel 타입 힌트를 lazy하게 만들어 런타임 임포트 회피

import json
import re
from typing import TYPE_CHECKING

# vlm_agent(qwen.py가 있는 폴더)를 sys.path에 넣는다 - hypo1/verifier 스크립트들과 동일한
# 패턴. 이걸 안 넣어두면 `from qwen import ...`(위 TYPE_CHECKING 블록/아래 _cli() 양쪽)가
# "이 스크립트를 어느 cwd/방식으로 실행했느냐"에 따라 조용히 ModuleNotFoundError가 날 수
# 있다 - agent/가 vlm_agent 바로 밑(현재 구조)이든 vlm_agent와 형제 폴더로 옮겨지든 둘 다
# 자동으로 찾는다.
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _candidate in (_os.path.join(_HERE, ".."), _os.path.join(_HERE, "..", "vlm_agent")):
    _candidate = _os.path.abspath(_candidate)
    if _os.path.isfile(_os.path.join(_candidate, "qwen.py")):
        if _candidate not in _sys.path:
            _sys.path.insert(0, _candidate)
        break

if TYPE_CHECKING:
    # 실행에는 필요 없고 타입 힌트용 - qwen.py(torch/transformers 등 무거운 의존성)를
    # 이 selftest 경로에서까지 강제로 임포트하지 않기 위해 TYPE_CHECKING 가드로 묶어둠.
    # (2026-08-07) 상대 import(from ..qwen import ...)는 vlm_agent/, vlm_agent/agent/에
    # __init__.py가 없어서 `python agent/planner.py`처럼 스크립트로 직접 실행하면
    # "attempted relative import with no known parent package"로 깨진다. agent_loop.py가
    # 이미 절대 import(`from qwen import QwenVLModel`, 스크립트를 vlm_agent/ 안에서 직접
    # 실행하는 걸 전제)를 쓰고 있어서, 여기도 그 스타일로 통일한다.
    from qwen import QwenVLModel

_ACTIONS = ("left_click", "double_click", "right_click", "drag", "type", "key", "scroll", "wait", "back", "terminate")

_FEWSHOT = """
Example:
Task: "Find the price of the cheapest flight from Seoul to Tokyo next Monday on Google Flights."
History:
  Step 1: left_click on "the departure city input box" -> typed "Seoul"
  Step 2: left_click on "the destination city input box" -> typed "Tokyo"
Current screenshot shows a search results page with a list of flights and prices.
{"reasoning": "The search results are visible and I can see the lowest price listed at the top of the sorted list.", "action": "terminate", "status": "success", "answer": "The cheapest flight is $210."}
""".strip()

_SYSTEM_PROMPT = f"""You are a web browsing agent. You are given an overall task, the current
screenshot of the browser, and a history of actions you have already taken. Decide the single
next action to make progress toward completing the task.

Available actions: {", ".join(_ACTIONS)}
- left_click / double_click / right_click: needs "target_description" (a short natural-language
  description of the UI element to interact with, NOT coordinates - e.g. "the search button" or
  "the first result link").
- drag: needs "target_description" (a short natural-language description of WHERE THE DRAG
  STARTS, NOT coordinates - e.g. "the fill handle at the bottom-right corner of cell A2") and
  "text" (a short natural-language description of WHERE THE DRAG ENDS - e.g. "cell A13").
- type: needs "text" (the string to type into the currently focused input).
- key: needs "text" (a key name, e.g. "Enter", "Tab", "Escape").
- scroll: needs "text" ("up" or "down").
- wait: no extra fields needed (use sparingly, only if the page seems to be loading).
- back: no extra fields needed. Goes back to the previous page in browser history. Use this when
  you navigated to the wrong page, opened something unintended, or need to undo a navigation -
  prefer this over repeatedly clicking around trying to find a way back.
- terminate: use this when the task is complete or you are stuck. Needs "status" ("success" or
  "failure") and, if the task asked a question, "answer" with your final answer text.

{_FEWSHOT}

Before deciding, check the action history: if you have already tried the same or a very similar
action recently without making progress, do not propose it again - repeating an ineffective action
will not suddenly start working. Instead actively look for a different approach (a different UI
element, a different navigation path such as a filters/sidebar panel instead of scrolling further,
or reconsidering whether your assumption about the current screen is correct). If the history shows
you are stuck in this way, it will be called out explicitly with a "REPETITION WARNING" - treat that
as a hard requirement to change strategy, not a suggestion.

Reply with ONLY a single JSON object with these fields (omit fields that don't apply to your
chosen action): {{"reasoning": "...", "action": "...", "target_description": "...", "text": "...",
"status": "...", "answer": "..."}}
Think through your reasoning first, then decide the action - write "reasoning" before "action" in
your JSON.
Always write all free-text field values ("reasoning", "target_description", "answer", etc.) in
English, even if the task description is given in another language.
"""

_REFLECTION_SYSTEM_PROMPT = """You are a skeptical, adversarial reviewer checking another agent's
proposed next action before it is executed on a real website. Default to assuming the proposer is
wrong until your own inspection of the screenshot convinces you otherwise - do not give it the
benefit of the doubt just because its reasoning sounds coherent. Coherent-sounding reasoning is not
evidence; only the screenshot is evidence.

You are given the overall task, the current screenshot, the action history so far, and the
proposed action (with its stated reasoning).

Your job, in order:
1. Independently describe what you actually observe in the screenshot that is relevant to the
   task - do this BEFORE looking at whether the proposed action agrees with it. Do not simply
   restate the proposer's reasoning; look at the image yourself and name concrete elements you
   actually see (not vague descriptions like "the page looks fine").
2. Before judging, actively try to find at least one concrete reason the proposed action could be
   wrong. State it even if you end up approving anyway - if you genuinely cannot find any plausible
   failure reason after looking, say so explicitly rather than skipping this step.
3. Check the action history for repetition: if the proposed action is the same as, or very similar
   to (same target/text/key), an action already attempted recently, treat this as a strong signal
   to reject - repeating an action that didn't already finish the task rarely produces a different
   result the second time.
4. Apply these action-specific checks:
   - terminate: reject unless your own observation clearly and independently confirms the claimed
     status/answer. A plausible-sounding claim by itself is not enough - you must be able to point
     to something specific you see that confirms it.
   - left_click / double_click / right_click: reject unless target_description maps to exactly one
     clearly identifiable element in your observation. If it's ambiguous (could match more than one
     element) or you cannot locate it at all, reject.
   - drag: reject unless BOTH target_description (drag start) and text (drag end) map to exactly
     one clearly identifiable element/location each in your observation. If either endpoint is
     ambiguous or not visible, reject.
   - type: reject unless your observation shows a text input area that is actually focused/active
     and appropriate for this text.
   - key: reject if the key press assumes a UI state (a focused field, an open menu, etc.) that is
     not visible in your observation.
   - scroll: reject if the content the reasoning is looking for already appears fully visible in
     the current screenshot, or if the scroll direction doesn't match where the reasoning says that
     content should be.
   - wait: approve only if there is a visible sign of loading/transition (spinner, blank or
     partially-rendered page); reject if the page already looks fully loaded and static.

When in doubt, reject - a false rejection just costs one extra planning step, but a false approval
executes a wrong action on a real website.

Reply with ONLY a single JSON object: {"observation": "<what you see, written before judging>",
"possible_failure_reason": "<at least one concrete reason this action could be wrong, even if you
approve anyway>", "approved": true or false, "critique": "<if not approved, a specific, actionable
reason the proposer can use to revise - otherwise empty string>"}
Always write all free-text field values ("observation", "possible_failure_reason", "critique") in
English, even if the task description is given in another language.
"""


# (2026-08-11 추가) 이 이상 "사실상 같은" 액션이 나오면 _format_history()가 경고 문구를
# 끼워 넣는다. eval_webvoyager_v2.py의 stuck_repeat_threshold(하드 조기종료, 기본 5)보다
# 낮게 잡아서 - 하드 조기종료로 에피소드 자체가 끊기기 전에 planner가 먼저 스스로 전략을
# 바꿀 기회를 준다. 사용자 요청대로 3회로 설정(= 4번째 제안 시점에 경고가 뜸).
REPEAT_WARNING_STREAK = 3

# (2026-08-11 수정 - 뺑뺑이/오실레이션 탐지) 원래는 "바로 직전과 연속으로 같은가"만 봤는데,
# 실측 우려대로 이러면 A -> B -> A -> B처럼 두세 개 액션을 번갈아 반복하는 패턴을 못 잡는다
# (매번 직전 액션과 다르므로 streak가 계속 1로 리셋됨). 그래서 "바로 직전 연속"이 아니라
# "최근 REPEAT_WARNING_WINDOW개 안에 같은 시도가 몇 번 있었는가"로 바꿨다 - 순서 상관없이
# 빈도만 본다. 윈도우 크기는 임계값의 2배로 잡아서, A/B 두 액션이 절반씩 번갈아 나오는
# 최악의 경우(2-사이클)도 윈도우가 다 차면 반드시 걸리도록 함(예: threshold=3, window=6이면
# A,B,A,B,A,B 상황에서 마지막 액션 기준 카운트가 3에 도달).
REPEAT_WARNING_WINDOW = REPEAT_WARNING_STREAK * 2


def _describe_action(a: dict) -> str:
    """액션 dict 하나를 사람이 읽는 한 줄 설명으로 바꾼다. _format_history()의 히스토리
    라인 렌더링과 반복 경고 문구 둘 다 이 설명을 재사용한다(같은 액션을 두 군데서 다르게
    묘사하면 "REPETITION WARNING에 적힌 그 행동"과 "history 줄에 적힌 그 행동"이 서로 다른
    문구로 보여서 모델이 둘을 같은 것으로 못 알아볼 위험이 있음)."""
    act = a.get("action")
    if act in ("left_click", "double_click", "right_click"):
        return f'{act} on "{a.get("target_description", "?")}"'
    if act == "drag":
        return f'drag from "{a.get("target_description", "?")}" to "{a.get("text", "?")}"'
    if act == "type":
        return f'type "{a.get("text", "")}"'
    if act == "key":
        return f'press key "{a.get("text", "")}"'
    if act == "scroll":
        return f'scroll {a.get("text", "down")}'
    if act == "wait":
        return "wait"
    if act == "back":
        return "go back to the previous page"
    return str(act)


def _is_similar_action(a: dict, b: dict) -> bool:
    """
    (2026-08-11 추가 - 반복 행동 경고) 두 액션이 "사실상 같은 시도"인지 비교한다.
    eval_webvoyager_v2._action_fingerprint()와 같은 목적이지만, 이 파일의 planner 스키마는
    좌표가 없고 target_description/text(자연어)만 있어서 좌표 버킷 비교 대신 문자열을
    대소문자 무시하고 비교하는 것으로 충분하다 - 같은 문구로 같은 요소/텍스트를 반복
    지목한다는 건 사실상 같은 시도라는 뜻이라서. _rejected 마커 유무는 비교에서 무시한다 -
    "시도했지만 반려당함"도 "같은 걸 또 시도하려 한다"는 패턴의 일부로 쳐야 함.
    """
    if a.get("action") != b.get("action"):
        return False
    act = a.get("action")
    if act in ("left_click", "double_click", "right_click"):
        return (a.get("target_description") or "").strip().lower() == (b.get("target_description") or "").strip().lower()
    if act == "drag":
        return (
            (a.get("target_description") or "").strip().lower() == (b.get("target_description") or "").strip().lower()
            and (a.get("text") or "").strip().lower() == (b.get("text") or "").strip().lower()
        )
    if act in ("type", "key"):
        return (a.get("text") or "").strip().lower() == (b.get("text") or "").strip().lower()
    if act == "scroll":
        return (a.get("text") or "down").strip().lower() == (b.get("text") or "down").strip().lower()
    if act == "wait":
        return True
    if act == "back":
        return True
    return False


def _trailing_repeat_streak(history_actions: list):
    """
    (2026-08-11 추가, 같은 날 -윈도우 방식으로 대체됨) history_actions 끝에서부터 거꾸로
    훑어서 "사실상 같은" 액션이 몇 번 연속됐는지 센다. 도중에 다른 액션이 하나라도 끼면 그
    지점에서 멈춘다(= 최신 연속 구간만 본다).

    [한계 - 왜 _format_history()가 이제 이 함수 대신 _windowed_repeat_count()를 쓰는가]
    이 "연속(streak)" 방식은 A -> B -> A -> B처럼 두세 개 액션을 번갈아 반복하는 뺑뺑이
    패턴을 못 잡는다 - 매 스텝 직전 액션과 다르므로 streak가 계속 1로 리셋되기 때문. 실측
    우려(WebVoyager 실행에서 검색 버튼을 못 찾고 이것저것 번갈아 시도하며 진행이 없는 케이스)
    로 이 한계가 지적되어 _windowed_repeat_count()로 교체했다. 이 함수 자체는 하위호환/단독
    테스트 목적으로 남겨둔다.

    Returns: (streak: int, action: dict | None) - streak는 마지막 액션 자신을 포함한 개수
        (즉 streak=1이면 반복 없음, streak=3이면 마지막 3개가 전부 같은 시도). history_actions가
        비어있으면 (0, None).
    """
    if not history_actions:
        return 0, None
    last = history_actions[-1]
    streak = 1
    for prev in reversed(history_actions[:-1]):
        if _is_similar_action(prev, last):
            streak += 1
        else:
            break
    return streak, last


def _windowed_repeat_count(history_actions: list, window_size: int):
    """
    (2026-08-11 추가 - 뺑뺑이/오실레이션 탐지) 최근 window_size개 액션 중, 마지막 액션과
    "사실상 같은" 시도가 몇 번 있었는지 순서 무관하게 센다(빈도 기반). _trailing_repeat_streak()
    와 달리 중간에 다른 액션이 끼어도 카운트가 리셋되지 않는다 - 그래서 A -> B -> A -> B처럼
    번갈아 반복하는 패턴도, 같은 시도가 윈도우 안에서 threshold번 이상 나오면 잡힌다.

    Returns: (count: int, action: dict | None) - count는 마지막 액션 자신을 포함한 개수.
        history_actions가 비어있으면 (0, None).
    """
    if not history_actions:
        return 0, None
    last = history_actions[-1]
    window = history_actions[-window_size:] if window_size else history_actions
    count = sum(1 for a in window if _is_similar_action(a, last))
    return count, last


def _format_history(
    history_actions, max_items=8, repeat_warning_streak=REPEAT_WARNING_STREAK,
    repeat_warning_window=None,
):
    """
    과거 액션들을 텍스트로 요약. 컨텍스트/연산량을 아끼려고 과거 스크린샷은 다시 안 넣고
    텍스트 요약만 준다 - 3B 모델 + 16GB 환경에서 매 스텝마다 이미지를 계속 누적해서
    프롬프트에 넣으면 금방 무거워짐. 최근 max_items개만 남기고 그 이전은 생략 문구로 처리.

    [2026-08-10 추가 - 반려된(실행 안 된) 액션 표시]
    호출부(예: eval_webvoyager.py의 plan_with_reflection 연동)가 reflection에게 끝까지
    반려당해서 실제로 실행은 안 한 액션을, "실행 안 됐다"는 사실을 숨기지 않고 history에
    남기고 싶을 때를 위한 것. 이런 항목은 `_rejected: True`(+ 선택적으로
    `_rejection_reason`)를 넣어서 넘기면 된다. 이게 필요해진 이유: 처음엔 반려된 액션을
    history에서 아예 빼버렸는데(실행 안 됐으니 "일어난 일"처럼 보이면 안 되니까), 그러면
    다음 스텝에서 모델이 "내가 방금 이걸 시도했다가 거부당했다"는 걸 전혀 기억 못 하고
    똑같은 화면을 보고 똑같은 액션을 계속 재제안하는 문제가 실측으로 나왔다(reflection의
    critique가 스텝을 넘어가면 완전히 유실됨). 그래서 "안 일어난 일"이라는 걸 명확히
    구분해서 보여주는 절충안으로 바꿨다 - "이미 했다"도 아니고 "아예 기록에 없다"도 아니고,
    "시도했지만 반려당했다(+이유)"를 그대로 알려줘서 모델이 같은 실수를 반복하지 않고 다른
    선택지를 찾도록 유도한다.

    [2026-08-11 추가 - 반복 행동 경고]
    위 _rejected 마커는 "reflection이 실행 전에 막은" 경우만 커버한다. 실측으로 더 흔한
    실패 패턴은 reflection이 꺼져 있거나 승인해준 액션(예: scroll down)을 여러 스텝에 걸쳐
    계속 반복하면서 진전이 없는 경우였다(WebVoyager 평가에서 필터를 못 찾고 계속
    스크롤하다가 eval_webvoyager_v2.py의 stuck_repeat_threshold에 걸려 조기종료된 사례
    다수 - idx 4/6/9). 그 하드 조기종료는 "더 못 하게 막는" 안전장치일 뿐 "다른 걸
    시도해보라"고 알려주진 않는다. 여기서는 최근 repeat_warning_window개 액션 안에 같은
    시도가 repeat_warning_streak번 이상 있으면(_windowed_repeat_count) 명시적인 경고 문구를
    history 텍스트 끝에 붙여서, 하드 조기종료가 걸리기 전에 planner가 스스로 전략을
    바꿀 기회를 준다.

    [2026-08-11 추가 - 연속(streak) 대신 윈도우(window) 방식으로 변경]
    처음엔 "바로 직전 액션과 연속으로 같은가"만 봤는데, 이러면 A -> B -> A -> B처럼 두세 개
    액션을 번갈아 반복하는 뺑뺑이 패턴을 못 잡는다(매번 직전과 다르므로 streak가 계속
    리셋됨). 그래서 "직전과 연속"이 아니라 "최근 N개 안에서의 빈도"로 바꿔서, 순서와
    무관하게 같은 시도가 반복되면 잡히도록 했다.
    """
    if not history_actions:
        return "(no actions taken yet)"
    shown = history_actions[-max_items:]
    lines = []
    if len(history_actions) > max_items:
        lines.append(f"...({len(history_actions) - max_items}개 이전 스텝 생략)")
    start_idx = len(history_actions) - len(shown) + 1
    for i, a in enumerate(shown, start=start_idx):
        desc = _describe_action(a)
        if a.get("_rejected"):
            reason = a.get("_rejection_reason") or ""
            reason_part = f' Reviewer said: "{reason}"' if reason else ""
            lines.append(
                f'Step {i}: ATTEMPTED "{desc}" but a reviewer REJECTED it before execution - '
                f"nothing changed on screen, this did NOT happen.{reason_part}"
            )
        else:
            lines.append(f"Step {i}: {desc}")
    text = "\n".join(lines)

    if repeat_warning_window is None:
        repeat_warning_window = repeat_warning_streak * 2
    count, last_action = _windowed_repeat_count(history_actions, repeat_warning_window)
    if count >= repeat_warning_streak and last_action is not None:
        desc = _describe_action(last_action)
        text += (
            f'\n\n*** REPETITION WARNING: you have proposed the SAME action ("{desc}") {count} times '
            f"within your last {repeat_warning_window} attempts and it has NOT made progress. "
            "Do NOT propose it again. Actively try a genuinely "
            "different approach instead - for example: look for a different UI element or navigation "
            "path (e.g. a filters/sidebar panel instead of scrolling further, a different button or "
            "link), reconsider whether your assumption about what's currently on screen is correct, or "
            "change direction/strategy entirely. If after trying different approaches this still seems "
            'impossible, terminate with status "failure" rather than repeating the same action again. ***'
        )
    return text


def _action_schema_valid(obj: dict) -> bool:
    """
    (2026-08-07 추가) action 이름만 _ACTIONS에 속하는지 보는 것만으론 부족했다 -
    실측으로 확인된 사례: grounding LoRA(어댑터)를 켠 채로 planning을 시키면, 자기가
    학습받은 좌표 tool-call 포맷 그대로 {"action": "left_click", "coordinate": [x, y]}
    를 내놓는데, "left_click"이 우연히 planner의 액션 이름과도 겹쳐서 _parse_planner_action이
    이걸 "정상 파싱"으로 잘못 받아들이고 있었다 - target_description이 아예 없는데도
    통과됨. action별 필수 필드까지 확인해야 이런 스키마 불일치(포맷은 다른데 action
    이름만 우연히 같은 경우)를 제대로 걸러낸다.
    """
    action = obj.get("action")
    if action in ("left_click", "double_click", "right_click"):
        target = obj.get("target_description")
        return isinstance(target, str) and target.strip() != ""
    if action == "drag":
        # (2026-08-07 추가) drag는 target_description(시작 지점)과 text(끝 지점) 둘 다
        # 있어야 함 - 둘 중 하나라도 없으면 어디서 어디로 드래그하는지 알 수 없음.
        target = obj.get("target_description")
        text = obj.get("text")
        return (
            isinstance(target, str) and target.strip() != ""
            and isinstance(text, str) and text.strip() != ""
        )
    if action in ("type", "key"):
        text = obj.get("text")
        return isinstance(text, str) and text.strip() != ""
    if action == "scroll":
        return obj.get("text") in ("up", "down")
    if action == "wait":
        return True
    if action == "back":
        return True
    if action == "terminate":
        return obj.get("status") in ("success", "failure")
    return False


def _parse_planner_action(response_text: str):
    """
    region_focus._parse_judge_verdict()와 같은 원칙: JSON 우선 파싱, action이 알려진
    값이 아니거나 파싱 자체가 실패하면 안전한 기본값(terminate/failure)으로 폴백해서
    조용히 잘못된 액션이 실행되는 걸 막는다. action 이름 확인뿐 아니라 action별 필수
    필드(_action_schema_valid)까지 통과해야 정상 파싱으로 인정한다.
    """
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if obj.get("action") in _ACTIONS and _action_schema_valid(obj):
                return obj
        except (json.JSONDecodeError, AttributeError):
            pass
    return {
        "reasoning": "(parse failure, unknown action, or missing required field - falling back to safe terminate)",
        "action": "terminate",
        "status": "failure",
        "answer": None,
        "_parse_failed": True,
        "_raw_response": response_text,
    }


def _format_candidate_plan(plan: dict) -> str:
    """reflection 프롬프트에 넣을 후보 plan 요약. plan_next_action()의 출력 스키마를 그대로 받는다."""
    parts = [f'action: {plan.get("action")}']
    if plan.get("target_description"):
        parts.append(f'target_description: "{plan.get("target_description")}"')
    if plan.get("text") is not None and plan.get("text") != "":
        parts.append(f'text: "{plan.get("text")}"')
    if plan.get("action") == "terminate":
        parts.append(f'status: {plan.get("status")}')
        if plan.get("answer"):
            parts.append(f'answer: "{plan.get("answer")}"')
    parts.append(f'reasoning given by proposer: "{plan.get("reasoning", "")}"')
    return "\n".join(parts)


def _parse_reflection_verdict(response_text: str) -> dict:
    """
    _parse_planner_action()과 같은 원칙(JSON 우선, 실패시 안전한 기본값)이되, 여기서는
    "안전한 기본값"이 approved=False임 - 파싱이 안 되거나 애매하면 일단 반려시키고
    재시도 한 번을 더 태우는 게, 애매한 걸 그냥 승인해서 실행해버리는 것보다 안전함.
    """
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj.get("approved"), bool):
                obj.setdefault("observation", "")
                obj.setdefault("possible_failure_reason", "")
                obj.setdefault("critique", "")
                return obj
        except (json.JSONDecodeError, AttributeError):
            pass
    return {
        "observation": "",
        "possible_failure_reason": "",
        "approved": False,
        "critique": "(reflection response failed to parse - falling back to safe rejection)",
        "_parse_failed": True,
        "_raw_response": response_text,
    }


def plan_next_action(
    qwen_model: QwenVLModel,
    task_instruction: str,
    screenshot,
    history_actions: list | None = None,
    max_new_tokens: int = 300,
    temperature: float = 0.0,
    revision_context: dict | None = None,
) -> dict:
    """
    다음 액션을 결정한다.

    Args:
        qwen_model: base 모델(LoRA 없음) 인스턴스. agent_loop.py가 나중에 어댑터를
            스왑해서 넘길 수도 있지만, 지금은 항상 base로만 씀.
        task_instruction: 전체 태스크 지시문 (WebVoyager의 "ques" 필드).
        screenshot: 현재 화면 (PIL.Image).
        history_actions: 지금까지 이 함수가 반환했던 action dict들의 리스트(호출부가 누적).
        revision_context: plan_with_reflection()이 반려된 후보를 재시도시킬 때만 씀.
            {"previous_plan": <이전 후보 dict>, "critique": "<reflection이 준 반려 사유>"}
            형태. None이면(기본) 첫 시도와 동일하게 동작.

    Returns: 파일 상단 docstring의 출력 스키마를 따르는 dict. 파싱 실패시 안전하게
        {"action": "terminate", "status": "failure", ...}를 반환한다.
    """
    history_text = _format_history(history_actions or [])
    user_text = (
        f'Task: "{task_instruction}"\n\n'
        f"History:\n{history_text}\n\n"
        "The attached image is the current screenshot. What is the next action?"
    )
    if revision_context:
        prev = _format_candidate_plan(revision_context["previous_plan"])
        user_text += (
            "\n\nYour previous proposed action was reviewed and REJECTED before execution:\n"
            f"{prev}\n"
            f'Reviewer critique: "{revision_context["critique"]}"\n'
            "Reconsider the current screenshot and propose a revised action that addresses this "
            "critique. Do not just repeat the same reasoning."
        )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    response = qwen_model.generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    return _parse_planner_action(response)


def _reflect_on_plan(
    qwen_model: QwenVLModel,
    task_instruction: str,
    screenshot,
    history_actions: list | None,
    candidate_plan: dict,
    max_new_tokens: int = 250,
    temperature: float = 0.4,
) -> dict:
    """
    candidate_plan 하나를 실행 전에 비평한다. planner와 별개 호출(system prompt가 다름) -
    _REFLECTION_SYSTEM_PROMPT가 "네 판단부터 독립적으로 써라"를 강제함. 여기 넘어오는
    qwen_model은 plan_with_reflection()이 reflection_model(지정 안 하면 제안과 동일한
    모델)을 그대로 전달한 것 - 이 함수 자체는 그 모델이 base인지 어떤 LoRA를 얹었는지
    모른다(duck-typing).

    Returns: {"observation": str, "approved": bool, "critique": str, ...}
    """
    history_text = _format_history(history_actions or [])
    user_text = (
        f'Task: "{task_instruction}"\n\n'
        f"History:\n{history_text}\n\n"
        "The attached image is the current screenshot.\n\n"
        f"Proposed next action:\n{_format_candidate_plan(candidate_plan)}\n\n"
        "Review this proposed action."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _REFLECTION_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    response = qwen_model.generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    return _parse_reflection_verdict(response)


def plan_with_reflection(
    qwen_model: QwenVLModel,
    task_instruction: str,
    screenshot,
    history_actions: list | None = None,
    max_new_tokens: int = 300,
    planner_temperature: float = 0.0,
    reflection_temperature: float = 0.4,
    max_iterations: int = 2,
    reflection_model: "QwenVLModel | None" = None,
) -> dict:
    """
    plan_next_action() <-> _reflect_on_plan() 루프. 실행 전에 후보 액션을 reflection이
    검토해서, 승인되면 그 액션을 반환하고 반려되면 critique를 planner에 되먹여 재시도한다.

    Args:
        qwen_model: 제안(plan_next_action) 호출에 쓸 모델.
        reflection_model: 비평(_reflect_on_plan) 호출에 쓸 모델. None이면(기본값)
            qwen_model을 그대로 재사용한다 - 제안/비평 둘 다 같은 모델 하나로 도는
            동작(하위 호환). 제안용 어댑터와 비평용 어댑터를 따로 쓰고 싶을 때(또는 둘 중
            하나만 어댑터가 있고 나머지는 base일 때)만 명시적으로 넘기면 된다 - 이 함수는
            qwen_model/reflection_model이 어댑터를 얹었는지 base인지 전혀 모른다.

    max_iterations 기본값은 2 - agent_loop.py에서 스텝마다 이 루프를 돌릴 때 "1차 시도 +
    반려시 1번만 수정 재시도"로 쓰기로 합의한 값. agent_loop.py 쪽에서 이 인자로 override
    가능하니 이 파일에 하드코딩하지 않고 파라미터로 남겨둠.

    max_iterations번 돌아도 승인이 안 나면 마지막 후보를 그대로 반환하되
    `_reflection_approved: False`로 표시만 하고 강제로 막지는 않는다(호출부/agent_loop.py가
    이 플래그를 보고 그냥 실행할지, 로그만 남기고 스킵할지 결정하게 함 - 이 파일이 임의로
    "무한 반려 -> 그냥 종료" 정책까지 정해버리면 오히려 디버깅하기 더 어려워짐).

    Returns: plan_next_action()과 같은 스키마 + 다음 필드 추가:
        "_reflection_approved": bool
        "_reflection_log": [{"iteration": int, "plan": dict, "verdict": dict}, ...]
    """
    reflect_qwen_model = reflection_model if reflection_model is not None else qwen_model

    log = []
    candidate = None
    verdict = None
    for i in range(1, max_iterations + 1):
        revision_context = None
        if i > 1:
            revision_context = {"previous_plan": candidate, "critique": verdict.get("critique", "")}
        candidate = plan_next_action(
            qwen_model,
            task_instruction,
            screenshot,
            history_actions=history_actions,
            max_new_tokens=max_new_tokens,
            temperature=planner_temperature,
            revision_context=revision_context,
        )
        # terminate/파싱실패 폴백은 이미 그 자체로 "안전 종료"라 reflection 없이 바로 통과시킴 -
        # 애초에 reflection이 잡으려는 건 "그럴듯하게 성공했다고 우기는" 케이스지, 파싱
        # 실패로 인한 명시적 실패 폴백이 아님.
        if candidate.get("_parse_failed"):
            candidate["_reflection_approved"] = False
            candidate["_reflection_log"] = log
            return candidate

        verdict = _reflect_on_plan(
            reflect_qwen_model,
            task_instruction,
            screenshot,
            history_actions,
            candidate,
            temperature=reflection_temperature,
        )
        # (2026-08-07 버그 수정) candidate를 그대로(같은 dict 객체 참조로) log에 넣으면,
        # 승인/소진 경로에서 밑에 candidate["_reflection_log"] = log를 실행하는 순간
        # candidate -> _reflection_log -> log[-1]["plan"] -> candidate로 되돌아오는 순환
        # 참조가 생겨서 json.dumps()가 "Circular reference detected"로 죽는다(실측 확인됨).
        # log에는 이 시점까지의 candidate 스냅샷(얕은 복사)만 남기면 이후 candidate에
        # 필드를 추가로 붙여도 log 안의 항목은 영향받지 않는다.
        log.append({"iteration": i, "plan": dict(candidate), "verdict": verdict})
        if verdict.get("approved"):
            candidate["_reflection_approved"] = True
            candidate["_reflection_log"] = log
            return candidate

    candidate["_reflection_approved"] = False
    candidate["_reflection_log"] = log
    return candidate


# ---------------------------------------------------------------------------
# mock 기반 단위 테스트 (실제 모델 없이 프롬프트 조립/파싱 로직만 검증)
# ---------------------------------------------------------------------------
def _run_mock_selftest():
    """`python planner.py --selftest`"""
    from unittest.mock import MagicMock

    from PIL import Image

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    # _format_history
    check("빈 히스토리", _format_history([]) == "(no actions taken yet)")
    hist = [
        {"action": "left_click", "target_description": "search box"},
        {"action": "type", "text": "python"},
    ]
    formatted = _format_history(hist)
    check("클릭 히스토리 포맷", 'left_click on "search box"' in formatted)
    check("type 히스토리 포맷", 'type "python"' in formatted)

    long_hist = [{"action": "wait"} for _ in range(12)]
    formatted_long = _format_history(long_hist, max_items=8)
    check("긴 히스토리 -> 생략 문구 포함", "생략" in formatted_long)
    check("긴 히스토리 -> 최근 8개만 표시", formatted_long.count("Step") == 8)

    # _parse_planner_action
    good = _parse_planner_action('{"reasoning": "r", "action": "left_click", "target_description": "X"}')
    check("정상 JSON 파싱", good["action"] == "left_click" and good["target_description"] == "X")

    bad = _parse_planner_action("this is not json at all")
    check("파싱 실패 -> terminate/failure 폴백", bad["action"] == "terminate" and bad["status"] == "failure")
    check("파싱 실패 -> _parse_failed 플래그", bad.get("_parse_failed") is True)

    unknown_action = _parse_planner_action('{"action": "fly_away"}')
    check("모르는 action -> 폴백", unknown_action["action"] == "terminate")

    # (2026-08-07 회귀 테스트) grounding LoRA를 켠 채로 planning시켰을 때 실측된 사례:
    # 좌표 tool-call 포맷({"action": "left_click", "coordinate": [x, y]})이 action 이름만
    # 우연히 겹쳐서 "정상 파싱"으로 잘못 통과되면 안 된다 - target_description이 없으면 폴백.
    coord_format_leak = _parse_planner_action('{"action": "left_click", "coordinate": [739, 465]}')
    check("좌표 tool-call 포맷 누수 -> target_description 없어서 폴백", coord_format_leak["action"] == "terminate")
    check("좌표 tool-call 포맷 누수 -> _parse_failed 플래그", coord_format_leak.get("_parse_failed") is True)

    missing_text = _parse_planner_action('{"action": "type"}')
    check("type인데 text 없음 -> 폴백", missing_text["action"] == "terminate")

    bad_scroll = _parse_planner_action('{"action": "scroll", "text": "sideways"}')
    check("scroll인데 up/down이 아님 -> 폴백", bad_scroll["action"] == "terminate")

    ok_wait = _parse_planner_action('{"action": "wait"}')
    check("wait는 추가 필드 없어도 통과", ok_wait["action"] == "wait" and not ok_wait.get("_parse_failed"))

    # (2026-08-11 추가) back - 브라우저 히스토리 뒤로가기, wait처럼 추가 필드 없어도 통과해야 함
    ok_back = _parse_planner_action('{"reasoning": "wrong page", "action": "back"}')
    check("back는 추가 필드 없어도 통과", ok_back["action"] == "back" and not ok_back.get("_parse_failed"))
    check("_describe_action -> back 설명 문구", _describe_action({"action": "back"}) == "go back to the previous page")
    check("_is_similar_action -> back끼리는 항상 동일 취급", _is_similar_action({"action": "back"}, {"action": "back"}))
    check("_is_similar_action -> back과 wait는 다름", not _is_similar_action({"action": "back"}, {"action": "wait"}))
    back_hist = _format_history([{"action": "back"}])
    check("back 히스토리 포맷에 설명 문구 포함", "go back to the previous page" in back_hist)

    # (2026-08-07 추가) drag 액션 - AgentNet 학습 데이터에서 3.9%가 moveTo+dragTo 조합으로
    # 나왔는데 원래 스키마에 없어서 스킵되던 것을 추가함.
    ok_drag = _parse_planner_action(
        '{"reasoning": "r", "action": "drag", "target_description": "the fill handle of cell A2", "text": "cell A13"}'
    )
    check("drag: target_description+text 둘 다 있으면 통과", ok_drag["action"] == "drag" and not ok_drag.get("_parse_failed"))

    missing_drag_end = _parse_planner_action('{"action": "drag", "target_description": "the fill handle of cell A2"}')
    check("drag: text(끝 지점) 없으면 폴백", missing_drag_end["action"] == "terminate")

    missing_drag_start = _parse_planner_action('{"action": "drag", "text": "cell A13"}')
    check("drag: target_description(시작 지점) 없으면 폴백", missing_drag_start["action"] == "terminate")

    drag_hist = _format_history([{"action": "drag", "target_description": "cell A2 fill handle", "text": "cell A13"}])
    check("drag 히스토리 포맷", 'drag from "cell A2 fill handle" to "cell A13"' in drag_hist)

    # (2026-08-10 추가) reflection에게 반려당한 액션을 history에 넣을 때 - "이미 했다"가 아니라
    # "시도했지만 반려당해서 실행 안 됐다"는 게 명확히 드러나야 함(eval_webvoyager.py가 이 형태로
    # 넘김 - 반려 사실이 스텝을 넘어 유실되면 같은 걸 계속 재제안하는 문제가 실측으로 확인됨).
    rejected_hist = _format_history([
        {
            "action": "left_click",
            "target_description": "the CAPTCHA checkbox",
            "_rejected": True,
            "_rejection_reason": "no CAPTCHA checkbox is visible in the screenshot",
        }
    ])
    check("반려된 액션 -> '실행 안 됐다'는 문구 포함", "did NOT happen" in rejected_hist)
    check("반려된 액션 -> 원래 액션 설명도 그대로 포함", 'left_click on "the CAPTCHA checkbox"' in rejected_hist)
    check("반려된 액션 -> 반려 사유(critique)도 포함", "no CAPTCHA checkbox is visible" in rejected_hist)

    # 반려 사유가 없어도(critique가 빈 문자열인 케이스) 에러 없이 동작해야 함
    rejected_hist_no_reason = _format_history([{"action": "wait", "_rejected": True}])
    check("반려된 액션(사유 없음) -> 에러 없이 포맷됨", "did NOT happen" in rejected_hist_no_reason)

    # --- (2026-08-11 추가) _is_similar_action ---
    check(
        "_is_similar_action -> 같은 target의 click은 동일 취급(대소문자 무시)",
        _is_similar_action(
            {"action": "left_click", "target_description": "the Add to Cart button"},
            {"action": "left_click", "target_description": "THE ADD TO CART BUTTON"},
        ),
    )
    check(
        "_is_similar_action -> target이 다른 click은 다름",
        not _is_similar_action(
            {"action": "left_click", "target_description": "the Add to Cart button"},
            {"action": "left_click", "target_description": "the Buy Now button"},
        ),
    )
    check(
        "_is_similar_action -> action 종류 자체가 다르면 다름",
        not _is_similar_action({"action": "left_click", "target_description": "x"}, {"action": "scroll", "text": "down"}),
    )
    check(
        "_is_similar_action -> scroll은 방향까지 같아야 동일",
        _is_similar_action({"action": "scroll", "text": "down"}, {"action": "scroll", "text": "DOWN"})
        and not _is_similar_action({"action": "scroll", "text": "down"}, {"action": "scroll", "text": "up"}),
    )
    check(
        "_is_similar_action -> type은 text까지 같아야 동일",
        _is_similar_action({"action": "type", "text": "cats"}, {"action": "type", "text": "cats"})
        and not _is_similar_action({"action": "type", "text": "cats"}, {"action": "type", "text": "dogs"}),
    )
    check("_is_similar_action -> wait끼리는 항상 동일 취급", _is_similar_action({"action": "wait"}, {"action": "wait"}))

    # --- (2026-08-11 추가) _trailing_repeat_streak ---
    streak0, action0 = _trailing_repeat_streak([])
    check("_trailing_repeat_streak -> 빈 히스토리는 (0, None)", streak0 == 0 and action0 is None)

    same3 = [{"action": "scroll", "text": "down"} for _ in range(3)]
    streak3, _ = _trailing_repeat_streak(same3)
    check("_trailing_repeat_streak -> 연속 3회 반복 -> streak=3", streak3 == 3)

    interrupted = [
        {"action": "scroll", "text": "down"},
        {"action": "scroll", "text": "down"},
        {"action": "left_click", "target_description": "filters"},  # 중간에 다른 액션 -> 스트릭 끊김
        {"action": "scroll", "text": "down"},
    ]
    streak_interrupted, _ = _trailing_repeat_streak(interrupted)
    check("_trailing_repeat_streak -> 중간에 다른 액션이 끼면 최신 구간만 셈(streak=1)", streak_interrupted == 1)

    # _rejected 마커가 붙어 있어도 반복으로 셈(반려당했어도 "같은 시도"라는 신호는 유효)
    rejected_streak = [
        {"action": "left_click", "target_description": "the CAPTCHA checkbox", "_rejected": True, "_rejection_reason": "not visible"},
        {"action": "left_click", "target_description": "the CAPTCHA checkbox", "_rejected": True, "_rejection_reason": "not visible"},
        {"action": "left_click", "target_description": "the CAPTCHA checkbox"},
    ]
    streak_rej, _ = _trailing_repeat_streak(rejected_streak)
    check("_trailing_repeat_streak -> _rejected 항목도 반복 스트릭에 포함됨", streak_rej == 3)

    # --- (2026-08-11 추가) _windowed_repeat_count ---
    wc0, wa0 = _windowed_repeat_count([], 6)
    check("_windowed_repeat_count -> 빈 히스토리는 (0, None)", wc0 == 0 and wa0 is None)

    wc_consec, _ = _windowed_repeat_count(same3, 6)
    check("_windowed_repeat_count -> 연속 반복도 그대로 카운트됨(streak와 동일 결과)", wc_consec == 3)

    # 핵심: A -> B -> A -> B 뺑뺑이 - streak 방식은 절대 못 잡지만(매번 직전과 다름),
    # 윈도우 방식은 마지막 액션(A)이 윈도우 안에서 몇 번 나왔는지로 세므로 잡힌다.
    oscillating = [
        {"action": "left_click", "target_description": "search icon"},
        {"action": "left_click", "target_description": "back button"},
        {"action": "left_click", "target_description": "search icon"},
        {"action": "left_click", "target_description": "back button"},
        {"action": "left_click", "target_description": "search icon"},
    ]
    streak_osc, _ = _trailing_repeat_streak(oscillating)
    check("뺑뺑이 패턴 -> _trailing_repeat_streak는 못 잡음(streak=1)", streak_osc == 1)
    wc_osc, wa_osc = _windowed_repeat_count(oscillating, 6)
    check(
        "뺑뺑이 패턴 -> _windowed_repeat_count는 잡음(윈도우 안 'search icon' 3회)",
        wc_osc == 3 and wa_osc["target_description"] == "search icon",
    )

    interrupted_window = _windowed_repeat_count(interrupted, 6)
    check(
        "_windowed_repeat_count -> 중간에 다른 액션이 껴도 윈도우 안이면 카운트에 포함(streak와의 차이)",
        interrupted_window[0] == 3,
    )

    # --- (2026-08-11 추가) _format_history의 REPETITION WARNING 삽입 ---
    below_threshold = _format_history(
        [{"action": "scroll", "text": "down"}, {"action": "scroll", "text": "down"}]
    )
    check("반복 2회(임계값 미만) -> 경고 없음", "REPETITION WARNING" not in below_threshold)

    at_threshold = _format_history(
        [{"action": "scroll", "text": "down"} for _ in range(3)]
    )
    check("반복 3회(기본 임계값) -> 경고 발생", "REPETITION WARNING" in at_threshold)
    check("경고 문구에 반복 횟수 포함", "3 times" in at_threshold)
    check("경고 문구에 반복된 행동 설명 포함", "scroll down" in at_threshold)

    reset_by_different_action = _format_history(
        [
            {"action": "scroll", "text": "down"},
            {"action": "scroll", "text": "down"},
            {"action": "left_click", "target_description": "filters"},
        ]
    )
    check(
        "중간에 다른 액션이 껴도 윈도우 안에서 임계값 미달이면 경고 없음",
        "REPETITION WARNING" not in reset_by_different_action,
    )

    # 뺑뺑이(A-B-A-B-A-B) 케이스 -> 기본 window(=streak*2=6)가 다 찰 때 경고 발생해야 함
    oscillating_hist = _format_history(
        [
            {"action": "scroll", "text": "down"},
            {"action": "left_click", "target_description": "filters"},
            {"action": "scroll", "text": "down"},
            {"action": "left_click", "target_description": "filters"},
            {"action": "scroll", "text": "down"},
            {"action": "left_click", "target_description": "filters"},
        ]
    )
    check("뺑뺑이(A-B 번갈아 6회) -> 경고 발생", "REPETITION WARNING" in oscillating_hist)

    custom_threshold = _format_history(
        [{"action": "wait"}, {"action": "wait"}],
        repeat_warning_streak=2,
    )
    check("repeat_warning_streak 인자로 임계값을 낮추면 2회에도 경고 발생", "REPETITION WARNING" in custom_threshold)

    custom_window = _format_history(
        [
            {"action": "left_click", "target_description": "x"},
            {"action": "left_click", "target_description": "y"},
            {"action": "left_click", "target_description": "x"},
        ],
        repeat_warning_streak=2,
        repeat_warning_window=2,
    )
    check(
        "repeat_warning_window을 좁히면 윈도우 밖의 과거 반복은 안 잡힘",
        "REPETITION WARNING" not in custom_window,
    )

    # plan_next_action: 모델 generate()를 mock으로 대체해서 메시지 조립까지만 검증
    fake_model = MagicMock()
    fake_model.generate.return_value = '{"reasoning": "ok", "action": "wait"}'
    result = plan_next_action(fake_model, "do something", Image.new("RGB", (4, 4)), history_actions=hist)
    check("plan_next_action -> generate 호출됨", fake_model.generate.called)
    check("plan_next_action -> 파싱된 action 반환", result["action"] == "wait")
    call_messages = fake_model.generate.call_args[0][0]
    check("system 메시지 포함", call_messages[0]["role"] == "system")
    check("user 메시지에 이미지 포함", any(c.get("type") == "image" for c in call_messages[1]["content"]))
    check(
        "user 메시지에 task instruction 포함",
        any("do something" in c.get("text", "") for c in call_messages[1]["content"] if c.get("type") == "text"),
    )

    # revision_context -> 프롬프트에 이전 후보/critique가 실제로 들어가는지
    fake_model.reset_mock()
    fake_model.generate.return_value = '{"reasoning": "revised", "action": "wait"}'
    plan_next_action(
        fake_model,
        "do something",
        Image.new("RGB", (4, 4)),
        revision_context={
            "previous_plan": {"action": "terminate", "status": "success", "reasoning": "looked done"},
            "critique": "the close button is still visible, task is not done",
        },
    )
    call_messages2 = fake_model.generate.call_args[0][0]
    revised_user_text = next(
        c["text"] for c in call_messages2[1]["content"] if c.get("type") == "text"
    )
    check("revision_context -> REJECTED 문구 포함", "REJECTED" in revised_user_text)
    check("revision_context -> critique 내용 포함", "close button is still visible" in revised_user_text)

    # _parse_reflection_verdict
    good_verdict = _parse_reflection_verdict(
        '{"observation": "close button visible", "approved": false, "critique": "not done yet"}'
    )
    check("reflection verdict 정상 파싱", good_verdict["approved"] is False and good_verdict["critique"] == "not done yet")

    bad_verdict = _parse_reflection_verdict("not json")
    check("reflection verdict 파싱 실패 -> approved=False 폴백", bad_verdict["approved"] is False)
    check("reflection verdict 파싱 실패 -> _parse_failed 플래그", bad_verdict.get("_parse_failed") is True)

    # _format_candidate_plan
    fmt = _format_candidate_plan(
        {"action": "terminate", "status": "success", "answer": "42", "reasoning": "done"}
    )
    check("candidate plan 포맷 -> action 포함", "action: terminate" in fmt)
    check("candidate plan 포맷 -> status 포함", "status: success" in fmt)
    check("candidate plan 포맷 -> answer 포함", 'answer: "42"' in fmt)

    # plan_with_reflection: 1회차 반려 -> 2회차 승인 시나리오
    seq_model = MagicMock()
    seq_model.generate.side_effect = [
        '{"reasoning": "seems closed already", "action": "terminate", "status": "success"}',  # 1차 plan
        '{"observation": "a dialog is still open", "approved": false, "critique": "dialog still visible"}',  # 1차 reflect
        '{"reasoning": "need to click close first", "action": "left_click", "target_description": "close button"}',  # 2차 plan(수정)
        '{"observation": "close button visible", "approved": true, "critique": ""}',  # 2차 reflect
    ]
    result = plan_with_reflection(seq_model, "close the dialog", Image.new("RGB", (4, 4)), max_iterations=3)
    check("plan_with_reflection -> 최종 승인", result["_reflection_approved"] is True)
    check("plan_with_reflection -> 승인된 액션은 수정된 액션", result["action"] == "left_click")
    check("plan_with_reflection -> 로그 2개(반려1+승인1)", len(result["_reflection_log"]) == 2)
    check("plan_with_reflection -> generate 4번 호출(plan/reflect x2)", seq_model.generate.call_count == 4)

    # plan_with_reflection: 계속 반려 -> max_iterations 소진 후 마지막 후보 그대로 반환
    always_reject_model = MagicMock()
    always_reject_model.generate.side_effect = [
        '{"reasoning": "r1", "action": "wait"}',
        '{"observation": "o1", "approved": false, "critique": "c1"}',
        '{"reasoning": "r2", "action": "wait"}',
        '{"observation": "o2", "approved": false, "critique": "c2"}',
    ]
    result2 = plan_with_reflection(always_reject_model, "task", Image.new("RGB", (4, 4)), max_iterations=2)
    check("plan_with_reflection -> max_iterations 소진시 approved=False", result2["_reflection_approved"] is False)
    check("plan_with_reflection -> max_iterations 소진시 마지막 후보 반환", result2["action"] == "wait")
    check("plan_with_reflection -> 로그 길이 == max_iterations", len(result2["_reflection_log"]) == 2)

    # plan_with_reflection: planner 자체가 파싱 실패로 즉시 종료 폴백을 내면 reflection 없이 바로 반환
    parse_fail_model = MagicMock()
    parse_fail_model.generate.return_value = "not json at all"
    result3 = plan_with_reflection(parse_fail_model, "task", Image.new("RGB", (4, 4)), max_iterations=3)
    check("plan_with_reflection -> planner 파싱실패시 reflection 스킵", parse_fail_model.generate.call_count == 1)
    check("plan_with_reflection -> planner 파싱실패시 approved=False", result3["_reflection_approved"] is False)

    # plan_with_reflection: reflection_model을 명시하면 제안/비평이 서로 다른 모델 객체로
    # 라우팅되는지 - planner 전용 어댑터(propose_model)와 reflector 전용 어댑터(또는 base)
    # (reflect_model)를 분리해서 쓰는 agent_loop.py 사용 패턴을 재현.
    propose_model = MagicMock()
    propose_model.generate.side_effect = [
        '{"reasoning": "seems closed already", "action": "terminate", "status": "success"}',
        '{"reasoning": "need to click close first", "action": "left_click", "target_description": "close button"}',
    ]
    reflect_model = MagicMock()
    reflect_model.generate.side_effect = [
        '{"observation": "a dialog is still open", "approved": false, "critique": "dialog still visible"}',
        '{"observation": "close button visible", "approved": true, "critique": ""}',
    ]
    result4 = plan_with_reflection(
        propose_model, "close the dialog", Image.new("RGB", (4, 4)),
        max_iterations=3, reflection_model=reflect_model,
    )
    check("reflection_model 지정 -> 최종 승인", result4["_reflection_approved"] is True)
    check("reflection_model 지정 -> propose_model이 제안 2번 호출됨", propose_model.generate.call_count == 2)
    check("reflection_model 지정 -> reflect_model이 비평 2번 호출됨", reflect_model.generate.call_count == 2)
    check(
        "reflection_model 지정 -> 비평 호출이 propose_model로는 안 감(라우팅 분리 확인)",
        not any("observation" in str(c) for c in propose_model.generate.call_args_list),
    )

    # reflection_model을 안 주면(기본값 None) 예전처럼 qwen_model 하나로 제안/비평이 다 감
    # (하위 호환 확인 - 위의 seq_model 테스트가 이미 이 경로를 검증하지만, 명시적으로 한 번 더)
    check(
        "reflection_model 기본값 None -> _reflect_on_plan()에도 qwen_model이 그대로 감(seq_model 재사용 테스트로 이미 확인됨)",
        seq_model.generate.call_count == 4,
    )

    n_fail = sum(1 for _, ok in checks if not ok)
    for name, ok in checks:
        print(("[OK]  " if ok else "[FAIL]") + " " + name)
    print(f"\n{len(checks) - n_fail}/{len(checks)} passed")
    if n_fail:
        raise SystemExit(1)


def _cli():
    import argparse

    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="스크린샷 이미지 경로 (수동 테스트용)")
    ap.add_argument("--task", help="태스크 지시문")
    ap.add_argument("--selftest", action="store_true", help="실제 모델 없이 프롬프트 조립/파싱 로직만 mock으로 검증")
    ap.add_argument(
        "--reflect", action="store_true", help="plan_next_action() 대신 plan_with_reflection() 사용 (실행 전 비평 루프)"
    )
    ap.add_argument("--max_iterations", type=int, default=2, help="--reflect일 때 최대 재시도 횟수 (agent_loop.py 기본값과 동일하게 2)")
    ap.add_argument(
        "--adapter_dir",
        default=None,
        help="(2026-08-07 추가) 지정하면 이 경로의 grounding LoRA(peft)를 얹은 채로 planning을 "
        "테스트한다. 미지정(기본)이면 파일 상단 docstring에 적힌 설계대로 base 모델(LoRA 없음)로 "
        "돈다. base vs adapter를 그냥 비교해보고 싶을 때(=grounding LoRA가 planning 포맷을 "
        "실제로 방해하는지 실측하고 싶을 때) 이 옵션을 켜고 끄면서 같은 --image/--task로 "
        "돌려보면 됨 (예: checkpoint-4130).",
    )
    ap.add_argument(
        "--min_pixels", type=int, default=None,
        help="미지정시 qwen.py의 DEFAULT_MIN_PIXELS(256*28*28=200704) 사용",
    )
    ap.add_argument(
        "--max_pixels", type=int, default=None,
        help="(2026-08-07 추가) 미지정시 700,000을 기본값으로 쓴다 - qwen.py의 "
        "DEFAULT_MAX_PIXELS는 501,760(50만)인데, checkpoint-4130 LoRA는 700,000으로 "
        "파인튜닝됐다. RegionFocus 쪽에서 이 둘을 헷갈려서 생긴 해상도 confound로 "
        "크게 데인 적 있어서(project_step14_ablation_dosample_noise 메모리 참고) 여기서는 "
        "qwen.py 기본값을 그대로 따르지 않는다. 50만과 비교하고 싶으면 --max_pixels 501760을 "
        "명시적으로 넘기면 됨.",
    )
    args = ap.parse_args()

    if args.selftest:
        _run_mock_selftest()
        return

    if not args.image or not args.task:
        raise SystemExit("--image와 --task 필요 (또는 --selftest)")

    # 실제 실행 시점에만 필요 (selftest는 이 임포트를 안 탐) - 절대 import로 통일
    # (agent_loop.py와 동일 스타일, vlm_agent/ 안에서 스크립트로 직접 실행하는 걸 전제).
    from qwen import DEFAULT_MIN_PIXELS, QwenVLModel

    min_pixels = args.min_pixels if args.min_pixels is not None else DEFAULT_MIN_PIXELS
    max_pixels = args.max_pixels if args.max_pixels is not None else 700_000

    model = QwenVLModel(adapter_dir=args.adapter_dir, min_pixels=min_pixels, max_pixels=max_pixels)
    screenshot = Image.open(args.image)
    if args.reflect:
        result = plan_with_reflection(model, args.task, screenshot, max_iterations=args.max_iterations)
    else:
        result = plan_next_action(model, args.task, screenshot)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()