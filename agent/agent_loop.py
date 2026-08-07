"""
agent_loop.py

전체 에이전트 루프. 두 단계로 구성된다:

  1) 모델 인스턴스 하나 + LoRA 어댑터 on/off 스위칭 (_BaseModelView / load_shared_model) -
     planner용 base 모델과 grounding용 LoRA 모델을 각각 별도 QwenVLModel 인스턴스로
     띄웠다가 실제로 RAM(16GB, CUDA 없는 CPU 환경)이 터진 적이 있어서(터미널/프로세스
     두 개가 각각 모델을 물고 있다가 두 번째 로딩에서 터짐), 이 문제를 프롬프트 병합이
     아니라 "모델은 grounding LoRA를 얹은 채로 딱 한 번만 로드하고, planning/reflection이
     필요할 때만 peft의 disable_adapter() 컨텍스트로 일시적으로 base 모델처럼 동작시킨다"는
     방식으로 구조적으로 없앤다.

  2) planning(어댑터 꺼짐) <-> grounding(어댑터 켜짐) <-> env.execute_action() 실제
     step loop (run_episode) - task 하나를 처음부터 끝까지 돌린다.

[왜 프롬프트 병합이 아니라 어댑터 스위칭인가 - 요약]
grounding LoRA는 coord_utils.PROMPT_TEMPLATE 하나, "(x,y)" 한 줄 출력이라는 아주 좁은
포맷으로만 SFT됐다(train.py). planning(JSON, reasoning, target_description 등)은 이
LoRA가 학습에서 한 번도 못 본 포맷이라, 어댑터를 켠 채로 planning을 시키면 - 실제로
planner.py의 _action_schema_valid() 회귀 테스트로 확인된 사례(어댑터 켠 채로 planning시
좌표 tool-call 포맷이 그대로 새어나옴)와 같은 방식으로 - planning이 오히려 더 망가질
위험이 있다. 그래서 planning/reflection은 어댑터를 꺼서 순수 base 모델 동작으로 돌리고,
grounding만 어댑터를 켠 상태로 돌린다 - 전부 같은 프로세스, 같은 모델 인스턴스 안에서.

[vision encoder는 항상 동일하다 - "grounding 모델이 이미지를 더 잘 본다"는 주장에 대한 반박 근거]
train.py의 LoraConfig.target_modules는 ["q_proj","k_proj","v_proj","o_proj"] - LLM 디코더
쪽 attention만 건드리고 vision encoder는 전혀 안 건드린다. 즉 어댑터를 껐다 켰다 해도
"이미지를 보는" 부분의 가중치는 항상 동일하다 - 바뀌는 건 그 시각 정보를 어떤 출력
포맷으로 뽑아내느냐뿐이다. planning을 어댑터 없이 시켜도 화면을 보는 능력 자체는
grounding 때와 다르지 않다.

[reflection 모드 on/off - 언제, 어떻게 결정되는가]
plan_next_action() <-> plan_with_reflection() 중 뭘 쓸지는 에피소드 "시작 시점에 한 번"
CLI(`--reflect`)로 고르고, 그 값이 에피소드 끝까지 고정된다 - 스텝마다 동적으로(예: 이전
액션 실패 여부에 따라) 바뀌지 않는다. run_episode()의 use_reflection 파라미터가 이 값을
그대로 받는다. planner.py 자체가 이미 이 두 함수를 독립적으로 노출하고 있어서
(plan_next_action / plan_with_reflection), agent_loop.py는 "어느 함수를 부를지"만
run_episode 진입 시 한 번 결정하면 된다.

[execute 루프 - run_episode()]
task 하나에 대해 env.reset() -> (planning -> [클릭류만 grounding으로 좌표 변환] ->
env.execute_action()) 반복 -> 아래 두 조건 중 하나에서 종료:
  1. 모델이 done/finish 신호를 냄 (planner가 action="terminate" 반환)
  2. 액션 실행 자체가 실패함 (grounding이 좌표를 못 뽑았거나, env.execute_action()이
     예외를 던짐)
두 조건 모두 즉시 루프를 멈춘다. max_steps는 이 두 조건과 별개로 존재하는 안전장치일
뿐이다(무한 루프 방지) - 필요 없으면 max_steps=None으로 끌 수 있다.

[해상도 - grounding LoRA가 학습된 해상도(700,000)로 기본 고정됨]
(2026-08 수정) load_shared_model()과 _plan_to_env_action()은 max_pixels를 명시하지
않으면 둘 다 _GROUNDING_TRAINED_MAX_PIXELS(700,000)를 기본값으로 쓴다 - checkpoint-4130이
실제로 그 해상도로 파인튜닝됐는데, 예전에 qwen.py 기본값(501,760)과 안 맞아서 grounding
정확도가 크게 깎인 전례가 있다(planner.py의 _cli() 주석 참고, "해상도 confound"). 이미지는
generate() 호출 전에 두 군데에서 각각 해상도 상한이 걸린다 - (1) gui_grounding.ground()가
호출 전에 직접 하는 PIL resize(ground()의 max_pixels 인자), (2) QwenVLModel/processor가
구성 시점에 들고 있는 (min_pixels, max_pixels)로 generate() 안에서 다시 도는 smart_resize.
둘 중 하나만 700,000으로 고치면 더 작은 쪽(안 고친 쪽의 기존 기본값)이 이겨서 무효화되므로,
_GROUNDING_TRAINED_MAX_PIXELS 상수 하나를 두 군데가 공유해서 항상 같이 바뀌게 했다. 다른
체크포인트를 쓴다면 load_shared_model(..., max_pixels=...)와
run_episode(..., ground_kwargs={"max_pixels": ...})를 함께 명시적으로 덮어쓸 것(하나만
바꾸면 안 됨).

[사용법]
    model, planning_view = load_shared_model(adapter_dir="checkpoint-4130")

    # 단발 테스트 (planning 한 번, grounding 한 번) - env/실제 브라우저 없이
    plan = plan_with_reflection(planning_view, task, screenshot, history)   # 어댑터 꺼짐
    result = ground(model, plan["target_description"], screenshot)          # 어댑터 켜짐(기본)

    # 실제 태스크 하나를 처음부터 끝까지: env_webvoyager.WebVoyagerEnv와 엮어서
    from env_webvoyager import WebVoyagerEnv
    env = WebVoyagerEnv(headless=True)
    outcome = run_episode(
        model, planning_view, env,
        task={"web": "https://en.wikipedia.org", "ques": "Find who wrote the Python article."},
        use_reflection=True,   # CLI --reflect와 동일한 스위치, 에피소드 내내 고정
    )
    env.close()

model 하나만 메모리에 올라간다 - planning_view는 같은 model을 감싸서 .generate() 호출을
disable_adapter() 컨텍스트로 넘겨주는 얇은 프록시일 뿐, 별도 모델이 아니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 실행에는 필요 없고 타입 힌트용 - torch/transformers/peft를 mock selftest 경로에서까지
    # 강제로 임포트하지 않기 위해 TYPE_CHECKING 가드로 묶어둠 (planner.py와 동일한 패턴).
    from qwen import QwenVLModel


# gui_grounding.py의 ground() 문서에 적혀 있듯, qwen_model.generate()에 이미지를 넘기면
# processor가 내부적으로 자기 자신의 (min_pixels, max_pixels)로 smart_resize를 "다시" 돌린다
# (이미 그 범위 안에 있는 크기면 멱등이라 그대로 통과, 범위를 벗어나면 그 순간 다시 리사이즈됨).
# 즉 해상도는 두 군데에서 동시에 맞아야 실제로 적용된다:
#   1) gui_grounding.ground()가 호출 전에 직접 하는 PIL resize (ground()의 max_pixels 인자)
#   2) QwenVLModel/processor 자체가 구성 시점에 들고 있는 (min_pixels, max_pixels)
# 둘 중 하나만 700,000으로 고치면, 둘 중 더 작은 쪽(=안 고친 쪽의 기존 기본값,
# qwen.py의 DEFAULT_MAX_PIXELS=501,760)이 최종적으로 이겨서 아무 효과가 없다 - 그래서
# load_shared_model()과 _plan_to_env_action() 둘 다 이 상수 하나를 공유해서 기본값으로 쓴다.
# checkpoint-4130 LoRA가 실제로 학습된 해상도가 700,000이라는 근거는 planner.py의 _cli()
# 주석 참고("qwen.py의 DEFAULT_MAX_PIXELS는 501,760인데 checkpoint-4130 LoRA는 700,000으로
# 파인튜닝됐다... 해상도 confound로 크게 데인 적 있음"). 다른 체크포인트를 쓴다면
# load_shared_model(..., max_pixels=...)/run_episode(..., ground_kwargs={"max_pixels": ...})로
# 명시적으로 덮어쓸 것.
_GROUNDING_TRAINED_MAX_PIXELS = 700_000


class _BaseModelView:
    """
    grounding LoRA를 얹은 QwenVLModel을 감싸서, .generate() 호출을 peft의
    disable_adapter() 컨텍스트 안에서 실행시키는 얇은 프록시.

    planner.py의 plan_next_action()/plan_with_reflection() 등은 qwen_model 인자에
    duck-typing으로 .generate(messages, max_new_tokens=..., temperature=..., top_p=...)만
    있으면 되므로, 이 프록시를 그대로 넘기면 planner.py를 전혀 수정하지 않고도 "어댑터
    꺼진 상태"로 동작시킬 수 있다.

    qwen_model.model이 peft.PeftModel이 아니면(=adapter_dir 없이 로드된 경우)
    disable_adapter()가 없다 - 이 프록시는 애초에 "어댑터를 얹은 모델을 잠깐 base처럼
    쓰고 싶을 때"용이라, 처음부터 어댑터가 없는 모델에는 쓸 이유가 없다(그냥 그 모델을
    직접 넘기면 됨). 이 경우를 generate() 호출 시점(reflection 루프 도중일 수도 있음)이
    아니라 생성 시점에 바로 걸러내서, 에러가 나더라도 어디서 뭐가 잘못됐는지 바로 알 수
    있게 한다.
    """

    def __init__(self, qwen_model: "QwenVLModel"):
        if not hasattr(qwen_model.model, "disable_adapter"):
            raise TypeError(
                "_BaseModelView는 LoRA 어댑터가 얹힌(peft.PeftModel) QwenVLModel에만 쓸 수 있음 - "
                f"qwen_model.model의 타입이 {type(qwen_model.model).__name__}이라 disable_adapter()가 "
                "없음. adapter_dir 없이 로드한 모델이면 _BaseModelView로 감쌀 필요 없이 그 모델을 "
                "그대로 planner.py 함수들에 넘기면 됨."
            )
        self._qwen_model = qwen_model

    def generate(
        self, messages: list, max_new_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0
    ) -> str:
        with self._qwen_model.model.disable_adapter():
            return self._qwen_model.generate(
                messages, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            )


def load_shared_model(adapter_dir: str, **qwen_model_kwargs):
    """
    QwenVLModel을 grounding LoRA를 얹은 상태로 "딱 한 번만" 로드하고,
    (grounding용 뷰, planning용 뷰) 튜플을 반환한다.

    grounding 호출은 반환된 model을 그대로 쓰면 됨(어댑터가 기본으로 켜진 상태 =
    gui_grounding.ground()/region_focus.region_focus() 등이 기대하는 상태 그대로).
    planning/reflection 호출은 반환된 planning_view를 planner.py의 함수들에 넘기면 됨.

    qwen_model_kwargs는 QwenVLModel(...)에 그대로 전달됨(min_pixels 등 조정용). max_pixels를
    명시하지 않으면 _GROUNDING_TRAINED_MAX_PIXELS(700,000)가 기본으로 들어간다(위 상수 주석
    참고). _plan_to_env_action()의 ground_kwargs 기본값과 짝을 이루니, 둘 중 하나만 바꾸지
    말고 같이 바꿀 것(다른 체크포인트로 바꿀 때 등).

    Returns: (model: QwenVLModel, planning_view: _BaseModelView)
    """
    from qwen import QwenVLModel  # 실제 로드 시점에만 임포트 (mock 테스트는 이 경로를 안 탐)

    qwen_model_kwargs.setdefault("max_pixels", _GROUNDING_TRAINED_MAX_PIXELS)
    model = QwenVLModel(adapter_dir=adapter_dir, **qwen_model_kwargs)
    planning_view = _BaseModelView(model)
    return model, planning_view


# ---------------------------------------------------------------------------
# step loop: planning(토글 가능) -> [grounding] -> env.execute_action()
# ---------------------------------------------------------------------------

_CLICK_ACTIONS = ("left_click", "double_click", "right_click")


def _plan_to_env_action(plan: dict, screenshot, model, ground_kwargs: dict | None = None) -> dict:
    """
    planner.py의 출력(plan dict, 파일 상단 docstring의 "출력 스키마" 참고) 하나를
    env_webvoyager.WebVoyagerEnv.execute_action()이 받는 action dict로 변환한다.

    - click류(left_click/double_click/right_click): target_description(자연어)을
      gui_grounding.ground()로 실제 좌표로 바꿔야 한다. ground()가 돌려주는 point는
      "원본 스크린샷 기준 0~1 정규화 좌표"라서(gui_grounding.py 문서 참고), env가 기대하는
      절대 픽셀 좌표로 되돌리려면 각각 screenshot.width/height를 곱하면 된다 - env가
      CDP Input.dispatchMouseEvent에 쓰는 좌표계가 곧 이 screenshot을 만든
      driver.get_screenshot_as_png()와 동일한 좌표계이기 때문에 별도 스케일 보정이
      필요 없다(env_webvoyager.py 문서의 CDP 채택 이유 참고). ground_kwargs에 max_pixels를
      안 넣으면 _GROUNDING_TRAINED_MAX_PIXELS(700,000)가 기본으로 들어간다 - load_shared_model()
      쪽 상수 주석 참고(model 쪽도 같이 이 값으로 맞춰져 있어야 실제로 적용됨).
    - type/key/scroll/wait: 좌표가 필요 없어서 grounding을 거치지 않고 필드 이름만
      맞춰서 바로 env action으로 변환한다.
    - terminate는 여기로 들어오면 안 된다 - run_episode()가 terminate를 env로 보내기
      전에 이미 걸러낸다(env_webvoyager.py도 terminate를 execute_action()에 보내면
      ValueError를 던지도록 방어해뒀다).

    grounding이 좌표를 못 뽑으면("wrong_format") ValueError를 던진다 - env로 보낼
    좌표 자체가 없어서 애초에 액션을 만들 수 없는 경우라, run_episode()의 "액션 실행
    실패" 종료 조건 쪽으로 그대로 흡수되게 한다(따로 특별 취급하지 않음).
    """
    from gui_grounding import ground

    action = plan.get("action")
    if action in _CLICK_ACTIONS:
        target = plan.get("target_description", "") or ""
        gk = dict(ground_kwargs or {})
        gk.setdefault("max_pixels", _GROUNDING_TRAINED_MAX_PIXELS)
        result = ground(model, target, screenshot, **gk)
        if result.get("result") != "positive":
            raise ValueError(
                f"grounding 실패(wrong_format) - target_description={target!r}, "
                f"raw_response={result.get('raw_response')!r}"
            )
        x_norm, y_norm = result["point"]
        x = x_norm * screenshot.width
        y = y_norm * screenshot.height
        return {"action": action, "coordinate": [x, y]}
    if action in ("type", "key"):
        return {"action": action, "text": plan.get("text", "")}
    if action == "scroll":
        return {"action": "scroll", "text": plan.get("text", "down")}
    if action == "wait":
        return {"action": "wait"}
    raise ValueError(
        f"env로 보낼 수 없는 action: {action!r} (terminate는 run_episode가 별도 처리해야 함 - "
        "여기까지 오면 안 됨)"
    )


def run_episode(
    model,
    planning_view,
    env,
    task,
    use_reflection: bool,
    max_steps: int | None = 50,
    reflection_max_iterations: int = 2,
    planner_kwargs: dict | None = None,
    ground_kwargs: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    태스크 하나를 처음부터 끝까지 돌린다: env.reset() -> (planning -> [클릭류만
    grounding] -> env.execute_action()) 반복 -> 종료.

    Args:
        model: 어댑터가 켜진(grounding용) QwenVLModel. _plan_to_env_action()이
            gui_grounding.ground()를 부를 때 이걸 씀.
        planning_view: load_shared_model()이 만든 _BaseModelView(어댑터 꺼짐). planner.py의
            plan_next_action()/plan_with_reflection() 호출에 씀.
        env: env_webvoyager.WebVoyagerEnv 인스턴스(reset/execute_action 인터페이스만
            맞으면 mock으로 대체 가능 - 아래 _run_mock_selftest 참고).
        task: env.reset(task)에 그대로 전달됨 ({"web":..., "ques":...} dict 또는
            (url, instruction) 튜플).
        use_reflection: True면 매 스텝 plan_with_reflection()(비평 루프)을, False면
            plan_next_action()을 쓴다. 에피소드 시작 시 한 번 정해지고 끝까지 고정됨
            (파일 상단 docstring의 "reflection 모드 on/off" 참고 - 스텝마다 바뀌지 않음).
            CLI에서는 --reflect 플래그로 이 값을 넘긴다.
        max_steps: 두 종료 조건과 별개인 안전장치(무한 루프 방지). None이면 끔.
        reflection_max_iterations: use_reflection=True일 때 plan_with_reflection()의
            max_iterations로 그대로 전달.
        planner_kwargs: plan_next_action()/plan_with_reflection() 둘 다에 추가로 넘길
            kwargs(예: max_new_tokens). max_iterations는 여기 넣지 말고
            reflection_max_iterations를 쓸 것(두 함수의 파라미터 이름이 서로 달라서
            충돌 방지 차원에서 따로 뺐음).
        ground_kwargs: _plan_to_env_action()이 gui_grounding.ground()를 부를 때 추가로
            넘길 kwargs(예: min_pixels). max_pixels를 명시하지 않으면
            _GROUNDING_TRAINED_MAX_PIXELS(700,000, checkpoint-4130이 실제로 학습된 해상도)가
            기본으로 들어간다 - 다른 체크포인트를 쓴다면 여기서 명시적으로 덮어쓸 것
            (load_shared_model()에 넘기는 max_pixels도 같이 맞출 것).

    Returns:
        {
            "status": "success" | "failure",
            "reason": "model_terminate" | "execution_failed" | "max_steps_reached",
            "answer": str | None,        # status=="success"인 terminate에서만 채워짐
            "steps": int,                 # 실제로 시도한 스텝 수
            "history": [plan dict, ...],  # 매 스텝의 plan (성공한 스텝은 "_env_action",
                                           # 실패한 마지막 스텝은 "_env_action_error" 필드 추가)
            "task_info": dict,            # 마지막으로 받은 env task_info
        }
    """
    from planner import plan_next_action, plan_with_reflection

    screenshot, task_info = env.reset(task)
    task_instruction = (task_info or {}).get("instruction")

    history: list = []
    planner_kw = dict(planner_kwargs or {})
    step = 0

    while max_steps is None or step < max_steps:
        step += 1

        if use_reflection:
            plan = plan_with_reflection(
                planning_view, task_instruction, screenshot,
                history_actions=history,
                max_iterations=reflection_max_iterations,
                **planner_kw,
            )
        else:
            plan = plan_next_action(
                planning_view, task_instruction, screenshot,
                history_actions=history,
                **planner_kw,
            )

        if verbose:
            print(
                f"[agent_loop] step {step}: action={plan.get('action')!r} "
                f"target={plan.get('target_description')!r} text={plan.get('text')!r}"
            )

        # --- 종료 조건 1: 모델이 done/finish 신호를 냄 ---
        if plan.get("action") == "terminate":
            history.append(plan)
            return {
                "status": plan.get("status", "failure"),
                "reason": "model_terminate",
                "answer": plan.get("answer"),
                "steps": step,
                "history": history,
                "task_info": task_info,
            }

        # --- 종료 조건 2: 액션 실행 자체가 실패(grounding 실패 포함) ---
        try:
            env_action = _plan_to_env_action(plan, screenshot, model, ground_kwargs=ground_kwargs)
            screenshot, _reward, _terminated, _truncated, task_info = env.execute_action(env_action)
        except Exception as e:
            plan["_env_action_error"] = str(e)
            history.append(plan)
            if verbose:
                print(f"[agent_loop] step {step}: 실행 실패 - {e}")
            return {
                "status": "failure",
                "reason": "execution_failed",
                "answer": None,
                "steps": step,
                "history": history,
                "task_info": task_info,
            }

        plan["_env_action"] = env_action
        history.append(plan)

    return {
        "status": "failure",
        "reason": "max_steps_reached",
        "answer": None,
        "steps": step,
        "history": history,
        "task_info": task_info,
    }


# ---------------------------------------------------------------------------
# mock 기반 단위 테스트 (실제 모델/torch/peft/selenium 없이 로직만 검증)
# ---------------------------------------------------------------------------
def _run_mock_selftest():
    """`python agent_loop.py --selftest`"""
    import sys
    import types
    from unittest.mock import MagicMock

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    # --- _BaseModelView: generate()가 disable_adapter() 컨텍스트 안에서 실제로 도는지 ---
    fake_qwen_model = MagicMock()
    fake_qwen_model.generate.return_value = "raw response text"

    view = _BaseModelView(fake_qwen_model)
    result = view.generate([{"role": "user", "content": []}], max_new_tokens=100, temperature=0.3)

    check("generate 결과가 그대로 반환됨", result == "raw response text")
    check("disable_adapter()가 호출됨", fake_qwen_model.model.disable_adapter.called)
    check("disable_adapter 컨텍스트 안에서 실제 generate가 호출됨", fake_qwen_model.generate.called)

    _, kwargs = fake_qwen_model.generate.call_args
    check("max_new_tokens이 그대로 전달됨", kwargs.get("max_new_tokens") == 100)
    check("temperature가 그대로 전달됨", kwargs.get("temperature") == 0.3)

    cm = fake_qwen_model.model.disable_adapter.return_value
    check("disable_adapter 컨텍스트 __enter__ 호출됨", cm.__enter__.called)
    check("disable_adapter 컨텍스트 __exit__ 호출됨", cm.__exit__.called)

    # --- _BaseModelView: 어댑터 없는(=disable_adapter가 없는) 모델을 넘기면 생성 시점에 바로 에러 ---
    class _PlainModelNoAdapter:
        pass

    class _FakeNoAdapterQwenModel:
        def __init__(self):
            self.model = _PlainModelNoAdapter()

    try:
        _BaseModelView(_FakeNoAdapterQwenModel())
        check("어댑터 없는 모델 -> 생성 시점에 TypeError", False)
    except TypeError as e:
        check("어댑터 없는 모델 -> 생성 시점에 TypeError", "disable_adapter" in str(e))

    # --- load_shared_model: QwenVLModel이 "딱 한 번만" 생성되는지 ---
    construct_calls = []

    class _FakeQwenVLModel:
        def __init__(self, **kwargs):
            construct_calls.append(kwargs)
            self.model = MagicMock()

        def generate(self, *a, **kw):
            return "ok"

    fake_qwen_module = types.ModuleType("qwen")
    fake_qwen_module.QwenVLModel = _FakeQwenVLModel
    sys.modules["qwen"] = fake_qwen_module
    try:
        model, planning_view = load_shared_model(adapter_dir="fake-checkpoint", min_pixels=123)
        check("QwenVLModel 생성 호출이 딱 1번", len(construct_calls) == 1)
        check("adapter_dir가 그대로 전달됨", construct_calls[0].get("adapter_dir") == "fake-checkpoint")
        check("나머지 kwargs(min_pixels 등)도 그대로 전달됨", construct_calls[0].get("min_pixels") == 123)
        check("model과 planning_view가 같은 인스턴스를 공유함(별도 모델 아님)", planning_view._qwen_model is model)
        check(
            "max_pixels 미지정시 학습 해상도(700,000)가 기본으로 들어감",
            construct_calls[0].get("max_pixels") == _GROUNDING_TRAINED_MAX_PIXELS == 700_000,
        )

        construct_calls.clear()
        load_shared_model(adapter_dir="fake-checkpoint-2", max_pixels=999_999)
        check(
            "max_pixels 명시하면 기본값(700,000)으로 덮어써지지 않고 그대로 전달됨",
            construct_calls[0].get("max_pixels") == 999_999,
        )
    finally:
        del sys.modules["qwen"]

    # =======================================================================
    # run_episode() / _plan_to_env_action(): planner.py와 gui_grounding.py를
    # 가짜 모듈로 대체해서(무거운 torch/transformers/selenium 의존성 없이) 순수
    # 오케스트레이션 로직만 검증한다. run_episode/​_plan_to_env_action은 둘 다
    # "planner"/"gui_grounding"을 함수 안에서 지연 임포트하므로, sys.modules에
    # 가짜 모듈만 미리 넣어두면 실제 import 시점에 그걸 그대로 집어간다
    # (load_shared_model 테스트에서 qwen 모듈을 가짜로 넣는 것과 같은 패턴).
    # =======================================================================
    class _FakeScreenshot:
        def __init__(self, width=1000, height=800):
            self.width = width
            self.height = height

    class _FakeEnv:
        def __init__(self, task_info=None, execute_side_effect=None):
            self.reset_calls = []
            self.execute_calls = []
            self._task_info = task_info or {"instruction": "do something", "url": "http://x"}
            self._execute_side_effect = execute_side_effect

        def reset(self, task):
            self.reset_calls.append(task)
            return _FakeScreenshot(), dict(self._task_info)

        def execute_action(self, action):
            self.execute_calls.append(action)
            if self._execute_side_effect is not None:
                effect = self._execute_side_effect
                if callable(effect):
                    effect(action)
                elif isinstance(effect, Exception):
                    raise effect
            return _FakeScreenshot(), None, False, False, dict(self._task_info)

    def _install_fake_planner_and_grounding(plan_next_action=None, plan_with_reflection=None, ground=None):
        fake_planner = types.ModuleType("planner")
        fake_planner.plan_next_action = plan_next_action or MagicMock(
            return_value={"reasoning": "r", "action": "wait"}
        )
        fake_planner.plan_with_reflection = plan_with_reflection or MagicMock(
            return_value={"reasoning": "r", "action": "wait", "_reflection_approved": True}
        )
        fake_grounding = types.ModuleType("gui_grounding")
        fake_grounding.ground = ground or MagicMock(
            return_value={"result": "positive", "point": [0.5, 0.25], "raw_response": "(500,250)"}
        )
        sys.modules["planner"] = fake_planner
        sys.modules["gui_grounding"] = fake_grounding
        return fake_planner, fake_grounding

    def _uninstall_fake_modules():
        for name in ("planner", "gui_grounding"):
            sys.modules.pop(name, None)

    # --- 종료 조건 1: 모델이 첫 스텝에서 바로 terminate ---
    plan_fn = MagicMock(return_value={"reasoning": "done", "action": "terminate", "status": "success", "answer": "42"})
    _install_fake_planner_and_grounding(plan_next_action=plan_fn)
    try:
        env = _FakeEnv()
        outcome = run_episode(
            model="fake-model", planning_view="fake-planning-view", env=env,
            task={"web": "http://x", "ques": "do something"}, use_reflection=False, verbose=False,
        )
        check("terminate 즉시 -> status success", outcome["status"] == "success")
        check("terminate 즉시 -> reason model_terminate", outcome["reason"] == "model_terminate")
        check("terminate 즉시 -> answer 보존", outcome["answer"] == "42")
        check("terminate 즉시 -> steps == 1", outcome["steps"] == 1)
        check("terminate 즉시 -> execute_action 호출 안 됨", len(env.execute_calls) == 0)
        check("use_reflection=False -> plan_next_action 사용", plan_fn.called)
    finally:
        _uninstall_fake_modules()

    # --- use_reflection=True -> plan_with_reflection이 쓰이고 plan_next_action은 안 쓰임 ---
    reflect_fn = MagicMock(
        return_value={"reasoning": "done", "action": "terminate", "status": "success", "answer": None}
    )
    next_fn = MagicMock(return_value={"reasoning": "should not be called", "action": "terminate", "status": "failure"})
    _install_fake_planner_and_grounding(plan_next_action=next_fn, plan_with_reflection=reflect_fn)
    try:
        env = _FakeEnv()
        outcome = run_episode(
            model="fake-model", planning_view="fake-planning-view", env=env,
            task=("http://x", "do something"), use_reflection=True,
            reflection_max_iterations=5, verbose=False,
        )
        check("use_reflection=True -> plan_with_reflection 호출됨", reflect_fn.called)
        check("use_reflection=True -> plan_next_action은 호출 안 됨", not next_fn.called)
        check(
            "reflection_max_iterations가 max_iterations로 전달됨",
            reflect_fn.call_args.kwargs.get("max_iterations") == 5,
        )
    finally:
        _uninstall_fake_modules()

    # --- click 액션 -> grounding으로 좌표 변환 -> execute_action에 절대 픽셀 좌표로 전달 -> 다음 스텝에서 terminate ---
    click_then_terminate = MagicMock(
        side_effect=[
            {"reasoning": "click it", "action": "left_click", "target_description": "the button"},
            {"reasoning": "done", "action": "terminate", "status": "success", "answer": "ok"},
        ]
    )
    ground_fn = MagicMock(return_value={"result": "positive", "point": [0.5, 0.25], "raw_response": "(500,250)"})
    _install_fake_planner_and_grounding(plan_next_action=click_then_terminate, ground=ground_fn)
    try:
        env = _FakeEnv()
        outcome = run_episode(
            model="fake-model", planning_view="fake-planning-view", env=env,
            task={"web": "http://x", "ques": "click the button"}, use_reflection=False, verbose=False,
        )
        check("click -> grounding 호출됨", ground_fn.called)
        check("click -> execute_action 1번 호출", len(env.execute_calls) == 1)
        sent = env.execute_calls[0]
        check("click -> action 이름 보존", sent["action"] == "left_click")
        check(
            "click -> 정규화 좌표(0.5,0.25)*1000x800 = (500,200) 절대픽셀로 변환",
            sent["coordinate"] == [500.0, 200.0],
        )
        check("click -> 최종 status success", outcome["status"] == "success")
        check("click -> steps == 2", outcome["steps"] == 2)
        check(
            "click -> history[0]에 _env_action 기록됨",
            outcome["history"][0].get("_env_action", {}).get("coordinate") == [500.0, 200.0],
        )
        check(
            "click -> ground_kwargs 미지정시 max_pixels가 학습 해상도(700,000)로 기본 전달됨",
            ground_fn.call_args.kwargs.get("max_pixels") == _GROUNDING_TRAINED_MAX_PIXELS == 700_000,
        )
    finally:
        _uninstall_fake_modules()

    # --- ground_kwargs로 max_pixels를 명시하면 기본값(700,000)으로 덮어써지지 않음 ---
    click_once = MagicMock(
        return_value={"reasoning": "click it", "action": "left_click", "target_description": "the button"}
    )
    ground_fn2 = MagicMock(return_value={"result": "positive", "point": [0.1, 0.1], "raw_response": "(100,100)"})
    _install_fake_planner_and_grounding(plan_next_action=click_once, ground=ground_fn2)
    try:
        env = _FakeEnv(execute_side_effect=RuntimeError("stop after one step"))
        run_episode(
            model="fake-model", planning_view="fake-planning-view", env=env,
            task={"web": "http://x", "ques": "click something"}, use_reflection=False,
            ground_kwargs={"max_pixels": 12345}, verbose=False,
        )
        check(
            "click -> ground_kwargs로 명시한 max_pixels가 기본값을 덮어씀",
            ground_fn2.call_args.kwargs.get("max_pixels") == 12345,
        )
    finally:
        _uninstall_fake_modules()

    # --- 종료 조건 2a: grounding이 wrong_format -> execution_failed로 즉시 종료, execute_action은 호출 안 됨 ---
    click_plan_fn = MagicMock(return_value={"reasoning": "click it", "action": "left_click", "target_description": "?"})
    bad_ground_fn = MagicMock(return_value={"result": "wrong_format", "point": None, "raw_response": "garbage"})
    _install_fake_planner_and_grounding(plan_next_action=click_plan_fn, ground=bad_ground_fn)
    try:
        env = _FakeEnv()
        outcome = run_episode(
            model="fake-model", planning_view="fake-planning-view", env=env,
            task={"web": "http://x", "ques": "click something"}, use_reflection=False, verbose=False,
        )
        check("grounding 실패 -> status failure", outcome["status"] == "failure")
        check("grounding 실패 -> reason execution_failed", outcome["reason"] == "execution_failed")
        check("grounding 실패 -> execute_action 호출 안 됨", len(env.execute_calls) == 0)
        check("grounding 실패 -> history에 _env_action_error 기록", "_env_action_error" in outcome["history"][-1])
    finally:
        _uninstall_fake_modules()

    # --- 종료 조건 2b: env.execute_action()이 예외를 던짐 -> execution_failed로 즉시 종료 ---
    type_plan_fn = MagicMock(return_value={"reasoning": "type it", "action": "type", "text": "hello"})
    _install_fake_planner_and_grounding(plan_next_action=type_plan_fn)
    try:
        env = _FakeEnv(execute_side_effect=RuntimeError("CDP boom"))
        outcome = run_episode(
            model="fake-model", planning_view="fake-planning-view", env=env,
            task={"web": "http://x", "ques": "type something"}, use_reflection=False, verbose=False,
        )
        check("execute_action 예외 -> status failure", outcome["status"] == "failure")
        check("execute_action 예외 -> reason execution_failed", outcome["reason"] == "execution_failed")
        check("execute_action 예외 -> 에러 메시지 보존", "CDP boom" in outcome["history"][-1].get("_env_action_error", ""))
    finally:
        _uninstall_fake_modules()

    # --- 안전장치: 계속 진행 가능한 액션만 나오면 max_steps에서 강제 종료 ---
    wait_forever_fn = MagicMock(return_value={"reasoning": "still waiting", "action": "wait"})
    _install_fake_planner_and_grounding(plan_next_action=wait_forever_fn)
    try:
        env = _FakeEnv()
        outcome = run_episode(
            model="fake-model", planning_view="fake-planning-view", env=env,
            task={"web": "http://x", "ques": "wait forever"}, use_reflection=False,
            max_steps=3, verbose=False,
        )
        check("max_steps 소진 -> status failure", outcome["status"] == "failure")
        check("max_steps 소진 -> reason max_steps_reached", outcome["reason"] == "max_steps_reached")
        check("max_steps 소진 -> steps == max_steps", outcome["steps"] == 3)
        check("max_steps 소진 -> execute_action이 딱 3번 호출됨", len(env.execute_calls) == 3)
    finally:
        _uninstall_fake_modules()

    # --- _plan_to_env_action 단위 테스트 (type/key/scroll/wait 필드 매핑, terminate 거부) ---
    _install_fake_planner_and_grounding()
    try:
        shot = _FakeScreenshot()
        check(
            "type -> {action, text} 그대로",
            _plan_to_env_action({"action": "type", "text": "hi"}, shot, "m") == {"action": "type", "text": "hi"},
        )
        check(
            "key -> {action, text} 그대로",
            _plan_to_env_action({"action": "key", "text": "Enter"}, shot, "m") == {"action": "key", "text": "Enter"},
        )
        check(
            "scroll -> text 기본값 down",
            _plan_to_env_action({"action": "scroll"}, shot, "m") == {"action": "scroll", "text": "down"},
        )
        check(
            "wait -> 추가 필드 없음",
            _plan_to_env_action({"action": "wait"}, shot, "m") == {"action": "wait"},
        )
        try:
            _plan_to_env_action({"action": "terminate", "status": "success"}, shot, "m")
            check("terminate는 _plan_to_env_action에서 거부", False)
        except ValueError:
            check("terminate는 _plan_to_env_action에서 거부", True)
    finally:
        _uninstall_fake_modules()

    n_fail = sum(1 for _, ok in checks if not ok)
    for name, ok in checks:
        print(("[OK]  " if ok else "[FAIL]") + " " + name)
    print(f"\n{len(checks) - n_fail}/{len(checks)} passed")
    if n_fail:
        raise SystemExit(1)


def _cli():
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="모델 인스턴스 하나로 planning(어댑터 꺼짐)과 grounding(어댑터 켜짐)이 "
        "둘 다 도는지 확인하거나(단발 테스트), 실제 WebVoyagerEnv에서 태스크 하나를 처음부터 "
        "끝까지(run_episode) 돌려보는 수동 확인용 CLI."
    )
    ap.add_argument("--selftest", action="store_true", help="실제 모델 없이 로직만 mock으로 검증")
    ap.add_argument("--adapter_dir", help="grounding LoRA 체크포인트 경로 (예: checkpoint-4130)")
    ap.add_argument(
        "--reflect", action="store_true",
        help="planning에 plan_with_reflection()(실행 전 비평 루프)을 쓴다. 안 주면 plan_next_action() "
        "만 씀. 에피소드/단발 테스트 내내 이 값이 고정됨(스텝마다 바뀌지 않음).",
    )
    ap.add_argument("--max_iterations", type=int, default=2, help="--reflect일 때 최대 재시도 횟수")

    single_step = ap.add_argument_group(
        "단발 테스트 (--image + --task, env/브라우저 없이 planning [+ grounding] 한 번만)"
    )
    single_step.add_argument("--image", help="테스트용 스크린샷 경로")
    single_step.add_argument("--task", help="planning 테스트용 태스크 지시문")
    single_step.add_argument(
        "--test_grounding_instruction",
        help="지정하면, 같은 모델 인스턴스로 이 문구에 대해 grounding까지 이어서 테스트",
    )

    episode = ap.add_argument_group(
        "run_episode 모드 (--run_episode, 실제 WebVoyagerEnv + Selenium/Chrome 필요)"
    )
    episode.add_argument(
        "--run_episode", action="store_true",
        help="planning -> grounding -> env.execute_action() 전체 루프를 실제로 실행",
    )
    episode.add_argument("--url", help="--run_episode용 시작 URL (--tasks_jsonl 대신 직접 지정)")
    episode.add_argument("--instruction", help="--run_episode용 태스크 지시문 (--url과 함께 사용)")
    episode.add_argument("--tasks_jsonl", help="WebVoyager 태스크 jsonl 경로 (--url/--instruction 대신)")
    episode.add_argument("--web_name", help="--tasks_jsonl에서 필터링할 사이트 이름 (예: Wikipedia)")
    episode.add_argument("--max_steps", type=int, default=50, help="run_episode 안전장치 상한 (기본 50)")
    episode.add_argument(
        "--no_headless", dest="headless", action="store_false", default=True,
        help="브라우저 창 띄워서 눈으로 확인 (GUI 있는 로컬 환경에서만)",
    )
    episode.add_argument("--width", type=int, default=1280)
    episode.add_argument("--height", type=int, default=800)

    args = ap.parse_args()

    if args.selftest:
        _run_mock_selftest()
        return

    if args.run_episode:
        if not (args.tasks_jsonl or (args.url and args.instruction)):
            raise SystemExit("--run_episode에는 --tasks_jsonl 또는 (--url + --instruction) 필요")

        from env_webvoyager import WebVoyagerEnv, load_webvoyager_tasks

        if args.tasks_jsonl:
            tasks = load_webvoyager_tasks(args.tasks_jsonl, web_name=args.web_name)
            if not tasks:
                raise SystemExit("조건에 맞는 태스크가 없음")
            task = tasks[0]
        else:
            task = (args.url, args.instruction)

        print("=== 모델 로딩 시작 (아래 '[qwen.py] Loading ...' 메시지가 한 번만 찍혀야 함) ===")
        model, planning_view = load_shared_model(args.adapter_dir)
        print("=== 로딩 끝 ===\n")

        env = WebVoyagerEnv(window_size=(args.width, args.height), headless=args.headless)
        try:
            outcome = run_episode(
                model, planning_view, env, task,
                use_reflection=args.reflect,
                max_steps=args.max_steps,
                reflection_max_iterations=args.max_iterations,
            )
        finally:
            env.close()

        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return

    if not args.image or not args.task:
        raise SystemExit("--image와 --task 필요 (또는 --run_episode / --selftest)")

    from PIL import Image

    from planner import plan_next_action, plan_with_reflection

    print("=== 모델 로딩 시작 (아래 '[qwen.py] Loading ...' 메시지가 한 번만 찍혀야 함) ===")
    model, planning_view = load_shared_model(args.adapter_dir)
    print("=== 로딩 끝 ===\n")

    screenshot = Image.open(args.image)

    print("--- planning (어댑터 꺼짐, base 모델처럼 동작) ---")
    if args.reflect:
        plan = plan_with_reflection(planning_view, args.task, screenshot, max_iterations=args.max_iterations)
    else:
        plan = plan_next_action(planning_view, args.task, screenshot)
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if args.test_grounding_instruction:
        from gui_grounding import ground

        print("\n--- grounding (어댑터 켜짐, 기본 상태) ---")
        result = ground(model, args.test_grounding_instruction, screenshot)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()