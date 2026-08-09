"""
agent_loop.py

전체 에이전트 루프. 두 단계로 구성된다:

  1) 모델 인스턴스 하나 + LoRA "다중 어댑터" 스위칭 (_AdapterView / load_shared_model) -
     grounding용, planner용, reflector용 모델을 각각 별도 QwenVLModel 인스턴스로 띄웠다가
     실제로 RAM(16GB, CUDA 없는 CPU 환경)이 터진 적이 있어서(터미널/프로세스 두 개가 각각
     모델을 물고 있다가 두 번째 로딩에서 터짐), 이 문제를 "backbone은 딱 한 번만 로드하고,
     그 위에 이름이 다른 LoRA 어댑터를 필요한 만큼 얹어둔 뒤(peft의 load_adapter), 역할마다
     필요한 순간에만 set_adapter()/disable_adapter()로 전환한다"는 방식으로 구조적으로
     없앤다.

  2) planning(제안 어댑터) <-> reflection(비평 어댑터) <-> grounding(어댑터) <->
     env.execute_action() 실제 step loop (run_episode) - task 하나를 처음부터 끝까지
     돌린다.

[역할 세 개, 각자 독립적으로 어댑터 선택 가능 - 없으면 base로 자동 폴백]
이 에이전트는 세 가지 역할로 모델을 부르고, 셋 다 "체크포인트 경로를 주면 그 LoRA로,
안 주면 base(어댑터 없음)로" 동작한다 - 서로 완전히 독립적으로 정해진다:
  - grounding: 화면 좌표를 찍는 역할. LoRA 어댑터 있음(checkpoint-4130, peft 내부에서
    이름이 "default" - qwen.py의 load_model_and_processor()가 PeftModel.from_pretrained()를
    adapter_name 지정 없이 부르기 때문에 peft가 자동으로 "default"라는 이름을 붙인다).
    다른 역할과 달리 이건 사실상 항상 있다고 가정(코드상으로는 이것도 이론적으로 없을 수
    있지만, 이 backbone에 어댑터가 하나도 안 얹히면 peft 모델 자체가 안 만들어져서
    disable_adapter()/set_adapter()가 없는 plain 모델이 되고, 그러면 애초에
    _AdapterView가 쓸 수 없다 - 그래서 grounding_adapter_dir는 필수 인자로 남겨뒀다. 다만
    grounding_view 자체가 어댑터 없이 base로 도는 걸 막는 코드는 없다 - adapter_name=None
    을 주면 grounding_view도 base로 돌 수 있다).
  - planner: 다음 액션을 제안하는 역할. planner_adapter_dir를 주면 그 LoRA("planner"라는
    이름으로 얹음)로, 안 주면 base로.
  - reflector: planner가 제안한 액션을 실행 전에 비평하는 역할. reflector_adapter_dir를
    주면 그 LoRA("reflector"라는 이름으로 얹음)로, 안 주면 base로.

[planner.py 쪽 대응 - reflection_model 파라미터]
planner.py의 plan_with_reflection(qwen_model, ..., reflection_model=None)은 제안
(plan_next_action)에 qwen_model을, 비평(_reflect_on_plan)에 reflection_model(안 주면
qwen_model 재사용)을 쓴다. agent_loop.py는 여기에 planner_view/reflector_view를 각각
넘긴다 - 이 두 view가 실제로 어댑터를 얹었는지 base인지는 planner.py가 전혀 몰라도 되고
(duck-typing, .generate()만 있으면 됨), agent_loop.py의 load_shared_model()이 CLI
인자(--planner_adapter_dir/--reflector_adapter_dir)로 넘어온 값에 따라 그때그때 결정한다.

[왜 프롬프트 병합이 아니라 어댑터 전환인가 - 요약]
grounding LoRA는 coord_utils.PROMPT_TEMPLATE 하나, "(x,y)" 한 줄 출력이라는 아주 좁은
포맷으로만 SFT됐다(train.py). planning(JSON, reasoning, target_description 등)은 이
LoRA가 학습에서 한 번도 못 본 포맷이라, 어댑터를 켠 채로 planning을 시키면 - 실제로
planner.py의 _action_schema_valid() 회귀 테스트로 확인된 사례(어댑터 켠 채로 planning시
좌표 tool-call 포맷이 그대로 새어나옴)와 같은 방식으로 - planning이 오히려 더 망가질
위험이 있다. planner/reflector 전용 LoRA도 마찬가지 이유로 서로 다른 포맷/역할에 맞게
학습됐을 가능성이 높다. 그래서 역할마다 자기 어댑터(또는 어댑터 없음)로만 돌리고, 전부
같은 프로세스, 같은 모델 인스턴스 안에서 전환한다.

[vision encoder는 항상 동일하다 - "특정 어댑터가 이미지를 더 잘 본다"는 주장에 대한 반박 근거]
train.py의 LoraConfig.target_modules는 ["q_proj","k_proj","v_proj","o_proj"] - LLM 디코더
쪽 attention만 건드리고 vision encoder는 전혀 안 건드린다. 즉 어댑터를 바꿔 껴도
"이미지를 보는" 부분의 가중치는 항상 동일하다 - 바뀌는 건 그 시각 정보를 어떤 출력
포맷으로 뽑아내느냐뿐이다. 이 가정이 각 LoRA에도 그대로 적용된다는 보장은 train.py가
그 LoRA들도 같은 target_modules로 학습한다는 전제 하에서다.

[reflection 모드 on/off - 언제, 어떻게 결정되는가]
plan_next_action() <-> plan_with_reflection() 중 뭘 쓸지는 에피소드 "시작 시점에 한 번"
CLI(`--reflect`)로 고르고, 그 값이 에피소드 끝까지 고정된다 - 스텝마다 동적으로(예: 이전
액션 실패 여부에 따라) 바뀌지 않는다. run_episode()의 use_reflection 파라미터가 이 값을
그대로 받는다. use_reflection=False면 reflector_view는 아예 쓰이지 않는다(plan_next_action
만 호출되므로).

[execute 루프 - run_episode()]
task 하나에 대해 env.reset() -> (planning -> [클릭류만 grounding으로 좌표 변환] ->
env.execute_action()) 반복 -> 아래 두 조건 중 하나에서 종료:
  1. 모델이 done/finish 신호를 냄 (planner가 action="terminate" 반환)
  2. 액션 실행 자체가 실패함 (grounding이 좌표를 못 뽑았거나, env.execute_action()이
     예외를 던짐)
두 조건 모두 즉시 루프를 멈춘다. max_steps는 이 두 조건과 별개로 존재하는 안전장치일
뿐이다(무한 루프 방지) - 필요 없으면 max_steps=None으로 끌 수 있다.

[해상도 - grounding LoRA가 학습된 해상도(700,000)로 기본 고정됨]
load_shared_model()과 _plan_to_env_action()은 max_pixels를 명시하지 않으면 둘 다
_GROUNDING_TRAINED_MAX_PIXELS(700,000)를 기본값으로 쓴다 - checkpoint-4130이 실제로 그
해상도로 파인튜닝됐는데, 예전에 qwen.py 기본값(501,760)과 안 맞아서 grounding 정확도가
크게 깎인 전례가 있다(planner.py의 _cli() 주석 참고, "해상도 confound"). 이미지는
generate() 호출 전에 두 군데에서 각각 해상도 상한이 걸린다 - (1) gui_grounding.ground()가
호출 전에 직접 하는 PIL resize(ground()의 max_pixels 인자), (2) QwenVLModel/processor가
구성 시점에 들고 있는 (min_pixels, max_pixels)로 generate() 안에서 다시 도는 smart_resize.
둘 중 하나만 700,000으로 고치면 더 작은 쪽(안 고친 쪽의 기존 기본값)이 이겨서 무효화되므로,
_GROUNDING_TRAINED_MAX_PIXELS 상수 하나를 두 군데가 공유해서 항상 같이 바뀌게 했다. 다른
체크포인트를 쓴다면 load_shared_model(..., max_pixels=...)와
run_episode(..., ground_kwargs={"max_pixels": ...})를 함께 명시적으로 덮어쓸 것(하나만
바꾸면 안 됨).

[사용법]
    # 세 어댑터 다 독립적으로 선택 - 없는 건 None으로 두면 자동으로 base
    model, grounding_view, planner_view, reflector_view = load_shared_model(
        grounding_adapter_dir="checkpoint-4130",
        planner_adapter_dir="planner-checkpoint-XXXX",   # 아직 없으면 None
        reflector_adapter_dir=None,                       # 아직 없으면 None -> base
    )

    # 단발 테스트 (planning 한 번, grounding 한 번) - env/실제 브라우저 없이
    plan = plan_with_reflection(
        planner_view, task, screenshot, history, reflection_model=reflector_view,
    )
    result = ground(grounding_view, plan["target_description"], screenshot)

    # 실제 태스크 하나를 처음부터 끝까지: env_webvoyager.WebVoyagerEnv와 엮어서
    from env_webvoyager import WebVoyagerEnv
    env = WebVoyagerEnv(headless=True)
    outcome = run_episode(
        grounding_view, planner_view, reflector_view, env,
        task={"web": "https://en.wikipedia.org", "ques": "Find who wrote the Python article."},
        use_reflection=True,   # CLI --reflect와 동일한 스위치, 에피소드 내내 고정
    )
    env.close()

    # WebVoyager 태스크 jsonl 전체를 배치로 돌려서 결과를 jsonl로 저장 (run_batch, resume 지원)
    from env_webvoyager import load_webvoyager_tasks
    tasks = load_webvoyager_tasks("/srv/project/data/processed/WebVoyager_data.jsonl")
    stats = run_batch(
        grounding_view, planner_view, reflector_view, env, tasks,
        output_jsonl_path="/srv/project/data/results/webvoyager_results.jsonl",
        use_reflection=True,
    )
    env.close()
    # CLI로 똑같이: python agent_loop.py --adapter_dir checkpoint-4130 --run_episode \
    #   --tasks_jsonl /srv/project/data/processed/WebVoyager_data.jsonl \
    #   --output_jsonl /srv/project/data/results/webvoyager_results.jsonl --reflect

model 하나만 메모리에 올라간다 - 세 view는 전부 같은 model을 감싸서 .generate() 호출
직전에 (peft의 set_adapter로) 자기 역할에 맞는 어댑터로 바꾸거나 (disable_adapter로)
어댑터를 끄는 얇은 프록시일 뿐, 별도 모델이 아니다.
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

# peft의 PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)는
# adapter_name을 지정하지 않으면 내부적으로 "default"라는 이름을 자동으로 붙인다.
# qwen.py의 load_model_and_processor()가 정확히 이렇게 호출하므로(adapter_name 인자 없이),
# QwenVLModel(adapter_dir=...)로 얹은 첫 어댑터(=grounding LoRA)의 peft 내부 이름은
# 항상 "default"다. 나머지 어댑터(planner/reflector)는 load_shared_model()이 직접
# model.model.load_adapter(path, adapter_name=..., ...)로 얹으면서 이름을 붙인다.
_GROUNDING_ADAPTER_NAME = "default"
_PLANNER_ADAPTER_NAME = "planner"
_REFLECTOR_ADAPTER_NAME = "reflector"


class _AdapterView:
    """
    QwenVLModel(내부에 peft.PeftModel, 여러 이름의 LoRA 어댑터가 얹혀 있을 수 있음)을
    감싸서, .generate() 호출 직전에 지정된 어댑터로 전환(또는 어댑터를 아예 끔)한 뒤
    생성하는 얇은 프록시.

    planner.py의 plan_next_action()/plan_with_reflection() 등, gui_grounding.py의
    ground() 등은 인자로 받은 모델 객체에 duck-typing으로
    .generate(messages, max_new_tokens=..., temperature=..., top_p=...)만 있으면 되므로,
    이 프록시를 그대로 넘기면 그 파일들을 전혀 수정하지 않고도 "역할에 맞는 어댑터가 켜진
    상태"로 동작시킬 수 있다.

    adapter_name=None이면 peft의 disable_adapter() 컨텍스트로 모든 어댑터를 일시적으로
    끄고 생성한다(base 모델처럼 동작) - 해당 역할의 어댑터 경로를 안 준 경우(아직 학습
    안 됨 등) load_shared_model()이 이 값을 넘긴다.
    adapter_name="default"/"planner"/"reflector" 등 문자열이면
    peft.PeftModel.set_adapter(name)으로 그 이름의 어댑터를 활성화한 뒤 생성한다 -
    set_adapter는 컨텍스트 매니저가 아니라 모델 객체에 지속되는 상태 변경이라서, 매
    generate() 호출 직전에 매번 다시 불러서 "직전에 다른 view가 어댑터를 바꿔놨을 수도
    있다"는 가정을 항상 깨끗하게 만족시킨다(그렇게 안 하면, 예를 들어
    planner_view.generate() 직후에 grounding_view.generate()를 호출했을 때
    grounding_view가 자기도 모르게 "planner" 어댑터를 켠 채로 좌표를 찍는 - 조용히 틀린
    결과를 내는 - 버그가 생긴다).

    qwen_model.model이 peft.PeftModel이 아니면(=grounding_adapter_dir 없이 로드된 경우)
    disable_adapter()가 없다 - 이 경우를 generate() 호출 시점이 아니라 생성 시점에 바로
    걸러내서, 에러가 나더라도 어디서 뭐가 잘못됐는지 바로 알 수 있게 한다.
    """

    def __init__(self, qwen_model: "QwenVLModel", adapter_name: str | None):
        if not hasattr(qwen_model.model, "disable_adapter"):
            raise TypeError(
                "_AdapterView는 LoRA 어댑터가 얹힌(peft.PeftModel) QwenVLModel에만 쓸 수 있음 - "
                f"qwen_model.model의 타입이 {type(qwen_model.model).__name__}이라 disable_adapter()가 "
                "없음. 어댑터가 아예 없는 모델이면 _AdapterView로 감쌀 필요 없이 그 모델을 "
                "그대로 넘기면 됨."
            )
        self._qwen_model = qwen_model
        self._adapter_name = adapter_name

    def generate(
        self, messages: list, max_new_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0
    ) -> str:
        if self._adapter_name is None:
            with self._qwen_model.model.disable_adapter():
                return self._qwen_model.generate(
                    messages, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                )
        self._qwen_model.model.set_adapter(self._adapter_name)
        return self._qwen_model.generate(
            messages, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
        )


def load_shared_model(
    grounding_adapter_dir: str,
    planner_adapter_dir: str | None = None,
    reflector_adapter_dir: str | None = None,
    **qwen_model_kwargs,
):
    """
    QwenVLModel을 grounding LoRA를 얹은 상태로 "딱 한 번만" 로드하고, planner_adapter_dir/
    reflector_adapter_dir가 주어지면 같은 backbone 위에 각각 "planner"/"reflector"라는
    이름으로 LoRA 어댑터를 추가로 얹는다(peft.PeftModel.load_adapter() - LoRA 가중치만
    추가되는 거라 backbone을 또 로드하는 게 아님, 메모리 부담이 거의 없다).
    (model, grounding_view, planner_view, reflector_view) 4-튜플을 반환한다.

    Args:
        grounding_adapter_dir: grounding LoRA 체크포인트 경로 (예: "checkpoint-4130").
            항상 필수 - 이게 없으면 로드할 backbone 자체가 peft 모델이 아니게 되어(=
            _AdapterView가 요구하는 disable_adapter()/set_adapter()가 없어서) 세 view
            생성이 전부 실패한다.
        planner_adapter_dir: planner 전용 LoRA 체크포인트 경로. 없으면(기본값 None)
            planner_view가 base(어댑터 없음)로 동작한다 - 학습이 끝나면 경로만 넘기면
            자동으로 전용 어댑터를 쓰기 시작한다(코드 변경 불필요).
        reflector_adapter_dir: reflector 전용 LoRA 체크포인트 경로. 없으면(기본값 None)
            reflector_view가 base(어댑터 없음)로 동작한다. planner_adapter_dir와 완전히
            독립적으로 결정된다 - 둘 중 하나만 있어도, 둘 다 없어도, 둘 다 있어도 된다.
        qwen_model_kwargs: QwenVLModel(...)에 그대로 전달됨(min_pixels 등 조정용). max_pixels를
            명시하지 않으면 _GROUNDING_TRAINED_MAX_PIXELS(700,000)가 기본으로 들어간다(파일
            상단 상수 주석 참고). _plan_to_env_action()의 ground_kwargs 기본값과 짝을
            이루니, 둘 중 하나만 바꾸지 말고 같이 바꿀 것(다른 체크포인트로 바꿀 때 등).

    Returns:
        (
            model: QwenVLModel,          # 디버깅/직접 접근용 - 보통은 안 써도 됨
            grounding_view: _AdapterView,  # gui_grounding.ground()에 넘길 것
            planner_view: _AdapterView,    # planner.py의 plan_next_action()/
                                            # plan_with_reflection()의 qwen_model 인자
            reflector_view: _AdapterView,  # plan_with_reflection()의 reflection_model 인자
        )
    """
    from qwen import QwenVLModel  # 실제 로드 시점에만 임포트 (mock 테스트는 이 경로를 안 탐)

    qwen_model_kwargs.setdefault("max_pixels", _GROUNDING_TRAINED_MAX_PIXELS)
    model = QwenVLModel(adapter_dir=grounding_adapter_dir, **qwen_model_kwargs)

    grounding_view = _AdapterView(model, _GROUNDING_ADAPTER_NAME)

    if planner_adapter_dir:
        print(f"[agent_loop.py] Loading planner LoRA adapter from {planner_adapter_dir} ...")
        model.model.load_adapter(planner_adapter_dir, adapter_name=_PLANNER_ADAPTER_NAME, is_trainable=False)
        planner_view = _AdapterView(model, _PLANNER_ADAPTER_NAME)
    else:
        planner_view = _AdapterView(model, None)

    if reflector_adapter_dir:
        print(f"[agent_loop.py] Loading reflector LoRA adapter from {reflector_adapter_dir} ...")
        model.model.load_adapter(reflector_adapter_dir, adapter_name=_REFLECTOR_ADAPTER_NAME, is_trainable=False)
        reflector_view = _AdapterView(model, _REFLECTOR_ADAPTER_NAME)
    else:
        reflector_view = _AdapterView(model, None)

    return model, grounding_view, planner_view, reflector_view


# ---------------------------------------------------------------------------
# step loop: planning(제안 어댑터) <-> reflection(비평 어댑터) -> [grounding] ->
#            env.execute_action()
# ---------------------------------------------------------------------------

_CLICK_ACTIONS = ("left_click", "double_click", "right_click")


def _plan_to_env_action(plan: dict, screenshot, grounding_view, ground_kwargs: dict | None = None) -> dict:
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
      쪽 상수 주석 참고(model 쪽도 같이 이 값으로 맞춰져 있어야 실제로 적용됨). grounding_view는
      load_shared_model()이 만든 _AdapterView(grounding 어댑터로 고정)를 넘길 것 - raw
      QwenVLModel을 직접 넘기면 직전에 다른 역할(planner 등)이 바꿔놓은 어댑터가 그대로 켜져
      있을 수 있어서 안 됨.
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
        result = ground(grounding_view, target, screenshot, **gk)
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
    grounding_view,
    planner_view,
    reflector_view,
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
        grounding_view: load_shared_model()이 만든, grounding 어댑터로 고정된 _AdapterView.
            _plan_to_env_action()이 gui_grounding.ground()를 부를 때 이걸 씀.
        planner_view: load_shared_model()이 만든 _AdapterView(planner 어댑터가 있으면
            그걸로, 없으면 base). planner.py의 plan_next_action()/plan_with_reflection()의
            qwen_model 인자(=제안 담당)로 씀.
        reflector_view: load_shared_model()이 만든 _AdapterView(reflector 어댑터가 있으면
            그걸로, 없으면 base). use_reflection=True일 때만 plan_with_reflection()의
            reflection_model 인자(=비평 담당)로 씀 - use_reflection=False면 아예 안 쓰임.
        env: env_webvoyager.WebVoyagerEnv 인스턴스(reset/execute_action 인터페이스만
            맞으면 mock으로 대체 가능 - 아래 _run_mock_selftest 참고).
        task: env.reset(task)에 그대로 전달됨 ({"web":..., "ques":...} dict 또는
            (url, instruction) 튜플).
        use_reflection: True면 매 스텝 plan_with_reflection()(제안=planner_view/비평=
            reflector_view)을, False면 plan_next_action()(planner_view만 씀)을 쓴다.
            에피소드 시작 시 한 번 정해지고 끝까지 고정됨(파일 상단 docstring의
            "reflection 모드 on/off" 참고 - 스텝마다 바뀌지 않음). CLI에서는 --reflect
            플래그로 이 값을 넘긴다.
        max_steps: 두 종료 조건과 별개인 안전장치(무한 루프 방지). None이면 끔.
        reflection_max_iterations: use_reflection=True일 때 plan_with_reflection()의
            max_iterations로 그대로 전달.
        planner_kwargs: plan_next_action()/plan_with_reflection() 둘 다에 추가로 넘길
            kwargs(예: max_new_tokens). max_iterations/reflection_model은 여기 넣지 말 것
            (각각 reflection_max_iterations 인자와 reflector_view로 이미 따로 전달됨 -
            여기 넣으면 중복 키워드 인자 에러가 남).
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
                planner_view, task_instruction, screenshot,
                history_actions=history,
                max_iterations=reflection_max_iterations,
                reflection_model=reflector_view,
                **planner_kw,
            )
        else:
            plan = plan_next_action(
                planner_view, task_instruction, screenshot,
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
            env_action = _plan_to_env_action(plan, screenshot, grounding_view, ground_kwargs=ground_kwargs)
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
# 배치 실행: WebVoyager jsonl(예: /srv/project/data/processed/WebVoyager_data.jsonl)의
# 태스크 여러 개를 순서대로 run_episode()로 돌리고, 결과를 jsonl로 한 줄씩 저장한다.
# ---------------------------------------------------------------------------
def run_batch(
    grounding_view,
    planner_view,
    reflector_view,
    env,
    tasks: list,
    output_jsonl_path: str,
    use_reflection: bool,
    max_steps: int | None = 50,
    reflection_max_iterations: int = 2,
    planner_kwargs: dict | None = None,
    ground_kwargs: dict | None = None,
    resume: bool = True,
    verbose: bool = True,
) -> dict:
    """
    WebVoyager 태스크 jsonl(env_webvoyager.load_webvoyager_tasks()가 읽는 그 포맷,
    {"web_name":..., "id":..., "ques":..., "web":...} 레코드들의 리스트)을 순서대로
    run_episode()에 하나씩 넘겨 돌리고, 매 태스크가 끝날 때마다 결과를
    output_jsonl_path에 한 줄(JSON)씩 append + flush한다 - 수백 개짜리 배치를 실제
    브라우저로 몇 시간씩 돌릴 수 있어서, 중간에 죽어도 그때까지 결과는 디스크에 남아있게
    하기 위함(전부 메모리에 모았다가 끝에 한 번에 쓰면 중간에 죽는 순간 그동안의 결과가
    전부 날아간다).

    [resume - 이미 끝난 태스크는 건너뛰고 이어서 실행]
    output_jsonl_path가 이미 존재하면(이전 실행이 중간에 끊겼거나 같은 커맨드를 다시
    돌리는 경우), 그 파일에 이미 기록된 task_id들을 먼저 읽어서 이번 배치에서는
    건너뛴다(resume=True, 기본값). 처음부터 전부 다시 돌리고 싶으면 resume=False를
    주거나 output_jsonl_path를 지우고 시작할 것. 이미 있는 파일에는 겹치는 task_id를
    다시 쓰지 않고 그대로 두고, 새로 처리한 태스크만 뒤에 이어서 append한다.

    [태스크 하나의 예외가 배치 전체를 안 죽이게]
    run_episode() 자체는 이미 "액션 실행 실패"(grounding 실패/env.execute_action 예외)를
    자체적으로 잡아서 정상적인 outcome(status="failure", reason="execution_failed")으로
    돌려준다 - 그건 여기서 또 잡을 필요가 없다. 여기서 추가로 감싸는 try/except는 그보다
    "이전" 단계, 즉 env.reset()이 Chrome 크래시/네트워크 완전 단절 등으로 예외를 던지거나
    plan_next_action()/plan_with_reflection() 자체가 (마지막 폴백조차 못 갈 정도로) 예외를
    던지는, run_episode()가 원래 처리 대상으로 두지 않은 종류의 실패를 잡기 위함이다. 이
    경우 해당 태스크만 {"status": "error", "error": "<메시지>"}로 기록하고 다음 태스크로
    넘어간다 - 태스크 597개 중 1개가 이런 이유로 죽었다고 나머지 596개를 못 돌리면 안 됨.

    env는 태스크 사이에 재사용한다 - WebVoyagerEnv.reset()이 이미 "driver가 있으면 먼저
    close()하고 새로 띄운다"는 로직을 갖고 있어서(env_webvoyager.py 참고), 매 태스크마다
    새 WebVoyagerEnv 인스턴스를 만들 필요 없이 env.reset(task)만 반복 호출하면 된다.

    Args:
        tasks: env_webvoyager.load_webvoyager_tasks()가 반환하는 것과 같은 dict 리스트.
            각 태스크는 "id" 필드가 있어야 resume 스킵/결과 식별이 정상 동작한다.
        output_jsonl_path: 결과를 append할 jsonl 파일 경로. 각 줄은
            {"task_id", "web_name", "web", "ques"} + run_episode()의 반환 dict 전체
            (또는 위 예외 케이스의 {"status": "error", "error": ...}).
        나머지 인자는 run_episode()에 각 태스크마다 그대로 전달됨(파라미터 설명은
        run_episode() 참고).

    Returns:
        {"total": int, "completed": int, "skipped": int, "succeeded": int,
         "failed": int, "errored": int}
        completed = skipped를 제외하고 이번에 실제로 처리한 태스크 수 (succeeded+failed+errored).
    """
    import json
    import os

    already_done = set()
    if resume and os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("task_id"):
                    already_done.add(row["task_id"])

    stats = {"total": len(tasks), "completed": 0, "skipped": 0, "succeeded": 0, "failed": 0, "errored": 0}

    with open(output_jsonl_path, "a", encoding="utf-8") as out_f:
        for i, task in enumerate(tasks, start=1):
            task_id = task.get("id")

            if resume and task_id is not None and task_id in already_done:
                stats["skipped"] += 1
                if verbose:
                    print(f"[agent_loop] [{i}/{len(tasks)}] {task_id!r} 이미 완료됨 - 건너뜀")
                continue

            if verbose:
                print(f"[agent_loop] [{i}/{len(tasks)}] {task_id!r} 시작: {task.get('ques', '')!r}")

            row = {
                "task_id": task_id,
                "web_name": task.get("web_name"),
                "web": task.get("web"),
                "ques": task.get("ques"),
            }
            try:
                outcome = run_episode(
                    grounding_view, planner_view, reflector_view, env, task,
                    use_reflection=use_reflection,
                    max_steps=max_steps,
                    reflection_max_iterations=reflection_max_iterations,
                    planner_kwargs=planner_kwargs,
                    ground_kwargs=ground_kwargs,
                    verbose=verbose,
                )
                row.update(outcome)
                if outcome.get("status") == "success":
                    stats["succeeded"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                row["status"] = "error"
                row["error"] = str(e)
                stats["errored"] += 1
                if verbose:
                    print(f"[agent_loop] [{i}/{len(tasks)}] {task_id!r} 처리 중 예외 - {e}")

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            stats["completed"] += 1

    if verbose:
        print(f"[agent_loop] 배치 완료: {stats}")
    return stats


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

    # =======================================================================
    # _AdapterView
    # =======================================================================
    fake_qwen_model = MagicMock()
    fake_qwen_model.generate.return_value = "raw response text"

    base_view = _AdapterView(fake_qwen_model, None)
    result = base_view.generate([{"role": "user", "content": []}], max_new_tokens=100, temperature=0.3)

    check("adapter_name=None -> generate 결과가 그대로 반환됨", result == "raw response text")
    check("adapter_name=None -> disable_adapter()가 호출됨", fake_qwen_model.model.disable_adapter.called)
    check("adapter_name=None -> set_adapter는 호출 안 됨", not fake_qwen_model.model.set_adapter.called)
    check("adapter_name=None -> disable_adapter 컨텍스트 안에서 실제 generate가 호출됨", fake_qwen_model.generate.called)

    _, kwargs = fake_qwen_model.generate.call_args
    check("max_new_tokens이 그대로 전달됨", kwargs.get("max_new_tokens") == 100)
    check("temperature가 그대로 전달됨", kwargs.get("temperature") == 0.3)

    cm = fake_qwen_model.model.disable_adapter.return_value
    check("disable_adapter 컨텍스트 __enter__ 호출됨", cm.__enter__.called)
    check("disable_adapter 컨텍스트 __exit__ 호출됨", cm.__exit__.called)

    fake_qwen_model2 = MagicMock()
    fake_qwen_model2.generate.return_value = "planner response"
    planner_view_t = _AdapterView(fake_qwen_model2, "planner")
    result2 = planner_view_t.generate([{"role": "user", "content": []}])
    check("adapter_name='planner' -> generate 결과 반환", result2 == "planner response")
    check(
        "adapter_name='planner' -> set_adapter('planner') 호출됨",
        fake_qwen_model2.model.set_adapter.call_args.args == ("planner",),
    )
    check("adapter_name='planner' -> disable_adapter는 호출 안 됨", not fake_qwen_model2.model.disable_adapter.called)

    fake_qwen_model3 = MagicMock()
    fake_qwen_model3.generate.return_value = "reflector response"
    reflector_view_t = _AdapterView(fake_qwen_model3, "reflector")
    reflector_view_t.generate([{"role": "user", "content": []}])
    check(
        "adapter_name='reflector' -> set_adapter('reflector') 호출됨",
        fake_qwen_model3.model.set_adapter.call_args.args == ("reflector",),
    )

    fake_qwen_model4 = MagicMock()
    fake_qwen_model4.generate.return_value = "grounding response"
    grounding_view_t = _AdapterView(fake_qwen_model4, "default")
    grounding_view_t.generate([{"role": "user", "content": []}])
    check(
        "adapter_name='default' -> set_adapter('default') 호출됨",
        fake_qwen_model4.model.set_adapter.call_args.args == ("default",),
    )

    fake_qwen_model5 = MagicMock()
    fake_qwen_model5.generate.return_value = "x"
    view5 = _AdapterView(fake_qwen_model5, "planner")
    view5.generate([{"role": "user", "content": []}])
    view5.generate([{"role": "user", "content": []}])
    check("generate 2번 호출 -> set_adapter도 2번 호출", fake_qwen_model5.model.set_adapter.call_count == 2)

    class _PlainModelNoAdapter:
        pass

    class _FakeNoAdapterQwenModel:
        def __init__(self):
            self.model = _PlainModelNoAdapter()

    try:
        _AdapterView(_FakeNoAdapterQwenModel(), None)
        check("어댑터 없는 모델 -> 생성 시점에 TypeError", False)
    except TypeError as e:
        check("어댑터 없는 모델 -> 생성 시점에 TypeError", "disable_adapter" in str(e))

    # =======================================================================
    # load_shared_model
    # =======================================================================
    construct_calls = []
    load_adapter_calls = []

    class _FakePeftModel:
        def __init__(self):
            self.disable_adapter = MagicMock()
            self.set_adapter = MagicMock()

        def load_adapter(self, path, adapter_name, is_trainable=False):
            load_adapter_calls.append(
                {"path": path, "adapter_name": adapter_name, "is_trainable": is_trainable}
            )

    class _FakeQwenVLModel:
        def __init__(self, **kwargs):
            construct_calls.append(kwargs)
            self.model = _FakePeftModel()

        def generate(self, *a, **kw):
            return "ok"

    fake_qwen_module = types.ModuleType("qwen")
    fake_qwen_module.QwenVLModel = _FakeQwenVLModel
    sys.modules["qwen"] = fake_qwen_module
    try:
        # --- planner_adapter_dir/reflector_adapter_dir 둘 다 없음 -> 둘 다 base로 폴백 ---
        model, grounding_view, planner_view, reflector_view = load_shared_model(
            grounding_adapter_dir="checkpoint-4130", min_pixels=123,
        )
        check("QwenVLModel 생성 호출이 딱 1번", len(construct_calls) == 1)
        check("grounding_adapter_dir가 adapter_dir로 전달됨", construct_calls[0].get("adapter_dir") == "checkpoint-4130")
        check("나머지 kwargs(min_pixels 등)도 그대로 전달됨", construct_calls[0].get("min_pixels") == 123)
        check(
            "max_pixels 미지정시 학습 해상도(700,000)가 기본으로 들어감",
            construct_calls[0].get("max_pixels") == _GROUNDING_TRAINED_MAX_PIXELS == 700_000,
        )
        check("둘 다 미지정 -> load_adapter 호출 안 됨", len(load_adapter_calls) == 0)
        check("planner_view.adapter_name은 None(base)", planner_view._adapter_name is None)
        check("reflector_view.adapter_name은 None(base)", reflector_view._adapter_name is None)

        model.model.set_adapter.reset_mock()
        grounding_view.generate([{"role": "user", "content": []}])
        check(
            "grounding_view -> set_adapter('default') 호출됨(항상 grounding 어댑터 고정)",
            model.model.set_adapter.call_args.args == (_GROUNDING_ADAPTER_NAME,) == ("default",),
        )

        model.model.disable_adapter.reset_mock()
        planner_view.generate([{"role": "user", "content": []}])
        check("planner_adapter_dir 없을 때 -> planner_view가 disable_adapter(base) 사용", model.model.disable_adapter.called)

        model.model.disable_adapter.reset_mock()
        reflector_view.generate([{"role": "user", "content": []}])
        check("reflector_adapter_dir 없을 때 -> reflector_view가 disable_adapter(base) 사용", model.model.disable_adapter.called)

        check("model/grounding_view/planner_view/reflector_view가 같은 backbone 공유", grounding_view._qwen_model is model)

        # --- max_pixels 명시하면 기본값(700,000)으로 덮어써지지 않음 ---
        construct_calls.clear()
        load_shared_model(grounding_adapter_dir="checkpoint-4130", max_pixels=999_999)
        check(
            "max_pixels 명시하면 기본값(700,000)으로 덮어써지지 않고 그대로 전달됨",
            construct_calls[0].get("max_pixels") == 999_999,
        )

        # --- planner_adapter_dir만 지정, reflector_adapter_dir는 미지정 -> 서로 독립적으로 반영 ---
        construct_calls.clear()
        load_adapter_calls.clear()
        model2, grounding_view2, planner_view2, reflector_view2 = load_shared_model(
            grounding_adapter_dir="checkpoint-4130", planner_adapter_dir="planner-checkpoint-99",
        )
        check("planner_adapter_dir만 지정해도 QwenVLModel 생성은 여전히 1번", len(construct_calls) == 1)
        check("load_adapter가 딱 1번 호출됨(planner만)", len(load_adapter_calls) == 1)
        check("load_adapter에 planner_adapter_dir 경로가 그대로 전달됨", load_adapter_calls[0]["path"] == "planner-checkpoint-99")
        check(
            "load_adapter에 adapter_name='planner'로 전달됨",
            load_adapter_calls[0]["adapter_name"] == _PLANNER_ADAPTER_NAME == "planner",
        )
        check("load_adapter는 is_trainable=False로 호출됨(추론 전용)", load_adapter_calls[0]["is_trainable"] is False)
        check("planner_view2.adapter_name은 'planner'", planner_view2._adapter_name == "planner")
        check("reflector_view2.adapter_name은 여전히 None(base, 독립적으로 결정됨)", reflector_view2._adapter_name is None)

        # --- 이번엔 반대로 reflector_adapter_dir만 지정, planner_adapter_dir는 미지정 ---
        construct_calls.clear()
        load_adapter_calls.clear()
        model3, grounding_view3, planner_view3, reflector_view3 = load_shared_model(
            grounding_adapter_dir="checkpoint-4130", reflector_adapter_dir="reflector-checkpoint-7",
        )
        check("load_adapter가 딱 1번 호출됨(reflector만)", len(load_adapter_calls) == 1)
        check(
            "load_adapter에 adapter_name='reflector'로 전달됨",
            load_adapter_calls[0]["adapter_name"] == _REFLECTOR_ADAPTER_NAME == "reflector",
        )
        check("planner_view3.adapter_name은 여전히 None(base, 독립적으로 결정됨)", planner_view3._adapter_name is None)
        check("reflector_view3.adapter_name은 'reflector'", reflector_view3._adapter_name == "reflector")

        # --- 둘 다 지정 ---
        construct_calls.clear()
        load_adapter_calls.clear()
        model4, grounding_view4, planner_view4, reflector_view4 = load_shared_model(
            grounding_adapter_dir="checkpoint-4130",
            planner_adapter_dir="planner-checkpoint-99",
            reflector_adapter_dir="reflector-checkpoint-7",
        )
        check("둘 다 지정해도 QwenVLModel 생성은 여전히 1번", len(construct_calls) == 1)
        check("load_adapter가 2번 호출됨(planner+reflector)", len(load_adapter_calls) == 2)
        check("planner_view4.adapter_name은 'planner'", planner_view4._adapter_name == "planner")
        check("reflector_view4.adapter_name은 'reflector'", reflector_view4._adapter_name == "reflector")
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
            grounding_view="fake-grounding-view", planner_view="fake-planner-view",
            reflector_view="fake-reflector-view", env=env,
            task={"web": "http://x", "ques": "do something"}, use_reflection=False, verbose=False,
        )
        check("terminate 즉시 -> status success", outcome["status"] == "success")
        check("terminate 즉시 -> reason model_terminate", outcome["reason"] == "model_terminate")
        check("terminate 즉시 -> answer 보존", outcome["answer"] == "42")
        check("terminate 즉시 -> steps == 1", outcome["steps"] == 1)
        check("terminate 즉시 -> execute_action 호출 안 됨", len(env.execute_calls) == 0)
        check("use_reflection=False -> plan_next_action 사용", plan_fn.called)
        check(
            "use_reflection=False -> plan_next_action이 planner_view로 호출됨",
            plan_fn.call_args.args[0] == "fake-planner-view",
        )
    finally:
        _uninstall_fake_modules()

    # --- use_reflection=True -> plan_with_reflection이 제안=planner_view/비평=reflector_view로 호출됨 ---
    reflect_fn = MagicMock(
        return_value={"reasoning": "done", "action": "terminate", "status": "success", "answer": None}
    )
    next_fn = MagicMock(return_value={"reasoning": "should not be called", "action": "terminate", "status": "failure"})
    _install_fake_planner_and_grounding(plan_next_action=next_fn, plan_with_reflection=reflect_fn)
    try:
        env = _FakeEnv()
        outcome = run_episode(
            grounding_view="fake-grounding-view", planner_view="fake-planner-view",
            reflector_view="fake-reflector-view", env=env,
            task=("http://x", "do something"), use_reflection=True,
            reflection_max_iterations=5, verbose=False,
        )
        check("use_reflection=True -> plan_with_reflection 호출됨", reflect_fn.called)
        check("use_reflection=True -> plan_next_action은 호출 안 됨", not next_fn.called)
        check(
            "reflection_max_iterations가 max_iterations로 전달됨",
            reflect_fn.call_args.kwargs.get("max_iterations") == 5,
        )
        check(
            "plan_with_reflection -> 제안(qwen_model 위치인자)은 planner_view",
            reflect_fn.call_args.args[0] == "fake-planner-view",
        )
        check(
            "plan_with_reflection -> reflection_model kwarg는 reflector_view",
            reflect_fn.call_args.kwargs.get("reflection_model") == "fake-reflector-view",
        )
    finally:
        _uninstall_fake_modules()

    # --- click 액션 -> grounding으로 좌표 변환(grounding_view로 호출) -> execute_action에
    #     절대 픽셀 좌표로 전달 -> 다음 스텝에서 terminate ---
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
            grounding_view="fake-grounding-view", planner_view="fake-planner-view",
            reflector_view="fake-reflector-view", env=env,
            task={"web": "http://x", "ques": "click the button"}, use_reflection=False, verbose=False,
        )
        check("click -> grounding 호출됨", ground_fn.called)
        check(
            "click -> ground()가 grounding_view로 호출됨(다른 view 아님)",
            ground_fn.call_args.args[0] == "fake-grounding-view",
        )
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
            grounding_view="fake-grounding-view", planner_view="fake-planner-view",
            reflector_view="fake-reflector-view", env=env,
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
            grounding_view="fake-grounding-view", planner_view="fake-planner-view",
            reflector_view="fake-reflector-view", env=env,
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
            grounding_view="fake-grounding-view", planner_view="fake-planner-view",
            reflector_view="fake-reflector-view", env=env,
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
            grounding_view="fake-grounding-view", planner_view="fake-planner-view",
            reflector_view="fake-reflector-view", env=env,
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
            _plan_to_env_action({"action": "type", "text": "hi"}, shot, "gv") == {"action": "type", "text": "hi"},
        )
        check(
            "key -> {action, text} 그대로",
            _plan_to_env_action({"action": "key", "text": "Enter"}, shot, "gv") == {"action": "key", "text": "Enter"},
        )
        check(
            "scroll -> text 기본값 down",
            _plan_to_env_action({"action": "scroll"}, shot, "gv") == {"action": "scroll", "text": "down"},
        )
        check(
            "wait -> 추가 필드 없음",
            _plan_to_env_action({"action": "wait"}, shot, "gv") == {"action": "wait"},
        )
        try:
            _plan_to_env_action({"action": "terminate", "status": "success"}, shot, "gv")
            check("terminate는 _plan_to_env_action에서 거부", False)
        except ValueError:
            check("terminate는 _plan_to_env_action에서 거부", True)
    finally:
        _uninstall_fake_modules()

    # =======================================================================
    # run_batch(): 실제 파일시스템에 임시 jsonl을 만들어서 append/resume/예외격리를 검증.
    # (run_episode 자체는 이미 위에서 충분히 검증했으므로, 여기서는 "여러 태스크를 순회
    # 하면서 jsonl에 쓰는" 오케스트레이션 로직만 집중적으로 본다.)
    # =======================================================================
    import json as _json
    import os as _os
    import tempfile as _tempfile

    tmp_dir = _tempfile.mkdtemp(prefix="agent_loop_selftest_")
    out_path = _os.path.join(tmp_dir, "results.jsonl")

    three_tasks = [
        {"id": "Site--0", "web_name": "Site", "web": "http://x/0", "ques": "task 0"},
        {"id": "Site--1", "web_name": "Site", "web": "http://x/1", "ques": "task 1"},
        {"id": "Site--2", "web_name": "Site", "web": "http://x/2", "ques": "task 2"},
    ]

    # 매 태스크마다 env.reset()이 새로 불리고 plan_next_action()이 곧장 terminate/success를
    # 내서 1스텝만에 끝나는 가장 단순한 성공 경로.
    always_terminate = MagicMock(
        return_value={"reasoning": "done", "action": "terminate", "status": "success", "answer": "ok"}
    )
    _install_fake_planner_and_grounding(plan_next_action=always_terminate)
    try:
        env = _FakeEnv()
        stats = run_batch(
            grounding_view="gv", planner_view="pv", reflector_view="rv", env=env,
            tasks=three_tasks, output_jsonl_path=out_path,
            use_reflection=False, verbose=False,
        )
        check("run_batch -> total == 3", stats["total"] == 3)
        check("run_batch -> completed == 3(처음 실행이라 스킵 없음)", stats["completed"] == 3)
        check("run_batch -> skipped == 0", stats["skipped"] == 0)
        check("run_batch -> succeeded == 3", stats["succeeded"] == 3)
        check("run_batch -> env.reset이 태스크마다 1번씩, 총 3번", len(env.reset_calls) == 3)

        with open(out_path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        check("run_batch -> jsonl에 3줄 기록됨", len(lines) == 3)
        rows = [_json.loads(line) for line in lines]
        check("run_batch -> 각 줄에 task_id 보존", [r["task_id"] for r in rows] == ["Site--0", "Site--1", "Site--2"])
        check("run_batch -> 각 줄에 ques 보존", rows[0]["ques"] == "task 0")
        check("run_batch -> 각 줄에 run_episode 결과(status) 병합됨", all(r["status"] == "success" for r in rows))
        check("run_batch -> 각 줄에 answer도 병합됨", rows[0]["answer"] == "ok")
    finally:
        _uninstall_fake_modules()

    # --- resume: 이미 3줄이 있는 out_path에 새 태스크 1개(Site--3) 추가해서 다시 돌리면,
    #     기존 3개는 건너뛰고 Site--3만 처리 + append(파일은 4줄이 됨) ---
    four_tasks = three_tasks + [{"id": "Site--3", "web_name": "Site", "web": "http://x/3", "ques": "task 3"}]
    _install_fake_planner_and_grounding(plan_next_action=always_terminate)
    try:
        env = _FakeEnv()
        stats2 = run_batch(
            grounding_view="gv", planner_view="pv", reflector_view="rv", env=env,
            tasks=four_tasks, output_jsonl_path=out_path,
            use_reflection=False, verbose=False,
        )
        check("resume -> skipped == 3(이미 끝난 것들)", stats2["skipped"] == 3)
        check("resume -> completed == 1(새 태스크 하나만)", stats2["completed"] == 1)
        check("resume -> env.reset은 새 태스크 것만 1번 호출", len(env.reset_calls) == 1)

        with open(out_path, "r", encoding="utf-8") as f:
            lines2 = [line for line in f if line.strip()]
        check("resume -> 기존 3줄 + 새 1줄 = 4줄(기존 줄 중복 안 됨)", len(lines2) == 4)
    finally:
        _uninstall_fake_modules()

    # --- resume=False면 이미 끝난 것도 다시 처리(+append, 기존 줄은 그대로 남아 중복 생김) ---
    _install_fake_planner_and_grounding(plan_next_action=always_terminate)
    try:
        env = _FakeEnv()
        stats3 = run_batch(
            grounding_view="gv", planner_view="pv", reflector_view="rv", env=env,
            tasks=three_tasks, output_jsonl_path=out_path,
            use_reflection=False, resume=False, verbose=False,
        )
        check("resume=False -> skipped == 0(전부 다시 처리)", stats3["skipped"] == 0)
        check("resume=False -> completed == 3", stats3["completed"] == 3)
    finally:
        _uninstall_fake_modules()

    # --- 태스크 하나에서 run_episode 호출 자체가 예외를 던지면(env.reset 실패 등),
    #     그 태스크만 status=error로 기록하고 나머지 태스크는 정상 처리 ---
    fresh_out_path = _os.path.join(tmp_dir, "results_with_error.jsonl")
    error_then_ok = MagicMock(
        side_effect=[
            {"reasoning": "done", "action": "terminate", "status": "success", "answer": "first ok"},
            RuntimeError("model crashed mid-task"),
            {"reasoning": "done", "action": "terminate", "status": "success", "answer": "third ok"},
        ]
    )
    _install_fake_planner_and_grounding(plan_next_action=error_then_ok)
    try:
        env = _FakeEnv()
        stats4 = run_batch(
            grounding_view="gv", planner_view="pv", reflector_view="rv", env=env,
            tasks=three_tasks, output_jsonl_path=fresh_out_path,
            use_reflection=False, verbose=False,
        )
        check("예외 격리 -> completed == 3(에러난 것도 처리는 됨, 스킵 아님)", stats4["completed"] == 3)
        check("예외 격리 -> succeeded == 2", stats4["succeeded"] == 2)
        check("예외 격리 -> errored == 1", stats4["errored"] == 1)
        check("예외 격리 -> 배치 전체가 안 죽고 3개 다 jsonl에 기록됨", True)  # 아래에서 실제 파일로 재확인

        with open(fresh_out_path, "r", encoding="utf-8") as f:
            err_rows = [_json.loads(line) for line in f if line.strip()]
        check("예외 격리 -> jsonl 3줄 모두 기록됨(에러난 태스크 포함)", len(err_rows) == 3)
        check("예외 격리 -> 에러난 태스크 status=='error'", err_rows[1]["status"] == "error")
        check("예외 격리 -> 에러 메시지 보존", "model crashed mid-task" in err_rows[1]["error"])
        check("예외 격리 -> 에러난 태스크도 task_id는 보존됨", err_rows[1]["task_id"] == "Site--1")
        check("예외 격리 -> 앞뒤 태스크는 정상 success", err_rows[0]["status"] == "success" and err_rows[2]["status"] == "success")
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
        description="모델 인스턴스 하나로 세 역할(grounding/planner/reflector)이 각자 독립적으로 "
        "선택된 어댑터(또는 base)로 도는지 확인하거나(단발 테스트), 실제 WebVoyagerEnv에서 "
        "태스크 하나를 처음부터 끝까지(run_episode) 돌려보는 수동 확인용 CLI."
    )
    ap.add_argument("--selftest", action="store_true", help="실제 모델 없이 로직만 mock으로 검증")
    ap.add_argument("--adapter_dir", required=False, help="grounding LoRA 체크포인트 경로 (예: checkpoint-4130, 필수)")
    ap.add_argument(
        "--planner_adapter_dir", default=None,
        help="planner 전용 LoRA 체크포인트 경로. 생략하면 planner도 base(어댑터 없음)로 동작한다.",
    )
    ap.add_argument(
        "--reflector_adapter_dir", default=None,
        help="reflector 전용 LoRA 체크포인트 경로. 생략하면 reflector도 base(어댑터 없음)로 "
        "동작한다. planner_adapter_dir와 독립적으로 지정 가능.",
    )
    ap.add_argument(
        "--reflect", action="store_true",
        help="planning에 plan_with_reflection()(실행 전 비평 루프, 제안=planner_view/"
        "비평=reflector_view)을 쓴다. 안 주면 plan_next_action()(planner_view)만 씀. "
        "에피소드/단발 테스트 내내 이 값이 고정됨(스텝마다 바뀌지 않음).",
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

    batch = ap.add_argument_group(
        "배치 모드 (--output_jsonl까지 같이 주면, --tasks_jsonl의 태스크를 첫 번째 하나만이 "
        "아니라 전부(또는 --limit개까지) 순서대로 돌려서 결과를 jsonl로 저장)"
    )
    batch.add_argument(
        "--output_jsonl",
        help="지정하면 배치 모드로 전환 - --tasks_jsonl의 태스크를 전부(또는 --limit개까지) "
        "run_episode()로 돌리고, 결과를 이 경로에 한 줄씩(JSON Lines) append한다. "
        "--tasks_jsonl과 함께 써야 함(--url/--instruction 단일 태스크 모드에는 안 됨).",
    )
    batch.add_argument(
        "--limit", type=int, default=None,
        help="--output_jsonl(배치 모드)일 때 처리할 태스크 수 상한. 미지정시 --tasks_jsonl "
        "(및 --web_name 필터링) 전체를 다 돈다. 전체 배치 전에 몇 개만 먼저 돌려서 "
        "확인해보고 싶을 때 씀(예: --limit 3).",
    )
    batch.add_argument(
        "--no_resume", dest="resume", action="store_false", default=True,
        help="기본은 resume=True - --output_jsonl 파일이 이미 있으면 거기 기록된 task_id는 "
        "건너뛰고 이어서 돌린다. 처음부터 전부 다시 돌리고 싶으면 이 플래그를 주거나 "
        "--output_jsonl 파일을 미리 지울 것.",
    )

    args = ap.parse_args()

    if args.selftest:
        _run_mock_selftest()
        return

    if not args.adapter_dir:
        raise SystemExit("--adapter_dir(grounding LoRA 체크포인트) 필요 (또는 --selftest)")

    if args.run_episode:
        if args.output_jsonl and not args.tasks_jsonl:
            raise SystemExit("--output_jsonl(배치 모드)에는 --tasks_jsonl 필요")
        if not (args.tasks_jsonl or (args.url and args.instruction)):
            raise SystemExit("--run_episode에는 --tasks_jsonl 또는 (--url + --instruction) 필요")

        from env_webvoyager import WebVoyagerEnv, load_webvoyager_tasks

        print("=== 모델 로딩 시작 (아래 '[qwen.py] Loading ...' 메시지가 한 번만 찍혀야 함) ===")
        model, grounding_view, planner_view, reflector_view = load_shared_model(
            args.adapter_dir,
            planner_adapter_dir=args.planner_adapter_dir,
            reflector_adapter_dir=args.reflector_adapter_dir,
        )
        print("=== 로딩 끝 ===\n")

        env = WebVoyagerEnv(window_size=(args.width, args.height), headless=args.headless)
        try:
            if args.output_jsonl:
                # --- 배치 모드: --tasks_jsonl의 태스크를 전부(또는 --limit개까지) 돌려서
                #     결과를 --output_jsonl에 한 줄씩 저장 ---
                tasks = load_webvoyager_tasks(args.tasks_jsonl, web_name=args.web_name)
                if args.limit is not None:
                    tasks = tasks[: args.limit]
                if not tasks:
                    raise SystemExit("조건에 맞는 태스크가 없음")
                print(f"=== 배치 시작: 태스크 {len(tasks)}개, 결과는 {args.output_jsonl}에 저장 ===")
                stats = run_batch(
                    grounding_view, planner_view, reflector_view, env, tasks,
                    output_jsonl_path=args.output_jsonl,
                    use_reflection=args.reflect,
                    max_steps=args.max_steps,
                    reflection_max_iterations=args.max_iterations,
                    resume=args.resume,
                )
                print(json.dumps(stats, ensure_ascii=False, indent=2))
            else:
                # --- 단일 태스크 모드 (기존 동작 그대로) ---
                if args.tasks_jsonl:
                    tasks = load_webvoyager_tasks(args.tasks_jsonl, web_name=args.web_name)
                    if not tasks:
                        raise SystemExit("조건에 맞는 태스크가 없음")
                    task = tasks[0]
                else:
                    task = (args.url, args.instruction)

                outcome = run_episode(
                    grounding_view, planner_view, reflector_view, env, task,
                    use_reflection=args.reflect,
                    max_steps=args.max_steps,
                    reflection_max_iterations=args.max_iterations,
                )
                print(json.dumps(outcome, ensure_ascii=False, indent=2))
        finally:
            env.close()

        return

    if not args.image or not args.task:
        raise SystemExit("--image와 --task 필요 (또는 --run_episode / --selftest)")

    from PIL import Image

    from planner import plan_next_action, plan_with_reflection

    print("=== 모델 로딩 시작 (아래 '[qwen.py] Loading ...' 메시지가 한 번만 찍혀야 함) ===")
    model, grounding_view, planner_view, reflector_view = load_shared_model(
        args.adapter_dir,
        planner_adapter_dir=args.planner_adapter_dir,
        reflector_adapter_dir=args.reflector_adapter_dir,
    )
    print("=== 로딩 끝 ===\n")

    screenshot = Image.open(args.image)

    print("--- planning (제안: planner 어댑터 있으면 그걸로 없으면 base / 비평: reflector 어댑터 있으면 그걸로 없으면 base) ---")
    if args.reflect:
        plan = plan_with_reflection(
            planner_view, args.task, screenshot,
            max_iterations=args.max_iterations, reflection_model=reflector_view,
        )
    else:
        plan = plan_next_action(planner_view, args.task, screenshot)
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if args.test_grounding_instruction:
        from gui_grounding import ground

        print("\n--- grounding (grounding 어댑터로 고정) ---")
        result = ground(grounding_view, args.test_grounding_instruction, screenshot)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()