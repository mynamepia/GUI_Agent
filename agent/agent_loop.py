"""
agent_loop.py

전체 에이전트 루프의 첫 조각. 지금 단계에서는 "모델 인스턴스 하나 + LoRA 어댑터
on/off 스위칭"만 구현한다 - planner용 base 모델과 grounding용 LoRA 모델을 각각
별도 QwenVLModel 인스턴스로 띄웠다가 실제로 RAM(16GB, CUDA 없는 CPU 환경)이 터진 적이
있어서(터미널/프로세스 두 개가 각각 모델을 물고 있다가 두 번째 로딩에서 터짐), 이 문제를
프롬프트 병합이 아니라 "모델은 grounding LoRA를 얹은 채로 딱 한 번만 로드하고,
planning/reflection이 필요할 때만 peft의 disable_adapter() 컨텍스트로 일시적으로
base 모델처럼 동작시킨다"는 방식으로 구조적으로 없앤다.

[왜 프롬프트 병합이 아니라 이 방식인가 - 요약]
grounding LoRA는 coord_utils.PROMPT_TEMPLATE 하나, "(x,y)" 한 줄 출력이라는 아주 좁은
포맷으로만 SFT됐다(train.py). planning(JSON, reasoning, target_description 등)은 이
LoRA가 학습에서 한 번도 못 본 포맷이라, 어댑터를 켠 채로 planning을 시키면 - 예전에
tool-call 포맷을 줬을 때 grounding 자체가 깨졌던 것과(gui_grounding.py/region_focus.py의
2026-08 수정 주석들 참고) 똑같은 방식으로 - planning이 오히려 더 망가질 위험이 있었다.
그래서 planning/reflection은 어댑터를 꺼서 순수 base 모델 동작으로 돌리고, grounding만
어댑터를 켠 상태로 돌린다 - 전부 같은 프로세스, 같은 모델 인스턴스 안에서.

[vision encoder는 항상 동일하다 - "grounding 모델이 이미지를 더 잘 본다"는 주장에 대한 반박 근거]
train.py의 LoraConfig.target_modules는 ["q_proj","k_proj","v_proj","o_proj"] - LLM 디코더
쪽 attention만 건드리고 vision encoder는 전혀 안 건드린다. 즉 어댑터를 껐다 켰다 해도
"이미지를 보는" 부분의 가중치는 항상 동일하다 - 바뀌는 건 그 시각 정보를 어떤 출력
포맷으로 뽑아내느냐뿐이다. planning을 어댑터 없이 시켜도 화면을 보는 능력 자체는
grounding 때와 다르지 않다.

[사용법 - grounding LoRA만 있을 때(planner LoRA 없이, 기존 방식)]
    model, planning_view = load_shared_model(adapter_dir="checkpoint-4130")
    plan = plan_with_reflection(planning_view, task, screenshot, history)   # 어댑터 꺼짐(base)
    result = ground(model, plan["target_description"], screenshot)          # 어댑터 켜짐(기본)

model 하나만 메모리에 올라간다 - planning_view는 같은 model을 감싸서 .generate() 호출을
disable_adapter() 컨텍스트로 넘겨주는 얇은 프록시일 뿐, 별도 모델이 아니다.

[2026-08-09 추가 - planner LoRA(멀티 어댑터 스왑)]
planner LoRA(`planner/train_planner.py`로 학습, JSON 액션 스키마 SFT)가 나온 뒤로는
planning을 "어댑터 없는 base"가 아니라 "planner 어댑터를 켠 상태"로 돌리는 게 진짜 검증이다.
peft의 PeftModel은 여러 named adapter를 동시에 로드해두고 `set_adapter(name)`으로 활성
어댑터만 바꿔 끼우는 것을 지원한다(LoRA 웨이트를 다시 로드하지 않으므로 스왑 비용이 거의
없음) - `disable_adapter()`(모든 어댑터를 끄고 base로) 하나만 있던 기존 구조에 이 방식을
추가했다. 사용법:

    model, planning_view = load_shared_model(
        adapter_dir="checkpoints/qwen2.5vl-3b-gui-lora-stage2/checkpoint-4130",   # grounding, 기본 활성
        planner_adapter_dir="checkpoints/qwen2.5vl-3b-planner-lora",              # 추가 로드
    )
    plan = plan_with_reflection(planning_view, task, screenshot, history)  # "planner" 어댑터로 전환해서 생성
    result = ground(model, plan["target_description"], screenshot)         # "default"(grounding)로 복원된 상태

`planner_adapter_dir`를 안 주면 기존 동작(디스에이블→base) 그대로 유지된다 - 하위 호환.

[아직 안 된 것]
실제 action 실행(env_webvoyager.WebVoyagerEnv.execute_action())까지 엮는 step loop는
아직 없다 - 지금은 "모델 하나로 두 역할이 실제로 도는지"부터 먼저 검증하는 단계.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# (2026-08-09 추가) vlm_agent(qwen.py가 있는 폴더)를 sys.path에 넣는다 - planner.py와 동일한
# 패턴(agent/의 다른 스크립트들이 이미 이 부트스트랩을 쓰고 있는데 이 파일만 빠져 있었음).
# 이게 없으면 `from qwen import ...`(load_shared_model 안)가 실행 cwd/방식에 따라
# ModuleNotFoundError로 조용히 깨진다(실측: `python agent\eval_webvoyager.py`를
# C:\gpu-work에서 실행했을 때 agent_loop.py가 import되면서 이 에러가 남).
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
    # 실행에는 필요 없고 타입 힌트용 - torch/transformers/peft를 mock selftest 경로에서까지
    # 강제로 임포트하지 않기 위해 TYPE_CHECKING 가드로 묶어둠 (planner.py와 동일한 패턴).
    from qwen import QwenVLModel


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


class _AdapterSwitchView:
    """
    (2026-08-09 추가) grounding + planner 두 개의 named LoRA 어댑터를 동시에 얹은
    QwenVLModel을 감싸서, .generate() 호출을 지정된 어댑터("planner")로 전환한 채로 실행하고
    끝나면 원래 활성 어댑터("default"=grounding)로 복원하는 프록시.

    peft의 disable_adapter()는 "모든 어댑터를 끄고 base로" 되돌리는 컨텍스트만 제공하고
    "어댑터 A로 켜져 있던 걸 잠깐 어댑터 B로 바꿔 쓰는" 기능은 없어서(_BaseModelView는 이
    disable_adapter()만 씀), model.set_adapter(name)을 직접 전/후로 호출해 전환한다.
    generate() 도중 예외가 나도 어댑터가 planner 상태로 눌어붙지 않도록 finally로 복원 -
    안 그러면 다음 grounding 호출이 조용히 planner 어댑터로 좌표를 뽑는 사고가 날 수 있음.
    """

    def __init__(self, qwen_model: "QwenVLModel", adapter_name: str, restore_to: str):
        if not hasattr(qwen_model.model, "set_adapter"):
            raise TypeError(
                "_AdapterSwitchView는 peft.PeftModel(멀티 어댑터 로드된 상태)에만 쓸 수 있음 - "
                f"qwen_model.model의 타입이 {type(qwen_model.model).__name__}이라 set_adapter()가 없음."
            )
        self._qwen_model = qwen_model
        self._adapter_name = adapter_name
        self._restore_to = restore_to

    def generate(
        self, messages: list, max_new_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0
    ) -> str:
        model = self._qwen_model.model
        model.set_adapter(self._adapter_name)
        try:
            return self._qwen_model.generate(
                messages, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            )
        finally:
            model.set_adapter(self._restore_to)


def load_shared_model(adapter_dir: str, planner_adapter_dir: str | None = None, **qwen_model_kwargs):
    """
    QwenVLModel을 grounding LoRA를 얹은 상태로 "딱 한 번만" 로드하고,
    (grounding용 뷰, planning용 뷰) 튜플을 반환한다.

    grounding 호출은 반환된 model을 그대로 쓰면 됨(어댑터가 기본으로 켜진 상태 =
    gui_grounding.ground()/region_focus.region_focus() 등이 기대하는 상태 그대로).
    planning/reflection 호출은 반환된 planning_view를 planner.py의 함수들에 넘기면 됨.

    qwen_model_kwargs는 QwenVLModel(...)에 그대로 전달됨(min_pixels/max_pixels 등 조정용).

    planner_adapter_dir: (2026-08-09 추가) 지정하면 grounding 어댑터("default")에 더해
    이 경로의 planner LoRA를 "planner"라는 이름으로 추가 로드하고, planning_view가
    _AdapterSwitchView(어댑터 스왑)가 된다 - planning이 진짜 planner LoRA로 돈다.
    지정 안 하면(기본) 기존 방식대로 _BaseModelView(disable_adapter → base)가 반환된다.

    Returns: (model: QwenVLModel, planning_view: _BaseModelView | _AdapterSwitchView)
    """
    from qwen import QwenVLModel  # 실제 로드 시점에만 임포트 (mock 테스트는 이 경로를 안 탐)

    model = QwenVLModel(adapter_dir=adapter_dir, **qwen_model_kwargs)

    if planner_adapter_dir:
        print(f"[agent_loop.py] Loading planner LoRA adapter from {planner_adapter_dir} ...")
        # peft: PeftModel.from_pretrained(model, adapter_dir)로 로드한 첫 어댑터는 "default"라는
        # 이름을 자동으로 받는다(qwen.py의 load_model_and_processor 참고). load_adapter()로 두 번째
        # named adapter를 추가하면 로드 직후 그 어댑터가 활성화되므로, grounding 호출부(model을
        # 그대로 쓰는 쪽)가 기대하는 "기본은 grounding 켜짐" 상태를 유지하려고 로드 직후 명시적으로
        # "default"로 되돌려놓는다.
        model.model.load_adapter(planner_adapter_dir, adapter_name="planner")
        model.model.set_adapter("default")
        planning_view = _AdapterSwitchView(model, adapter_name="planner", restore_to="default")
    else:
        planning_view = _BaseModelView(model)

    return model, planning_view


# ---------------------------------------------------------------------------
# mock 기반 단위 테스트 (실제 모델/torch/peft 없이 로직만 검증)
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

    # --- _AdapterSwitchView: generate()가 지정 어댑터로 전환됐다가 원래 어댑터로 복원되는지 ---
    fake_qwen_model2 = MagicMock()
    fake_qwen_model2.generate.return_value = "planner response text"
    set_adapter_calls = []
    fake_qwen_model2.model.set_adapter.side_effect = lambda name: set_adapter_calls.append(name)

    switch_view = _AdapterSwitchView(fake_qwen_model2, adapter_name="planner", restore_to="default")
    result2 = switch_view.generate([{"role": "user", "content": []}], max_new_tokens=50)

    check("_AdapterSwitchView: generate 결과가 그대로 반환됨", result2 == "planner response text")
    check(
        "_AdapterSwitchView: planner로 전환 후 generate, 끝나면 default로 복원(순서까지)",
        set_adapter_calls == ["planner", "default"],
    )

    # --- _AdapterSwitchView: generate 도중 예외가 나도 원래 어댑터로 복원되는지(finally) ---
    fake_qwen_model3 = MagicMock()
    fake_qwen_model3.generate.side_effect = RuntimeError("boom")
    set_adapter_calls3 = []
    fake_qwen_model3.model.set_adapter.side_effect = lambda name: set_adapter_calls3.append(name)

    switch_view3 = _AdapterSwitchView(fake_qwen_model3, adapter_name="planner", restore_to="default")
    try:
        switch_view3.generate([{"role": "user", "content": []}])
        check("_AdapterSwitchView: 예외 발생해도 복원됨(finally)", False)
    except RuntimeError:
        check("_AdapterSwitchView: 예외 발생해도 복원됨(finally)", set_adapter_calls3 == ["planner", "default"])

    # --- _AdapterSwitchView: set_adapter 없는(멀티 어댑터 아닌) 모델을 넘기면 생성 시점에 바로 에러 ---
    try:
        _AdapterSwitchView(_FakeNoAdapterQwenModel(), adapter_name="planner", restore_to="default")
        check("set_adapter 없는 모델 -> 생성 시점에 TypeError", False)
    except TypeError as e:
        check("set_adapter 없는 모델 -> 생성 시점에 TypeError", "set_adapter" in str(e))

    # --- load_shared_model: QwenVLModel이 "딱 한 번만" 생성되는지(기존 방식, planner_adapter_dir 없음) ---
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
        check("planner_adapter_dir 없으면 planning_view가 _BaseModelView", isinstance(planning_view, _BaseModelView))
    finally:
        del sys.modules["qwen"]

    # --- load_shared_model: planner_adapter_dir을 주면 멀티 어댑터로 로드되고 _AdapterSwitchView가 되는지 ---
    construct_calls2 = []
    load_adapter_calls = []

    class _FakeQwenVLModel2:
        def __init__(self, **kwargs):
            construct_calls2.append(kwargs)
            self.model = MagicMock()
            self.model.load_adapter.side_effect = lambda path, adapter_name: load_adapter_calls.append(
                (path, adapter_name)
            )

        def generate(self, *a, **kw):
            return "ok"

    fake_qwen_module2 = types.ModuleType("qwen")
    fake_qwen_module2.QwenVLModel = _FakeQwenVLModel2
    sys.modules["qwen"] = fake_qwen_module2
    try:
        model2, planning_view2 = load_shared_model(
            adapter_dir="fake-grounding-checkpoint", planner_adapter_dir="fake-planner-checkpoint"
        )
        check("QwenVLModel 생성 호출이 여전히 딱 1번(멀티 어댑터도 모델은 하나)", len(construct_calls2) == 1)
        check(
            "planner LoRA가 load_adapter(path, adapter_name='planner')로 추가 로드됨",
            load_adapter_calls == [("fake-planner-checkpoint", "planner")],
        )
        check("로드 직후 활성 어댑터가 grounding(default)로 복원됨", model2.model.set_adapter.call_args_list[-1].args == ("default",))
        check("planner_adapter_dir 있으면 planning_view가 _AdapterSwitchView", isinstance(planning_view2, _AdapterSwitchView))
        check("_AdapterSwitchView가 같은 model 인스턴스를 공유함", planning_view2._qwen_model is model2)
    finally:
        del sys.modules["qwen"]

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
        "둘 다 도는지 수동 확인용 CLI."
    )
    ap.add_argument("--selftest", action="store_true", help="실제 모델 없이 로직만 mock으로 검증")
    ap.add_argument("--adapter_dir", help="grounding LoRA 체크포인트 경로 (예: checkpoint-4130)")
    ap.add_argument(
        "--planner_adapter_dir",
        default=None,
        help="(2026-08-09 추가) 지정하면 planner LoRA(checkpoints/qwen2.5vl-3b-planner-lora)를 "
        "grounding 어댑터와 함께 로드해서, planning을 실제로 이 어댑터로 돌린다(기존 "
        "disable_adapter→base 대신 set_adapter로 스왑). 안 주면 기존처럼 base로 planning.",
    )
    ap.add_argument("--image", help="테스트용 스크린샷 경로")
    ap.add_argument("--task", help="planning 테스트용 태스크 지시문")
    ap.add_argument("--reflect", action="store_true", help="plan_next_action 대신 plan_with_reflection 사용")
    ap.add_argument(
        "--test_grounding_instruction",
        help="지정하면, 같은 모델 인스턴스로 이 문구에 대해 grounding까지 이어서 테스트",
    )
    args = ap.parse_args()

    if args.selftest:
        _run_mock_selftest()
        return

    if not args.adapter_dir or not args.image or not args.task:
        raise SystemExit("--adapter_dir, --image, --task 필요 (또는 --selftest)")

    from PIL import Image

    from planner import plan_next_action, plan_with_reflection

    print("=== 모델 로딩 시작 (아래 '[qwen.py] Loading ...' 메시지가 한 번만 찍혀야 함) ===")
    model, planning_view = load_shared_model(args.adapter_dir, planner_adapter_dir=args.planner_adapter_dir)
    print("=== 로딩 끝 ===\n")

    screenshot = Image.open(args.image)

    planning_desc = "planner LoRA 어댑터로 동작" if args.planner_adapter_dir else "어댑터 꺼짐, base 모델처럼 동작"
    print(f"--- planning ({planning_desc}) ---")
    if args.reflect:
        plan = plan_with_reflection(planning_view, args.task, screenshot)
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