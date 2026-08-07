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

[사용법]
    model, planning_view = load_shared_model(adapter_dir="checkpoint-4130")
    plan = plan_with_reflection(planning_view, task, screenshot, history)   # 어댑터 꺼짐
    result = ground(model, plan["target_description"], screenshot)          # 어댑터 켜짐(기본)

model 하나만 메모리에 올라간다 - planning_view는 같은 model을 감싸서 .generate() 호출을
disable_adapter() 컨텍스트로 넘겨주는 얇은 프록시일 뿐, 별도 모델이 아니다.

[아직 안 된 것]
실제 action 실행(env_webvoyager.WebVoyagerEnv.execute_action())까지 엮는 step loop는
아직 없다 - 지금은 "모델 하나로 두 역할이 실제로 도는지"부터 먼저 검증하는 단계.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def load_shared_model(adapter_dir: str, **qwen_model_kwargs):
    """
    QwenVLModel을 grounding LoRA를 얹은 상태로 "딱 한 번만" 로드하고,
    (grounding용 뷰, planning용 뷰) 튜플을 반환한다.

    grounding 호출은 반환된 model을 그대로 쓰면 됨(어댑터가 기본으로 켜진 상태 =
    gui_grounding.ground()/region_focus.region_focus() 등이 기대하는 상태 그대로).
    planning/reflection 호출은 반환된 planning_view를 planner.py의 함수들에 넘기면 됨.

    qwen_model_kwargs는 QwenVLModel(...)에 그대로 전달됨(min_pixels/max_pixels 등 조정용).

    Returns: (model: QwenVLModel, planning_view: _BaseModelView)
    """
    from qwen import QwenVLModel  # 실제 로드 시점에만 임포트 (mock 테스트는 이 경로를 안 탐)

    model = QwenVLModel(adapter_dir=adapter_dir, **qwen_model_kwargs)
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
    model, planning_view = load_shared_model(args.adapter_dir)
    print("=== 로딩 끝 ===\n")

    screenshot = Image.open(args.image)

    print("--- planning (어댑터 꺼짐, base 모델처럼 동작) ---")
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