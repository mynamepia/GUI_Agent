"""
gui_actor_grounding.py

microsoft/GUI-Actor-3B-Qwen2.5-VL 을 gui_grounding.ground()와 완전히 동일한
인터페이스({"result", "point", "raw_response"})로 감싸는 어댑터.

agent_loop.py / region_focus.py 등 기존 호출부는 전부 `ground(model, instruction, image)`
형태로만 grounding을 호출하므로, 이 파일의 GUIActorModel + ground()로 갈아끼우면
기존 파이프라인 코드(agent_loop.py의 두 곳, region_focus.py)를
    from gui_grounding import ground          ->  from gui_actor_grounding import ground
    model = QwenVLModel(...)                  ->  model = GUIActorModel(...)
두 줄만 바꿔서 grounding 백엔드를 통째로 교체해볼 수 있다.

[중요 - 이 모듈은 gui_grounding.ground()와 내부 동작이 완전히 다르다]
gui_grounding.ground()는 우리 LoRA가 학습받은 포맷(coord_utils.PROMPT_TEMPLATE,
"(x,y)" 0~1000 정규화 텍스트)으로 묻고 텍스트를 파싱해서 좌표를 만든다.
GUI-Actor는 좌표를 텍스트로 생성하지 않고, attention 기반 pointer head가 이미지
패치 토큰 중 하나를 직접 가리키는 "coordinate-free" 방식이다(이미 0~1 정규화된
[x, y] float를 바로 반환하므로 여기서 별도 좌표 변환도 필요 없다).
겉보기 인터페이스만 맞춘 어댑터라는 점을 유의할 것.

필요 설치(로컬 PC의 이 프로젝트 venv 안에서, 이 리포와는 별도 위치에 clone):
    git clone https://github.com/microsoft/GUI-Actor.git
    cd GUI-Actor
    pip install -e .
    (torch/transformers/qwen-vl-utils/pillow는 qwen.py 쪽에서 이미 설치돼 있을 것)
    (flash_attention_2를 쓰려면 추가로: pip install flash-attn --no-build-isolation
     - 안 깔려 있으면 그냥 sdpa로 돌리면 됨, 추가 설치 없이 torch 내장)

VRAM 참고: 이 모델은 4B params(bf16 기준 ~8GB 가중치) + 추론 전용(그래디언트/옵티마이저
상태 없음)이라, 16GB에서 여유 있게 들어간다. 학습이 아니라 순수 추론 테스트이므로
지금까지 논의한 QLoRA/8bit optimizer 등의 최적화는 여기선 필요 없다.
"""

import sys as _sys
import time
import types as _types

from PIL import Image

# (2026-08 추가 - transformers 버전 충돌 회피, 모듈 최상단으로 이동) gui_actor.modeling_qwen25vl은
# 추론에 전혀 필요 없는 gui_actor.trainer.rank0_print(단순 "rank 0에서만 print" 유틸)를 최상단에서
# import하는데, gui_actor.trainer.py 자체가 `from transformers import Trainer`를 끌어온다. 이
# Trainer/peft 체인이 이 프로젝트 환경의 transformers 버전과 충돌해 깨지는 게 실측으로 확인됐다 -
# Trainer 전체를 끌어오는 무거운 import 자체가 우리한테는 불필요하므로, 진짜 gui_actor.trainer 대신
# rank0_print만 들어있는 가짜 모듈을 sys.modules에 미리 등록해서 이 import 체인을 통째로 우회한다.
# (이전엔 GUIActorModel.__init__ 안에 있었는데, 그러면 "from gui_actor.modeling_qwen25vl import ..."를
# GUIActorModel을 생성하기 전에 먼저 실행하는 코드에서는 스텁이 등록되기 전에 진짜 gui_actor.trainer가
# 로드돼버리는 순서 문제가 있어서 모듈 최상단 - 이 파일을 import하는 즉시 실행 - 으로 옮겼다.)
if "gui_actor.trainer" not in _sys.modules:
    _stub = _types.ModuleType("gui_actor.trainer")

    def _rank0_print(*args, **kwargs):
        import os as _os

        if int(_os.environ.get("LOCAL_RANK", "0")) == 0:
            print(*args, **kwargs)

    _stub.rank0_print = _rank0_print
    _sys.modules["gui_actor.trainer"] = _stub


# (2026-08 추가 - topk 후보 다양성 조정, 사용자 승인됨) gui_actor.inference.inference()는 내부에서
# get_prediction_region_point(attn_scores, ..., activation_threshold=0.3)을 인자 노출 없이
# 호출한다 - 이 함수는 최대 attention 대비 activation_threshold 이상인 패치들만 모아 연결된
# "영역"으로 묶고, 영역별 평균 activation으로 정렬한 뒤 그 정렬된 리스트를 region_points로
# 돌려준다(1등이 best_point). topk_points는 이 region_points를 앞에서 topk개 자른 것뿐이라,
# 애초에 0.3 threshold를 넘는 영역이 1개만 나오면 topk를 아무리 올려도 후보가 1개로 고정된다
# (실측: RegionFocus가 오답 판정 후 재탐색하려 해도 2순위 후보가 아예 없어서 재탐색 자체가
# 무산되는 케이스가 반복 관찰됨). threshold를 0.3 -> 0.2로 살짝 낮춰서 애매하게 낮은 확신의
# 영역도 후보 풀에 들어오게 한다 - 정렬 로직(평균 activation 내림차순)과 best_point(1등,
# topk_points[0]) 선택 기준 자체는 전혀 안 바꾼다. 즉 "1순위 판단"은 원래 GUI-Actor 그대로고,
# "2순위 이후 후보가 아예 없어서 RegionFocus 안전장치가 무력화되는" 상황만 완화하는 목적.
# 이 값을 바꾸면 모델의 실제 후보 산출 결과가 달라지므로(순수 인프라 수정이 아님), 실행 시점에
# 항상 로그로 남긴다 - 나중에 "이 결과가 어떤 threshold로 나온 건지" 헷갈리지 않게.
GUI_ACTOR_ACTIVATION_THRESHOLD = 0.2  # GUI-Actor 원본 기본값은 0.3


def _patch_gui_actor_activation_threshold():
    import gui_actor.inference as _inf_mod

    if getattr(_inf_mod, "_activation_threshold_patched", False):
        return
    _original_get_prediction_region_point = _inf_mod.get_prediction_region_point

    def _patched_get_prediction_region_point(
        attn_scores, n_width, n_height, top_n=30,
        activation_threshold=GUI_ACTOR_ACTIVATION_THRESHOLD,
        return_all_regions=True, rect_center=False,
    ):
        # best_point/정렬 로직은 원본 함수 그대로 위임 - 여기선 기본 threshold 값만 바꾼다.
        return _original_get_prediction_region_point(
            attn_scores, n_width, n_height, top_n=top_n,
            activation_threshold=activation_threshold,
            return_all_regions=return_all_regions, rect_center=rect_center,
        )

    _inf_mod.get_prediction_region_point = _patched_get_prediction_region_point
    _inf_mod._activation_threshold_patched = True
    print(
        f"[gui_actor_grounding] activation_threshold 패치 적용됨: "
        f"GUI-Actor 기본 0.3 -> {GUI_ACTOR_ACTIVATION_THRESHOLD} "
        f"(1순위 선택 로직은 그대로, 2순위 이후 후보 풀만 넓힘)"
    )


class GUIActorModel:
    """
    qwen.py의 QwenVLModel과 같은 역할("모델 하나 들고 grounding 호출한다"는 인터페이스)을
    하지만, 내부는 GUI-Actor의 pointer 기반 모델이다.
    """

    def __init__(
        self,
        model_id: str = "microsoft/GUI-Actor-3B-Qwen2.5-VL",
        device: str = "cuda",
        dtype=None,
        attn_implementation: str = "sdpa",
        # flash_attention_2가 설치돼 있으면 "flash_attention_2"로 바꾸면 VRAM을 더 아낄 수
        # 있음. 안 깔았으면 sdpa가 추가 설치 없이 쓸 수 있는 안전한 기본값.
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        # (v3 수정 - 버그) 처음 버전은 min_pixels/max_pixels를 아예 안 받고 AutoProcessor를
        # 기본값으로만 로드했다 - 그러면 프로젝트의 --min_pixels/--max_pixels가 GUI-Actor
        # 경로에서는 조용히 무시되고 GUI-Actor 자체 프로세서 기본 해상도로만 도는 상태가
        # 된다(qwen.py.load_model_and_processor는 processor_kwargs로 이걸 명시적으로 넘기는데
        # 여기만 빠져 있었음). ground()가 아니라 여기(모델/프로세서 로드 시점)에서 해상도를
        # 고정하는 이유는 GUI-Actor의 추론 경로(gui_actor.inference())가 매 호출마다 별도
        # smart_resize 인자를 받지 않고, processor에 이미 설정된 min/max_pixels를 그대로
        # 쓰기 때문이다 - qwen.py 쪽(호출마다 smart_resize)과 제어 지점이 다르다는 점에 주의.
    ):
        import torch
        from transformers import AutoProcessor

        # gui_actor 패키지는 pip 패키지가 아니라, 위 README의 GUI-Actor 리포를
        # `pip install -e .`로 설치해야 import된다. 안 깔려 있으면 여기서 바로
        # ImportError가 나니, 설치 안내 메시지를 붙여서 원인을 바로 알 수 있게 한다.
        try:
            from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
        except ImportError as e:
            raise ImportError(
                "gui_actor 패키지를 찾을 수 없습니다. "
                "https://github.com/microsoft/GUI-Actor 를 clone한 뒤 "
                "그 디렉토리에서 `pip install -e .`로 설치하세요."
            ) from e

        # gui_actor 패키지 import가 위에서 확인됐으니, 이 시점에 activation_threshold 패치를
        # 적용한다(모듈 최상단에서 바로 하면 gui_actor 패키지가 안 깔린 환경에서 import 자체가
        # 실패할 수 있어 여기로 미룸 - 위 try/except가 이미 그 케이스를 안내 메시지로 처리해줌).
        _patch_gui_actor_activation_threshold()

        self.device = device
        actual_dtype = dtype or torch.bfloat16

        print(
            f"[gui_actor_grounding] Loading {model_id} on {device} (attn={attn_implementation}, "
            f"min_pixels={min_pixels}, max_pixels={max_pixels}) ..."
        )
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_id,
            torch_dtype=actual_dtype,
            device_map=device,
            attn_implementation=attn_implementation,
        ).eval()
        print("[gui_actor_grounding] Model loaded.")

    def predict_point(self, instruction: str, image, topk: int = 3) -> dict:
        """
        GUI-Actor 자체 inference()를 감싼다. 반환값의 "topk_points"[0]가 가장 확신하는
        지점 (원본 이미지 기준 0~1 정규화 [x, y]).
        """
        from gui_actor.inference import inference

        # (2026-08 추가 - 실제 추론 해상도 실측 로그) processor의 min_pixels/max_pixels
        # 설정값은 생성자 로그로 이미 확인했지만, "그 설정이 실제로 매 호출마다 리사이즈에
        # 반영되는가"는 별개 질문이다(설정은 맞는데 내부 경로가 다른 값을 쓰는 버그가 있을
        # 수도 있음 - 이전에 우리 LoRA 경로에서 --max_pixels가 조용히 무시되던 사례가 실제로
        # 있었다). qwen_vl_utils/transformers가 내부적으로 쓰는 것과 동일한 smart_resize
        # 공식으로 "이 이미지가 실제로 리사이즈될 크기"를 우리가 직접 계산해서 매 호출마다
        # 찍어본다 - 로그에 매번 몇 픽셀로 들어가는지 눈으로 확인 가능하게.
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

            _ip = self.processor.image_processor
            _patch = getattr(_ip, "patch_size", 14)
            _merge = getattr(_ip, "merge_size", 2)
            _resized_h, _resized_w = smart_resize(
                image.height, image.width,
                factor=_patch * _merge,
                min_pixels=_ip.min_pixels, max_pixels=_ip.max_pixels,
            )
            print(
                f"[gui_actor_grounding.predict_point] 원본={image.width}x{image.height} "
                f"-> 실제 추론 리사이즈={_resized_w}x{_resized_h} "
                f"({_resized_w * _resized_h}px, max_pixels 설정={_ip.max_pixels})"
            )
        except Exception as e:  # noqa: BLE001 - 실측 로그 실패로 실제 추론까지 막으면 안 됨
            print(f"[gui_actor_grounding.predict_point] 해상도 실측 로그 실패(무시하고 진행): {e}")

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a GUI agent. Given a screenshot of the current GUI and "
                            "a human instruction, your task is to locate the screen element "
                            "that corresponds to the instruction."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            },
        ]

        return inference(
            conversation, self.model, self.tokenizer, self.processor,
            use_placeholder=True, topk=topk,
        )

    def generate(
        self, messages: list, max_new_tokens: int = 512,
        temperature: float = 0.0, top_p: float = 1.0,
    ) -> str:
        """
        [v3 추가 - RF 결합용] GUI-Actor 모델로 "일반 텍스트" 답을 받는 경로.
        gui_actor.inference()(포인터 head로 좌표를 뽑는 전용 경로)와는 별개로, RegionFocus의
        judge_inference/next_action_regionfocus_aggregation처럼 좌표가 아니라 자유 텍스트
        (YES/NO, "Selected point: #")로 답해야 하는 단계에 필요하다.

        이게 가능한 근거: HF의 microsoft/GUI-Actor-3B-Qwen2.5-VL 모델 카드 사용 예시가
        gui_actor 패키지 없이 `AutoProcessor` + `Qwen2_5_VLForConditionalGenerationWithPointer`
        만으로 `processor.apply_chat_template(...)` -> `model.generate(**inputs)` -> 일반
        텍스트 디코딩("What animal is on the candy?" 같은 평범한 VQA)이 그대로 되는 걸 보여준다
        - 포인터 head는 모델이 특정 placeholder 토큰을 낼 때만 관여하는 추가 메커니즘이고,
        기반 LM 자체는 표준 causal LM으로 그대로 남아있다는 뜻. 즉 judge/aggregation처럼
        "좌표를 요구하지 않는 프롬프트"에는 gui_actor.inference()를 거칠 필요 없이 이 메서드로
        바로 답을 받을 수 있다. (다만 이 모델은 GUI grounding 데이터로 파인튜닝됐으므로, 일반
        VQA/추론 능력이 base Qwen2.5-VL-3B-Instruct보다 떨어질 수는 있다 - RF의 judge/aggregation
        프롬프트처럼 짧고 구조화된 답을 요구하는 용도로는 실측상 문제없이 쓸 수 있을 것으로 보고
        시도하는 것.)

        qwen.py의 generate_text()/QwenVLModel.generate()와 시그니처를 동일하게 맞춰서,
        agent_loop.py의 duck-typing(reflection_view, judge 등)에도 그대로 꽂아 쓸 수 있게 했다.
        """
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(max_new_tokens=max_new_tokens)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return output_text[0]


def ground(
    gui_actor_model: GUIActorModel,
    instruction: str,
    image,
    debug_text: bool = False,
    task_id=None,
    **_ignored,
    # gui_grounding.ground()의 min_pixels/max_pixels/max_new_tokens/temperature 등,
    # 호출부가 관성적으로 넘길 수 있는 인자들을 조용히 무시한다(GUI-Actor에는 해당 없음).
) -> dict:
    """
    gui_grounding.ground()와 반환 스키마가 100% 동일:
        {
            "result": "positive" | "wrong_format",
            "point": [x_norm, y_norm] | None,  # 원본 이미지 기준 0~1 정규화 좌표
            "raw_response": str,
        }

    agent_loop.py / region_focus.py에서
        from gui_grounding import ground
    를
        from gui_actor_grounding import ground
    로 바꾸고, model 인자로 QwenVLModel 대신 GUIActorModel 인스턴스를 넘기면
    나머지 하류 로직(bbox hit-test, region_focus의 crop 계산 등)은 그대로 동작한다.
    """
    pil_image = Image.open(image) if isinstance(image, str) else image

    _t0 = time.time()
    pred = gui_actor_model.predict_point(instruction, pil_image)
    print(f"[gui_actor_grounding.ground] inference() 완료 - {time.time() - _t0:.1f}초")

    topk_points = pred.get("topk_points") or []
    if not topk_points:
        return {"result": "wrong_format", "point": None, "raw_response": str(pred)}

    px, py = topk_points[0]
    x_norm = max(0.0, min(1.0, float(px)))
    y_norm = max(0.0, min(1.0, float(py)))

    if debug_text:
        from gui_grounding import dump_prompt_debug
        dump_prompt_debug(
            [{"role": "user", "content": [{"type": "text", "text": instruction}]}],
            str(pred), task_id=task_id, step_name="ground_gui_actor",
        )

    return {"result": "positive", "point": [x_norm, y_norm], "raw_response": str(pred)}


def _cli():
    """gui_grounding.py의 _cli()와 대칭되는 단발성 테스트 CLI.

    예:
        python gui_actor_grounding.py --image debug/some_task/screenshot.png \\
            --instruction "click the search bar"
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="스크린샷 이미지 경로")
    ap.add_argument("--instruction", required=True, help="grounding할 지시문")
    ap.add_argument("--model_id", default="microsoft/GUI-Actor-3B-Qwen2.5-VL")
    ap.add_argument(
        "--attn_implementation", default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    ap.add_argument("--min_pixels", type=int, default=None)
    ap.add_argument("--max_pixels", type=int, default=None)
    args = ap.parse_args()

    model = GUIActorModel(
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    result = ground(model, args.instruction, args.image)
    print(result)


if __name__ == "__main__":
    _cli()
