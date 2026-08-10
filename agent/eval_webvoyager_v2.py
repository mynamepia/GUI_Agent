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


# (2026-08-11 추가) 같은 액션이 이만큼 연속으로 반복되면 "막혔다"고 보고 조기 종료한다.
# CAPTCHA에 걸려서 planner가 같은 걸 계속 재시도하는 경우(Allrecipes 사례, planner.py
# docstring 참고)의 일반화된 안전장치 - CAPTCHA뿐 아니라 grounding이 계속 같은 지점을
# 잘못 찍는 등 "어떤 이유로든 진행이 안 되는" 상황을 폭넓게 잡는다.
DEFAULT_STUCK_REPEAT_THRESHOLD = 4


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
                    print(f"[eval_webvoyager.py] 디버그 이미지 저장 실패(무시하고 진행): {e}")
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
    .model 프로퍼티를 내부 객체에 통과시켜서(disable_adapter() 등 내부에서 .model을 쓰는
    코드가 그대로 동작하게) 다른 코드는 이게 debug wrapper인지 전혀 모른다."""

    def __init__(self, inner, recorder: _PromptRecorder, tag: str):
        self._inner = inner
        self._recorder = recorder
        self._tag = tag

    @property
    def model(self):
        return getattr(self._inner, "model", self._inner)

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

    # (2026-08-11 추가) debug_dir가 주어지면 planning/reflection/grounding 각각의 .generate()를
    # _DebugModelView로 감싼다 - 실제로 뭘 호출하는지는 그대로, 프롬프트/응답만 옆에서 기록.
    recorder = _PromptRecorder(debug_dir, save_images=debug_save_images) if debug_dir else None
    if recorder is not None:
        debug_planning_view = _DebugModelView(planning_view, recorder, "planner")
        debug_reflection_view = (
            _DebugModelView(reflection_view, recorder, "reflection") if reflection_view is not None else None
        )
        debug_grounding_click_view = _DebugModelView(grounding_model, recorder, "grounding")
        debug_grounding_answer_view = _DebugModelView(grounding_model, recorder, "answer_extraction")
    else:
        debug_planning_view = planning_view
        debug_reflection_view = reflection_view
        debug_grounding_click_view = grounding_model
        debug_grounding_answer_view = grounding_model

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

        # (2026-08-10 추가) terminate인데 answer가 비어 있으면 별도 QA 호출로 채운다.
        # _extract_final_answer() docstring 참고 - planner LoRA가 애초에 이 필드를 학습에서
        # 못 본 문제라 재시도/reflection으로는 안 고쳐짐. reflection이 반려한 terminate는 위에서
        # 이미 걸러졌으니, 여기 도달하는 terminate는 (reflection이 껐거나) 승인된 것만 남는다.
        if plan.get("action") == "terminate" and not plan.get("answer"):
            try:
                extracted = _extract_final_answer(debug_grounding_answer_view, instruction, screenshot)
            except Exception as e:  # noqa: BLE001 - 최종 답변 추출 실패로 에피소드 전체를 죽이지 않음
                print(f"[agent_step] answer 추출 실패(무시하고 진행): {e}")
                extracted = None
            if extracted:
                plan = dict(plan)
                plan["answer"] = extracted

        env_action = _convert_planner_action_to_env(plan, debug_grounding_click_view, screenshot, ground_kwargs)

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
    2) 같은 액션이 stuck_repeat_threshold회 연속 반복되면 원인(CAPTCHA든 다른 이유든)과
       무관하게 "멈췄다"고 보고 종료 - detect_bot_check()가 못 잡는 케이스(알려지지 않은
       문구를 쓰는 봇 차단 페이지 등)까지 잡아내는 일반화된 안전장치.

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

    fingerprint_streak: list = []
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
                blocked = True
                blocked_reason = f"스텝 진행 중 bot-check 감지: {bot_check.get('reason')}"
                hit_max_steps = False
                break

            fp = _action_fingerprint(action)
            if fingerprint_streak and fingerprint_streak[-1] == fp:
                fingerprint_streak.append(fp)
            else:
                fingerprint_streak = [fp]
            if len(fingerprint_streak) >= stuck_repeat_threshold:
                blocked = True
                blocked_reason = f"같은 액션이 {stuck_repeat_threshold}회 연속 반복됨(멈춘 것으로 판단): {fp}"
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
              judge_repeats=DEFAULT_JUDGE_REPEATS, out_path=None,
              stuck_repeat_threshold=DEFAULT_STUCK_REPEAT_THRESHOLD):
    rows = []
    out_f = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for i, task in enumerate(tasks):
            t0 = time.time()
            traj = run_episode(
                env, task, agent_step_fn, max_steps=max_steps, stuck_repeat_threshold=stuck_repeat_threshold,
            )

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
                    n_repeats=judge_repeats,
                )
            row = {
                "idx": i,
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
                f"[{i + 1}/{len(tasks)}] {status_tag} "
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
                debug_dir=dbg_dir2, debug_save_images=False,
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
    ap.add_argument("--judge", choices=["qwen", "openai"], default="qwen")
    ap.add_argument("--openai_model", default="gpt-4o")
    ap.add_argument("--adapter_dir", default=None,
                     help="Qwen judge용 LoRA 어댑터 (선택). --reuse_agent_model_for_judge를 켜면 무시됨.")
    ap.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--judge_repeats", type=int, default=DEFAULT_JUDGE_REPEATS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
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
    ap.add_argument("--stuck_repeat_threshold", type=int, default=DEFAULT_STUCK_REPEAT_THRESHOLD,
                     help="같은 액션이 이만큼 연속 반복되면(bot-check 신호 유무와 무관하게) 멈춘 것으로 "
                          "보고 조기 종료한다(기본 4).")
    # (2026-08-09 추가) 실제 planner+grounding 정책. --agent_grounding_adapter_dir를 안 주면
    # 기존처럼 dummy_agent_step(즉시 실패)로 동작 - 파이프라인 배선만 확인하고 싶을 때는 그대로 둘 것.
    ap.add_argument("--agent_grounding_adapter_dir", default=None,
                     help="grounding LoRA 체크포인트(예: checkpoints/qwen2.5vl-3b-gui-lora-stage2/"
                          "checkpoint-4130). 지정해야 dummy_agent_step 대신 실제 정책(planner+grounding)이 돈다.")
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
    ap.add_argument("--planner_api_key", default=None, help="미지정시 환경변수 OPENAI_API_KEY 사용")
    ap.add_argument("--planner_api_base_url", default=None,
                     help="OpenAI 호환 엔드포인트(vLLM 등)를 쓸 때 지정 - 미지정시 OpenAI 공식 엔드포인트")
    # (2026-08-11 수정 - 버그) default=False로 돼 있었던 걸 True로 고침. --no_reflect는
    # action="store_false"라 "플래그를 주면 끈다"는 의미인데, default까지 False였던 탓에
    # 플래그를 주든 안 주든 reflection이 항상 꺼진 채로 돌고 있었다(실측: 실제 실행 로그에
    # _reflection_log가 한 번도 안 남음 - CLI로는 reflection을 켤 방법 자체가 없었음).
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

        env = WebVoyagerEnv(captcha_reset_retries=args.captcha_reset_retries)

        if args.planner_backend == "openai" and args.agent_planner_adapter_dir:
            raise SystemExit(
                "--planner_backend openai와 --agent_planner_adapter_dir는 같이 못 씀 - planner가 "
                "API로 도니 로컬 planner LoRA를 로드할 이유가 없음."
            )

        agent_model = None
        if args.agent_grounding_adapter_dir:
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
                    api_key=args.planner_api_key,
                    base_url=args.planner_api_base_url,
                )
                print(
                    f"[eval_webvoyager.py] planner backend = OpenAI API (model={args.planner_api_model!r}) "
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
            )
            if not args.no_debug_dump:
                print(f"[eval_webvoyager.py] 태스크별 프롬프트/응답 덤프 경로: {os.path.abspath(args.debug_dir)}")
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

        try:
            run_batch(
                tasks, env, agent_step_fn, judge_fn,
                max_steps=args.max_steps, judge_repeats=args.judge_repeats, out_path=args.out,
                stuck_repeat_threshold=args.stuck_repeat_threshold,
            )
        finally:
            env.close()