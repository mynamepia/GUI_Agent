"""
api_planner.py

planner.py의 plan_next_action()/plan_with_reflection()이 기대하는 duck-typing 인터페이스
    .generate(messages, max_new_tokens=..., temperature=..., top_p=...) -> str
를 로컬 QwenVLModel이 아니라 OpenAI Chat Completions vision API(GPT-4o 등)로 구현한 버전.

[왜 필요한가]
agent_loop.py의 load_shared_model()이 반환하는 planning_view는 항상 로컬에 로드된
QwenVLModel(+선택적으로 planner LoRA)을 감싼 것이라, planner LoRA가 아직 없거나
(base 모델만으로는 3B가 약하다는 게 planner.py 파일 docstring에도 이미 적혀 있음),
더 강한 baseline과 비교해보고 싶을 때 로컬 모델 말고 API 모델로 planning을 돌려볼
방법이 없었다. 이 파일은 그 자리에 꽂을 수 있는 API 기반 대체 구현체를 제공한다.

[적용 범위 - grounding은 안 건드림]
이 클래스는 planning(및 원하면 reflection)만 대체한다. target_description을 실제
좌표로 바꾸는 grounding(gui_grounding.ground()/region_focus.py)은 여전히 로컬
QwenVLModel(+grounding LoRA)이 담당한다 - grounding LoRA가 학습한 "(x,y)" 텍스트
포맷을 흉내내서 API 모델에게 좌표를 직접 뽑게 하는 건 이 파일의 범위 밖이다
(eval_webvoyager.py의 build_planner_grounding_agent_step()이 grounding_model과
planning_view를 애초에 분리된 인자로 받게 설계돼 있어서, planning_view 자리에만
이 클래스를 넣으면 나머지 파이프라인은 그대로 재사용된다).

[메시지 포맷 변환]
planner.py가 만드는 메시지는 Qwen 챗 템플릿 스타일이다:
    [{"role": "system"/"user", "content": [{"type": "text", "text": ...} |
                                            {"type": "image", "image": PIL.Image}, ...]}]
OpenAI Chat Completions vision API는 이미지를 {"type": "image_url", "image_url": {"url":
"data:image/png;base64,..."}}로 받는다 - _convert_qwen_messages_to_openai()가 이 변환을
담당한다(eval_webvoyager.make_openai_judge()가 judge용으로 이미 하던 것과 같은 방식).

[max_tokens vs max_completion_tokens]
OpenAI가 max_tokens를 max_completion_tokens로 대체했다(구 파라미터는 하위호환으로만
지원, 신규 파라미터가 현재 권장 방식) - 여기서는 max_completion_tokens를 직접 쓴다.

필요 패키지: pip install openai (실제 generate() 호출 시점에만 필요 - lazy import라
openai 미설치 환경에서도 이 파일 import/선택은 문제없음, eval_webvoyager.make_openai_judge()와
동일한 원칙).
"""

import base64
import io
import time

DEFAULT_OPENAI_PLANNER_MODEL = "gpt-4o"


def _is_rate_limit_error(e: Exception) -> bool:
    """
    (2026-08-11 추가 - 실측 크래시 대응) 실제 실행에서 TPM(분당 토큰) rate limit(HTTP 429)에
    걸려서 배치 전체가 죽는 걸 확인했다 - 재시도 없이 그대로 RuntimeError를 올리던 게 원인.
    openai 패키지가 있으면 RateLimitError 타입으로 정확히 판별하고, 없거나(예: 테스트에서
    openai를 mock 모듈로 대체한 경우) 다른 예외 타입으로 감싸져 온 경우엔 메시지 문자열로
    폴백 판별한다.
    """
    try:
        from openai import RateLimitError

        if isinstance(e, RateLimitError):
            return True
    except ImportError:
        pass
    msg = str(e).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg


def _call_with_retry(fn, max_retries=5, base_delay=1.0, max_delay=60.0, on_retry=None):
    """
    (2026-08-11 추가) fn()을 호출하고, rate limit(429) 에러면 지수 백오프(1s -> 2s -> 4s ->
    ... -> max_delay 상한)로 최대 max_retries번까지 재시도한다. rate limit이 아닌 다른
    에러는 즉시 그대로 재발생(네트워크 끊김/인증 실패 등을 무한정 재시도하면 안 되므로).
    max_retries를 다 쓰고도 여전히 rate limit이면 마지막 에러를 그대로 올린다.
    on_retry(attempt, delay, exc): 재시도할 때마다 호출되는 선택적 콜백(로깅/테스트용).
    """
    delay = base_delay
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not _is_rate_limit_error(e) or attempt >= max_retries:
                raise
            if on_retry:
                on_retry(attempt + 1, delay, e)
            time.sleep(delay)
            delay = min(delay * 2, max_delay)
            attempt += 1


def _pil_to_data_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _convert_qwen_messages_to_openai(messages: list) -> list:
    """
    planner.py(및 region_focus.py 등)가 만드는 Qwen 스타일 메시지를 OpenAI Chat
    Completions vision 포맷으로 변환한다.

    system 메시지는 이 프로젝트에서 항상 텍스트만 담고 있어서(planner._SYSTEM_PROMPT,
    _REFLECTION_SYSTEM_PROMPT 둘 다 이미지 없음) OpenAI 관례대로 단순 문자열 content로
    변환한다 - 이미지가 섞인 system 메시지가 들어오면(현재 이 프로젝트에서는 안 일어남)
    방어적으로 list 포맷을 그대로 유지한다.
    """
    converted = []
    for msg in messages:
        role = msg["role"]
        parts = msg["content"]
        openai_parts = []
        for part in parts:
            ptype = part.get("type")
            if ptype == "text":
                openai_parts.append({"type": "text", "text": part["text"]})
            elif ptype == "image":
                openai_parts.append(
                    {"type": "image_url", "image_url": {"url": _pil_to_data_url(part["image"])}}
                )
            else:
                raise ValueError(f"알 수 없는 content part type: {ptype!r} (part={part!r})")

        if role == "system" and all(p["type"] == "text" for p in openai_parts):
            converted.append({"role": "system", "content": "\n".join(p["text"] for p in openai_parts)})
        else:
            converted.append({"role": role, "content": openai_parts})
    return converted


class OpenAIPlannerModel:
    """
    planner.py가 기대하는 .generate(messages, max_new_tokens, temperature, top_p) -> str
    인터페이스를 OpenAI Chat Completions vision API로 구현한 어댑터.

    agent_loop.load_shared_model()이 반환하는 (model, planning_view) 대신, 이 클래스의
    인스턴스를 그대로 planner.py의 plan_next_action()/plan_with_reflection()에
    qwen_model(및 원하면 reflection_model)로 넘기면 로컬 GPU/LoRA 없이 planning이
    API 호출로 돈다. 로컬 QwenVLModel과 duck-typing 인터페이스가 동일해서 planner.py는
    이 클래스가 API인지 로컬 모델인지 전혀 모른다(수정 불필요).
    """

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_PLANNER_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        request_timeout: float = 60.0,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._client = None  # lazy - 실제 generate() 호출 시점에만 openai 패키지/클라이언트 생성

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import (make_openai_judge()와 동일 원칙)

            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(
        self, messages: list, max_new_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0
    ) -> str:
        client = self._get_client()
        openai_messages = _convert_qwen_messages_to_openai(messages)

        kwargs = dict(
            model=self.model,
            messages=openai_messages,
            max_completion_tokens=max_new_tokens,
            timeout=self._request_timeout,
        )
        # temperature<=0 -> QwenVLModel.generate()의 greedy decoding 관례와 맞춘다(완전한
        # 결정론은 OpenAI 쪽에서 보장 안 하지만 - temperature=0으로 최대한 근접).
        if temperature and temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        else:
            kwargs["temperature"] = 0

        def _on_retry(attempt, delay, exc):
            print(
                f"[api_planner.py] rate limit(429) 감지 - {delay:.1f}초 대기 후 재시도 "
                f"({attempt}/{self._max_retries}): {exc}"
            )

        try:
            resp = _call_with_retry(
                lambda: client.chat.completions.create(**kwargs),
                max_retries=self._max_retries, base_delay=self._retry_base_delay, on_retry=_on_retry,
            )
        except Exception as e:  # noqa: BLE001
            # 로컬 QwenVLModel.generate()는 실패하면 예외를 그대로 던지는 게 기본 동작이라
            # (별도 폴백 없음), 인터페이스 일관성을 맞추려고 여기서도 삼키지 않고 그대로
            # 올린다 - planner.py._parse_planner_action()이 이미 "generate()가 이상한/빈
            # 문자열을 반환하는 경우"에 대한 폴백을 갖고 있지만, "API 호출 자체가 실패"하는
            # 경우까지 조용히 흡수해서 빈 문자열을 돌려주면 원인 파악이 더 어려워진다.
            # (2026-08-11 추가) rate limit(429)만 위 _call_with_retry가 자동 재시도하고,
            # 그래도 다 소진되거나 다른 종류의 에러면 여기로 내려와서 RuntimeError로 올라간다.
            raise RuntimeError(f"[api_planner.py] OpenAI API 호출 실패: {e}") from e

        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# mock 기반 단위 테스트 (실제 OpenAI API 호출 없이 변환/연동 로직만 검증)
# ---------------------------------------------------------------------------
def _run_mock_selftest():
    """`python api_planner.py --selftest`"""
    import sys
    import types
    from unittest.mock import MagicMock

    from PIL import Image

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    # --- _pil_to_data_url ---
    url = _pil_to_data_url(Image.new("RGB", (2, 2)))
    check("_pil_to_data_url -> data URL 접두사", url.startswith("data:image/png;base64,"))

    # --- _convert_qwen_messages_to_openai ---
    qwen_messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys prompt"}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (2, 2))},
                {"type": "text", "text": "user text"},
            ],
        },
    ]
    converted = _convert_qwen_messages_to_openai(qwen_messages)
    check("system -> 단순 문자열 content로 변환", converted[0] == {"role": "system", "content": "sys prompt"})
    check("user -> content가 list", isinstance(converted[1]["content"], list))
    check(
        "user -> image가 image_url(data URL)로 변환",
        converted[1]["content"][0]["type"] == "image_url"
        and converted[1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,"),
    )
    check("user -> text 파트 보존", converted[1]["content"][1] == {"type": "text", "text": "user text"})

    try:
        _convert_qwen_messages_to_openai([{"role": "user", "content": [{"type": "video", "video": None}]}])
        check("알 수 없는 content type -> ValueError", False)
    except ValueError:
        check("알 수 없는 content type -> ValueError", True)

    # --- OpenAIPlannerModel.generate() ---
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content='{"reasoning":"r","action":"wait"}'))]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    fake_openai_module = types.ModuleType("openai")
    fake_openai_module.OpenAI = MagicMock(return_value=fake_client)
    sys.modules["openai"] = fake_openai_module
    try:
        planner_model = OpenAIPlannerModel(model="gpt-4o-test", api_key="sk-fake", base_url="https://fake/v1")
        result = planner_model.generate(qwen_messages, max_new_tokens=200, temperature=0.0)
        check("generate() -> 응답 텍스트 반환", result == '{"reasoning":"r","action":"wait"}')
        check("OpenAI() 생성자에 api_key/base_url 전달됨", fake_openai_module.OpenAI.call_args.kwargs == {
            "api_key": "sk-fake", "base_url": "https://fake/v1",
        })
        _, call_kwargs = fake_client.chat.completions.create.call_args
        check("model 전달됨", call_kwargs["model"] == "gpt-4o-test")
        check("max_new_tokens -> max_completion_tokens로 전달됨", call_kwargs["max_completion_tokens"] == 200)
        check("temperature<=0 -> temperature=0으로 정규화", call_kwargs["temperature"] == 0)
        check("temperature<=0일 땐 top_p는 안 실림(그리디)", "top_p" not in call_kwargs)
        check("messages가 OpenAI 포맷으로 변환되어 전달됨", call_kwargs["messages"][0]["role"] == "system")

        fake_client.chat.completions.create.reset_mock()
        planner_model.generate(qwen_messages, temperature=0.7, top_p=0.9)
        _, call_kwargs2 = fake_client.chat.completions.create.call_args
        check("temperature>0 -> temperature/top_p 그대로 전달됨", call_kwargs2["temperature"] == 0.7 and call_kwargs2["top_p"] == 0.9)

        # API 호출 자체가 실패하면 삼키지 않고 RuntimeError로 재발생
        fake_client.chat.completions.create.reset_mock()
        fake_client.chat.completions.create.side_effect = ConnectionError("network down")
        try:
            planner_model.generate(qwen_messages)
            check("API 호출 실패 -> RuntimeError로 재발생", False)
        except RuntimeError as e:
            check("API 호출 실패 -> RuntimeError로 재발생", "OpenAI API 호출 실패" in str(e))

        # (2026-08-11 추가) rate limit(429)이 아닌 에러는 재시도 없이 즉시 실패해야 함
        # (네트워크 끊김/인증 실패를 무한정 재시도하면 안 되므로) - 위 ConnectionError 테스트에서
        # create()가 정확히 1번만 호출됐는지로 확인.
        check("rate limit이 아닌 에러 -> 재시도 없이 1번만 호출됨", fake_client.chat.completions.create.call_count == 1)
    finally:
        del sys.modules["openai"]

    # --- (2026-08-11 추가) _is_rate_limit_error / _call_with_retry ---
    check("_is_rate_limit_error -> '429' 포함되면 True", _is_rate_limit_error(Exception("Error code: 429 - rate_limit_exceeded")))
    check("_is_rate_limit_error -> 'rate_limit' 포함되면 True", _is_rate_limit_error(Exception("rate_limit exceeded")))
    check("_is_rate_limit_error -> 무관한 에러는 False", not _is_rate_limit_error(ConnectionError("network down")))

    orig_sleep = time.sleep
    time.sleep = lambda *a, **k: None  # 재시도 백오프 대기 때문에 테스트가 느려지는 것 방지
    try:
        # 2번 실패(rate limit) 후 3번째에 성공 -> 재시도로 결국 성공해야 함
        call_log = []

        def _flaky():
            call_log.append(1)
            if len(call_log) < 3:
                raise Exception("Error code: 429 - rate_limit_exceeded")
            return "ok"

        retry_log = []
        result_ok = _call_with_retry(
            _flaky, max_retries=5, base_delay=0.01,
            on_retry=lambda attempt, delay, exc: retry_log.append(attempt),
        )
        check("_call_with_retry -> 2번 실패 후 3번째에 성공", result_ok == "ok" and len(call_log) == 3)
        check("_call_with_retry -> on_retry가 실패 횟수만큼 호출됨", retry_log == [1, 2])

        # rate limit이 계속되면 max_retries만큼만 재시도하고 결국 그 에러를 그대로 올림
        always_fail_calls = {"n": 0}

        def _always_fail():
            always_fail_calls["n"] += 1
            raise Exception("Error code: 429 - rate_limit_exceeded")

        try:
            _call_with_retry(_always_fail, max_retries=3, base_delay=0.01)
            check("_call_with_retry -> max_retries 소진 후에도 계속 실패하면 예외 재발생", False)
        except Exception as e:
            check("_call_with_retry -> max_retries 소진 후에도 계속 실패하면 예외 재발생", "429" in str(e))
        check("_call_with_retry -> 최초 시도 + max_retries(3) = 총 4번 호출", always_fail_calls["n"] == 4)

        # rate limit이 아닌 에러는 즉시(재시도 없이) 재발생
        non_rate_limit_calls = {"n": 0}

        def _fail_other():
            non_rate_limit_calls["n"] += 1
            raise ConnectionError("network down")

        try:
            _call_with_retry(_fail_other, max_retries=5, base_delay=0.01)
            check("_call_with_retry -> rate limit 아니면 재시도 없이 즉시 실패", False)
        except ConnectionError:
            check("_call_with_retry -> rate limit 아니면 재시도 없이 즉시 실패", True)
        check("_call_with_retry -> rate limit 아니면 1번만 호출됨(재시도 안 함)", non_rate_limit_calls["n"] == 1)
    finally:
        time.sleep = orig_sleep

    # --- (2026-08-11 추가) OpenAIPlannerModel.generate()가 429를 자동 재시도로 흡수하는지 통합 확인 ---
    fake_response_retry = MagicMock()
    fake_response_retry.choices = [MagicMock(message=MagicMock(content='{"reasoning":"r","action":"wait"}'))]
    fake_client_retry = MagicMock()
    fake_client_retry.chat.completions.create.side_effect = [
        Exception("Error code: 429 - rate_limit_exceeded"),
        fake_response_retry,
    ]
    fake_openai_module_retry = types.ModuleType("openai")
    fake_openai_module_retry.OpenAI = MagicMock(return_value=fake_client_retry)
    sys.modules["openai"] = fake_openai_module_retry
    orig_sleep2 = time.sleep
    time.sleep = lambda *a, **k: None
    try:
        planner_model_retry = OpenAIPlannerModel(model="gpt-4o-test", retry_base_delay=0.01)
        result_retry = planner_model_retry.generate(qwen_messages)
        check(
            "generate() -> 429 한 번은 자동 재시도로 흡수하고 결국 정상 응답 반환",
            result_retry == '{"reasoning":"r","action":"wait"}',
        )
        check("generate() -> create()가 재시도 포함 2번 호출됨", fake_client_retry.chat.completions.create.call_count == 2)
    finally:
        time.sleep = orig_sleep2
        del sys.modules["openai"]

    # --- planner.py와의 실제 연동 확인 (duck-typing 인터페이스 검증) ---
    fake_response2 = MagicMock()
    fake_response2.choices = [
        MagicMock(message=MagicMock(content='{"reasoning": "ok", "action": "left_click", "target_description": "the search box"}'))
    ]
    fake_client2 = MagicMock()
    fake_client2.chat.completions.create.return_value = fake_response2
    fake_openai_module2 = types.ModuleType("openai")
    fake_openai_module2.OpenAI = MagicMock(return_value=fake_client2)
    sys.modules["openai"] = fake_openai_module2
    try:
        from planner import plan_next_action

        planner_model2 = OpenAIPlannerModel(model="gpt-4o-test")
        plan = plan_next_action(planner_model2, "search for cats", Image.new("RGB", (4, 4)))
        check(
            "planner.plan_next_action()이 OpenAIPlannerModel을 그대로 받아 정상 동작함(duck-typing)",
            plan["action"] == "left_click" and plan["target_description"] == "the search box",
        )
    finally:
        del sys.modules["openai"]

    n_fail = sum(1 for _, ok in checks if not ok)
    for name, ok in checks:
        print(("[OK]  " if ok else "[FAIL]") + " " + name)
    print(f"\n{len(checks) - n_fail}/{len(checks)} passed")
    if n_fail:
        raise SystemExit(1)


def _cli():
    """`python api_planner.py --image X --task Y [--reflect]` (수동 테스트용, 실제 API 호출됨)"""
    import argparse
    import json

    from PIL import Image

    ap = argparse.ArgumentParser(
        description="OpenAI API 기반 planner 수동 테스트용 CLI (실제 API 키/호출 필요)."
    )
    ap.add_argument("--selftest", action="store_true", help="실제 API 호출 없이 로직만 mock으로 검증")
    ap.add_argument("--image", help="스크린샷 이미지 경로")
    ap.add_argument("--task", help="태스크 지시문")
    ap.add_argument("--model", default=DEFAULT_OPENAI_PLANNER_MODEL)
    ap.add_argument("--api_key", default=None, help="미지정시 환경변수 OPENAI_API_KEY 사용")
    ap.add_argument("--base_url", default=None, help="OpenAI 호환 엔드포인트(vLLM 등)를 쓸 때 지정")
    ap.add_argument("--reflect", action="store_true", help="plan_next_action 대신 plan_with_reflection 사용")
    ap.add_argument("--max_iterations", type=int, default=2)
    args = ap.parse_args()

    if args.selftest:
        _run_mock_selftest()
        return

    if not args.image or not args.task:
        raise SystemExit("--image와 --task 필요 (또는 --selftest)")

    from planner import plan_next_action, plan_with_reflection

    model = OpenAIPlannerModel(model=args.model, api_key=args.api_key, base_url=args.base_url)
    screenshot = Image.open(args.image)
    if args.reflect:
        result = plan_with_reflection(model, args.task, screenshot, max_iterations=args.max_iterations)
    else:
        result = plan_next_action(model, args.task, screenshot)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()