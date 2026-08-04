"""
env_webvoyager.py

WebVoyager 스타일 벤치마크(실제 살아있는 웹사이트 15개, MinorJerry/WebVoyager 저장소의
태스크 jsonl 포맷: {"web_name":..., "id":..., "ques":..., "web":...})를 위한 환경 wrapper.
MiniWob과 달리 패키징된 gymnasium 라이브러리가 없어서 Selenium을 직접 써서 만들었다
(env_miniwob.py와 인터페이스(reset/execute_action/close)는 최대한 맞춤).

[env_miniwob.py와의 차이 - 왜 이렇게 짰는지]
1) reward/성공 판정이 없음: MiniWob은 gymnasium env가 reward/terminated를 자동으로 줬지만,
   실제 사이트는 그런 게 없다. 이 wrapper는 순수하게 "탐색 + 액션 실행 + 스크린샷"만
   담당하고, 성공 판정은 WebVoyager 원 논문처럼 별도 LLM judge가 하거나 나중에
   agent_loop/eval 스크립트가 담당해야 한다 - execute_action()은 항상 reward=None,
   terminated=False를 반환한다.
2) 화면 크기를 우리가 직접 통제 가능: MiniWob은 CSS에 160x210으로 하드코딩돼 있어서
   못 키웠지만, 실제 사이트는 반응형이라 window_size를 키우면 레이아웃도 같이 커진다 -
   MiniWob에서 WebVoyager로 넘어온 이유(진짜 grounding 난이도) 그대로.
3) 좌표 클릭/타이핑은 Selenium ActionChains의 offset 방식 대신 Chrome DevTools
   Protocol(CDP)의 Input.dispatchMouseEvent / dispatchKeyEvent / insertText를 직접 쓴다.
   ActionChains의 좌표 기반 클릭은 스크롤 상태/브라우저 버전에 따라 스크린샷과 어긋나는
   이슈가 잘 알려져 있어서, 스크린샷(get_screenshot_as_png, 이것도 내부적으로 CDP 기반)과
   동일한 좌표계를 쓰는 CDP 마우스 이벤트가 "grounding이 찍은 좌표를 그대로 실행"하기에
   더 안전하다.

[검증 관련 안내]
이 코드를 짠 샌드박스도 env_miniwob.py 때와 마찬가지로 Chrome을 못 깔아서(root 권한 없음,
바이너리 다운로드 네트워크 차단) 실제 브라우저로 끝까지 확인은 못 했다. CDP
명령 조립 로직(_click/_scroll/_type/_key 등)은 driver를 MagicMock으로 대체해서 단위
테스트했다 (`python env_webvoyager.py --selftest`). 실제 사이트 로딩/클릭까지는
Chrome+chromedriver 있는 로컬 환경에서 최종 확인 필요 - env_miniwob.py는 이미 그
환경에서 성공했으니 Chrome/chromedriver 설치 자체는 문제 없을 것으로 예상.

필요 패키지: pip install selenium pillow (miniwob 설치 때 selenium은 이미 같이 깔렸을 것)
"""

import io
import json
import time
from PIL import Image

DEFAULT_WINDOW_SIZE = (1280, 800)  # WaveUI desktop 샘플 해상도(1280x720)와 비슷하게 기본값 설정


def _make_driver(window_size, headless=True, user_agent=None):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    options = Options()
    w, h = window_size
    options.add_argument(f"--window-size={w},{h}")
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
    # 서버/컨테이너(root)에서 흔한 크래시 원인 두 가지 방지 - env_miniwob.py에서
    # --render(비-headless) 때 겪었던 크래시와 같은 종류의 문제를 사전에 막는다.
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    if user_agent:
        options.add_argument(f"--user-agent={user_agent}")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(w, h)
    return driver


class WebVoyagerEnv:
    """
    reset(task)            -> (screenshot: PIL.Image, task_info: dict)
    execute_action(action) -> (screenshot, reward(=None), terminated(=False), truncated(=False), task_info)
    close()

    task: {"web": url, "ques": instruction, ...} dict(WebVoyager jsonl 포맷 그대로) 또는
          (url, instruction) 튜플로 직접 지정 가능.

    action: gui_grounding.ComputerUseTool과 동일한 스키마
        {"action": "left_click", "coordinate": [x, y]}
        {"action": "type", "text": "hello"}
        {"action": "key", "text": "Enter"}                     # Selenium Keys 이름 or 단일 문자
        {"action": "scroll", "coordinate": [x, y], "text": "down", "amount": 300}
        {"action": "wait", "time": 1.0}
    terminate는 execute_action으로 보내지 말 것 - agent_loop가 처리(driver에 안 보냄).
    """

    def __init__(self, window_size=DEFAULT_WINDOW_SIZE, headless=True, page_load_timeout=20, user_agent=None):
        self.window_size = window_size
        self.headless = headless
        self.page_load_timeout = page_load_timeout
        self.user_agent = user_agent
        self.driver = None
        self.task_info = None

    # ------------------------------------------------------------------
    def reset(self, task):
        if self.driver is not None:
            self.close()
        self.driver = _make_driver(self.window_size, headless=self.headless, user_agent=self.user_agent)
        self.driver.set_page_load_timeout(self.page_load_timeout)

        url, instruction, extra = self._parse_task(task)
        try:
            self.driver.get(url)
        except Exception as e:
            # 느린 페이지 등으로 load timeout이 나도 이미 렌더링된 부분으로 계속 진행한다
            # (WebVoyager 원 논문도 완전한 load를 강제하지 않는 편) - 완전 실패면 스크린샷
            # 자체가 비어있을 수 있으니 호출부에서 확인하는 걸 권장.
            print(f"[env_webvoyager.py] driver.get({url!r}) 중 예외(무시하고 진행): {e}")
        time.sleep(1.0)  # 초기 렌더/스크립트 실행 여유

        self.task_info = {"instruction": instruction, "url": url, **extra}
        return self._screenshot(), dict(self.task_info)

    def _parse_task(self, task):
        if isinstance(task, dict):
            url = task.get("web") or task.get("url")
            instruction = task.get("ques") or task.get("instruction")
            extra = {k: v for k, v in task.items() if k not in ("web", "url", "ques", "instruction")}
            return url, instruction, extra
        if isinstance(task, (list, tuple)) and len(task) == 2:
            return task[0], task[1], {}
        raise ValueError(f"알 수 없는 task 포맷: {task!r} (dict 또는 (url, instruction) 필요)")

    # ------------------------------------------------------------------
    def _screenshot(self) -> Image.Image:
        png_bytes = self.driver.get_screenshot_as_png()
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")

    # ------------------------------------------------------------------
    def execute_action(self, action: dict):
        if self.driver is None:
            raise RuntimeError("reset()을 먼저 호출해야 함")

        act_type = action.get("action")
        if act_type == "terminate":
            raise ValueError(
                "terminate는 execute_action()이 아니라 agent_loop에서 처리해야 함 (드라이버에 안 보냄)"
            )

        dispatch = {
            "left_click": self._click,
            "double_click": self._double_click,
            "right_click": self._right_click,
            "mouse_move": self._mouse_move,
            "left_click_drag": self._drag,
            "scroll": self._scroll,
            "type": self._type,
            "key": self._key,
            "wait": self._wait,
        }
        fn = dispatch.get(act_type)
        if fn is None:
            raise ValueError(f"알 수 없는 action: {act_type!r}")
        fn(action)

        # reward/terminated는 이 wrapper의 책임이 아님(파일 상단 docstring 참고) - 항상 None/False.
        return self._screenshot(), None, False, False, dict(self.task_info)

    # ------------------------------------------------------------------
    # CDP 기반 좌표 액션들
    # ------------------------------------------------------------------
    def _coord(self, action) -> tuple:
        c = action.get("coordinate")
        if c is None or len(c) != 2:
            raise ValueError(f"coordinate=[x,y]가 필요함, 받은 값: {c!r}")
        return float(c[0]), float(c[1])

    def _cdp_mouse(self, event_type, x, y, button="left", click_count=1):
        self.driver.execute_cdp_cmd(
            "Input.dispatchMouseEvent",
            {
                "type": event_type,
                "x": x,
                "y": y,
                "button": button,
                "clickCount": click_count,
            },
        )

    def _click(self, action):
        x, y = self._coord(action)
        self._cdp_mouse("mouseMoved", x, y)
        self._cdp_mouse("mousePressed", x, y, click_count=1)
        self._cdp_mouse("mouseReleased", x, y, click_count=1)

    def _double_click(self, action):
        x, y = self._coord(action)
        self._cdp_mouse("mouseMoved", x, y)
        self._cdp_mouse("mousePressed", x, y, click_count=2)
        self._cdp_mouse("mouseReleased", x, y, click_count=2)

    def _right_click(self, action):
        x, y = self._coord(action)
        self._cdp_mouse("mouseMoved", x, y)
        self._cdp_mouse("mousePressed", x, y, button="right", click_count=1)
        self._cdp_mouse("mouseReleased", x, y, button="right", click_count=1)

    def _mouse_move(self, action):
        x, y = self._coord(action)
        self._cdp_mouse("mouseMoved", x, y)

    def _drag(self, action):
        # ComputerUseTool 스키마엔 시작점 필드가 따로 없어서, coordinate 하나만으로 드래그의
        # 시작/끝을 둘 다 결정할 수 없음 - env_miniwob.py와 동일한 이유로 아직 미구현.
        raise NotImplementedError(
            "left_click_drag은 시작점 정보가 스키마에 없어 아직 미구현 (mousedown/up을 별도 "
            "액션으로 두거나 스키마에 start_coordinate를 추가한 뒤 구현할 것)"
        )

    def _scroll(self, action):
        if action.get("coordinate"):
            x, y = self._coord(action)
        else:
            x, y = self.window_size[0] / 2, self.window_size[1] / 2
        direction = (action.get("text") or "down").lower()
        amount = float(action.get("amount", 300))
        delta_y = amount if direction == "down" else -amount
        self.driver.execute_cdp_cmd(
            "Input.dispatchMouseEvent",
            {"type": "mouseWheel", "x": x, "y": y, "deltaX": 0, "deltaY": delta_y},
        )

    # ------------------------------------------------------------------
    # 키보드 액션
    # ------------------------------------------------------------------
    def _type(self, action):
        # CDP Input.insertText: 현재 포커스된 요소에 텍스트를 삽입하고 input 이벤트도 같이
        # 발생시킨다(Chrome 66+ 기준 React 등 controlled input에서도 대체로 잘 동작함).
        # 특정 사이트에서 입력 이벤트를 못 잡으면, 문자 하나씩 dispatchKeyEvent로 보내는
        # 방식으로 교체가 필요할 수 있음 - 지금은 범위 밖.
        text = action.get("text", "")
        self.driver.execute_cdp_cmd("Input.insertText", {"text": text})

    def _key(self, action):
        from selenium.webdriver.common.keys import Keys

        key_name = action.get("text", "")
        key_value = getattr(Keys, key_name.upper(), None)
        if key_value is None:
            key_value = key_name  # 단일 문자(예: "a")는 그대로 사용
        try:
            self.driver.switch_to.active_element.send_keys(key_value)
        except Exception as e:
            raise RuntimeError(f"key 액션 실패 (text={key_name!r}): {e}") from e

    def _wait(self, action):
        time.sleep(float(action.get("time", 1.0)))

    # ------------------------------------------------------------------
    def close(self):
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None


def load_webvoyager_tasks(jsonl_path: str, web_name: str | None = None):
    """
    WebVoyager 저장소(https://github.com/MinorJerry/WebVoyager)의 태스크 jsonl을 읽는다.
    각 줄이 {"web_name": ..., "id": ..., "ques": ..., "web": ...} 형태라고 가정.
    web_name을 주면 그 사이트 태스크만 필터링(예: "Wikipedia").
    """
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if web_name and rec.get("web_name") != web_name:
                continue
            tasks.append(rec)
    return tasks


# ---------------------------------------------------------------------------
# mock 기반 단위 테스트 (실제 브라우저 없이 액션 dispatch/CDP 커맨드 구성만 검증)
# ---------------------------------------------------------------------------
def _fake_png_bytes():
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _run_mock_selftest():
    """
    실제 Chrome 없이 액션 dispatch 로직만 검증. driver를 MagicMock으로 대체해서
    execute_cdp_cmd/send_keys가 올바른 인자로 호출되는지 확인한다.
    `python env_webvoyager.py --selftest`로 실행.
    """
    from unittest.mock import MagicMock

    env = WebVoyagerEnv.__new__(WebVoyagerEnv)  # __init__(driver 필요) 안 타고 순수 로직만 테스트
    env.window_size = (1280, 800)
    env.task_info = {"instruction": "test", "url": "http://example.com"}
    env.driver = MagicMock()
    env.driver.get_screenshot_as_png.return_value = _fake_png_bytes()

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    # left_click -> mouseMoved + mousePressed + mouseReleased (CDP), 좌표/clickCount 확인
    env.execute_action({"action": "left_click", "coordinate": [100, 200]})
    calls = [c.args[1] for c in env.driver.execute_cdp_cmd.call_args_list if c.args[0] == "Input.dispatchMouseEvent"]
    check("left_click -> mouseMoved/Pressed/Released 3콜", len(calls) == 3)
    check("left_click -> 좌표 보존", all(c["x"] == 100 and c["y"] == 200 for c in calls))
    check("left_click -> clickCount=1", calls[1]["clickCount"] == 1 and calls[2]["clickCount"] == 1)
    env.driver.reset_mock()

    # double_click -> clickCount=2
    env.execute_action({"action": "double_click", "coordinate": [10, 20]})
    calls = [c.args[1] for c in env.driver.execute_cdp_cmd.call_args_list if c.args[0] == "Input.dispatchMouseEvent"]
    check("double_click -> clickCount=2", calls[1]["clickCount"] == 2)
    env.driver.reset_mock()

    # right_click -> button="right"
    env.execute_action({"action": "right_click", "coordinate": [5, 5]})
    calls = [c.args[1] for c in env.driver.execute_cdp_cmd.call_args_list if c.args[0] == "Input.dispatchMouseEvent"]
    check("right_click -> button=right", all(c["button"] == "right" for c in calls[1:]))
    env.driver.reset_mock()

    # scroll 방향 (up -> deltaY 음수, down -> 양수)
    env.execute_action({"action": "scroll", "coordinate": [1, 1], "text": "up"})
    call = [c.args[1] for c in env.driver.execute_cdp_cmd.call_args_list if c.args[1].get("type") == "mouseWheel"][0]
    check("scroll up -> deltaY 음수", call["deltaY"] < 0)
    env.driver.reset_mock()
    env.execute_action({"action": "scroll", "coordinate": [1, 1], "text": "down"})
    call = [c.args[1] for c in env.driver.execute_cdp_cmd.call_args_list if c.args[1].get("type") == "mouseWheel"][0]
    check("scroll down -> deltaY 양수", call["deltaY"] > 0)
    env.driver.reset_mock()

    # scroll 좌표 생략 -> 화면 중앙 기본값
    env.execute_action({"action": "scroll", "text": "down"})
    call = [c.args[1] for c in env.driver.execute_cdp_cmd.call_args_list if c.args[1].get("type") == "mouseWheel"][0]
    check("scroll 좌표 생략 -> 화면 중앙", call["x"] == 640 and call["y"] == 400)
    env.driver.reset_mock()

    # type -> Input.insertText
    env.execute_action({"action": "type", "text": "hello"})
    call = [c.args[1] for c in env.driver.execute_cdp_cmd.call_args_list if c.args[0] == "Input.insertText"][0]
    check("type -> insertText 텍스트 보존", call["text"] == "hello")
    env.driver.reset_mock()

    # key -> active_element.send_keys 호출됨
    env.execute_action({"action": "key", "text": "Enter"})
    check("key(Enter) -> send_keys 호출", env.driver.switch_to.active_element.send_keys.called)
    env.driver.reset_mock()

    # coordinate 누락 -> ValueError
    try:
        env.execute_action({"action": "left_click"})
        check("coordinate 누락 -> ValueError", False)
    except ValueError:
        check("coordinate 누락 -> ValueError", True)

    # 미구현 액션
    try:
        env.execute_action({"action": "left_click_drag", "coordinate": [0, 0]})
        check("left_click_drag -> NotImplementedError", False)
    except NotImplementedError:
        check("left_click_drag -> NotImplementedError", True)

    # terminate는 execute_action에서 거부
    try:
        env.execute_action({"action": "terminate"})
        check("terminate -> ValueError", False)
    except ValueError:
        check("terminate -> ValueError", True)

    # 알 수 없는 action -> ValueError
    try:
        env.execute_action({"action": "fly"})
        check("알 수 없는 action -> ValueError", False)
    except ValueError:
        check("알 수 없는 action -> ValueError", True)

    # 스크린샷 PNG bytes -> PIL 변환
    img = env._screenshot()
    check("screenshot PNG -> PIL.Image", img.size == (2, 2))

    # task 파싱 (dict / tuple 둘 다)
    dummy = WebVoyagerEnv.__new__(WebVoyagerEnv)
    url, instr, extra = dummy._parse_task({"web": "http://x.com", "ques": "do X", "web_name": "X", "id": "X--0"})
    check("task dict 파싱", url == "http://x.com" and instr == "do X" and extra == {"web_name": "X", "id": "X--0"})
    url2, instr2, extra2 = dummy._parse_task(("http://y.com", "do Y"))
    check("task tuple 파싱", url2 == "http://y.com" and instr2 == "do Y" and extra2 == {})

    n_fail = sum(1 for _, ok in checks if not ok)
    for name, ok in checks:
        print(("[OK]  " if ok else "[FAIL]") + " " + name)
    print(f"\n{len(checks) - n_fail}/{len(checks)} passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="시작 URL (수동 테스트용)")
    ap.add_argument("--instruction", default=None, help="태스크 지시문 (수동 테스트용)")
    ap.add_argument("--tasks_jsonl", default=None, help="WebVoyager 태스크 jsonl 경로")
    ap.add_argument("--web_name", default=None, help="tasks_jsonl에서 필터링할 사이트 이름 (예: Wikipedia)")
    ap.add_argument("--no_headless", dest="headless", action="store_false", default=True,
                     help="브라우저 창 띄워서 눈으로 확인 (GUI 있는 로컬 환경에서만)")
    ap.add_argument("--width", type=int, default=DEFAULT_WINDOW_SIZE[0])
    ap.add_argument("--height", type=int, default=DEFAULT_WINDOW_SIZE[1])
    ap.add_argument("--selftest", action="store_true", help="실제 브라우저 없이 액션 dispatch 로직만 mock으로 검증")
    args = ap.parse_args()

    if args.selftest:
        _run_mock_selftest()
    else:
        if args.tasks_jsonl:
            tasks = load_webvoyager_tasks(args.tasks_jsonl, web_name=args.web_name)
            if not tasks:
                raise SystemExit("조건에 맞는 태스크가 없음")
            task = tasks[0]
        elif args.url and args.instruction:
            task = (args.url, args.instruction)
        else:
            task = ("https://en.wikipedia.org", "Find out who wrote the article on Python (programming language).")

        env = WebVoyagerEnv(window_size=(args.width, args.height), headless=args.headless)
        screenshot, task_info = env.reset(task)
        print("instruction:", task_info["instruction"])
        print("url:", task_info["url"])
        print("screenshot size:", screenshot.size)
        screenshot.save("/tmp/webvoyager_reset.png")
        print("[env_webvoyager.py] reset 완료. screenshot 저장: /tmp/webvoyager_reset.png")
        env.close()