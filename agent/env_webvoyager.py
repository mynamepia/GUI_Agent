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

[위치/통화 강제 영미권 고정 - 2026-08-11 추가]
실행 환경(IP)이 한국이면 브라우저가 별다른 설정 없이는 한국어/원화(KRW)로 페이지를 보여주는
게 실측으로 확인됐다(Amazon - WebVoyager 평가 결과에서 달러 가격 조건 태스크의 답이 원화로
나와서 judge가 조건 대조를 못 하고 오탐한 사례). _make_driver()가 Accept-Language 헤더/
navigator.language/타임존/위치를 en-US·뉴욕 기준으로 CDP로 고정하고(_apply_locale_overrides()),
그걸로도 못 이기는 사이트(쿠키로 언어/통화를 우선 판단하는 Amazon 등)는 reset()이
WebVoyagerEnv._force_site_locale()로 사이트별 쿠키를 한 번 더 심는다(_SITE_LOCALE_COOKIES).
단, 이건 전부 브라우저/쿠키 시그널이고 실제 접속 IP 자체를 바꾸는 게 아니라서, 서버가 쿠키도
없이 순수 IP 지리위치만으로 지역을 판단하는 사이트까지는 완전히 못 이길 수 있음 - 그런 경우
확인되면 실제 미국 리전 프록시가 필요하고 이건 이 파일의 책임 범위 밖이다.

필요 패키지: pip install selenium pillow (miniwob 설치 때 selenium은 이미 같이 깔렸을 것)
"""

import io
import json
import os
import time
from PIL import Image

DEFAULT_WINDOW_SIZE = (1280, 800)  # WaveUI desktop 샘플 해상도(1280x720)와 비슷하게 기본값 설정

# (2026-08-10 추가) 실측: Allrecipes가 headless Selenium을 봇으로 감지해서 CAPTCHA를 띄우는 게
# 확인됨(planner/reflection이 정상 작동해도 CAPTCHA는 애초에 풀 수 없어서 태스크 자체가 막힘) -
# 흔히 알려진 Selenium 자동화 탐지 우회 기본값. 실제 최신 데스크톱 Chrome의 UA 문자열을 흉내내서
# "headless Chrome"이라는 티가 나는 기본 UA를 대체한다. 필요하면 WebVoyagerEnv(user_agent=...)로
# 덮어쓸 수 있음.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# (2026-08-11 추가 - 위치/통화 강제 영미권 고정) 실측: 이 프로젝트를 돌리는 환경(한국 IP)에서
# Amazon 같은 사이트에 접속하면 브라우저가 별다른 설정 없이는 한국어/원화(KRW)로 페이지를
# 보여준다 - WebVoyager 평가 결과 jsonl에서 "50~75달러 사이" 같은 가격 조건 태스크의
# final_answer가 "KRW 28,343"처럼 원화로 나와서 judge가 달러 조건과 대조를 못 하고 오탐(success)
# 처리한 게 확인됨. _make_driver()가 이 상수들로 Accept-Language/navigator.language/타임존/
# 위치를 en-US·뉴욕 기준으로 강제 고정한다(_apply_locale_overrides() 참고).
DEFAULT_ACCEPT_LANGUAGE = "en-US,en;q=0.9"
DEFAULT_TIMEZONE_ID = "America/New_York"
DEFAULT_GEOLOCATION = {"latitude": 40.7128, "longitude": -74.0060, "accuracy": 100}  # New York, US

# (2026-08-11 추가) 사이트별로 언어/통화 쿠키를 우선적으로 신뢰하는 경우가 있어서(Amazon이 그런
# 케이스로 확인됨 - _force_site_locale() 참고), 브라우저 시그널만으로 안 잡히면 사이트별로
# 쿠키를 직접 심어서 한 번 더 보정한다. 키는 URL에 부분 매칭시킬 문자열(소문자), 값은
# (쿠키 dict, ...) 튜플. 다른 WebVoyager 사이트에서 같은 증상이 확인되면 여기 추가하면 됨.
_SITE_LOCALE_COOKIES = {
    "amazon": (
        {"name": "i18n-prefs", "value": "USD"},
        {"name": "lc-main", "value": "en_US"},
    ),
}

# (2026-08-11 추가) detect_bot_check()가 title/URL에서 찾는 흔한 CAPTCHA/bot-check 신호들.
# 완벽한 목록이 아니라 보수적인 휴리스틱 - 여기 없는 문구를 쓰는 차단 페이지는 못 잡지만,
# eval_webvoyager_v2.run_episode()의 stuck-repeat 안전장치가 그런 경우도 결국 잡아낸다
# (같은 액션이 계속 반복되면 원인 불문 조기 종료).
_BOT_CHECK_KEYWORDS = (
    "captcha",
    "recaptcha",
    "hcaptcha",
    "cloudflare",
    "just a moment",
    "checking your browser",
    "verify you are human",
    "unusual traffic",
    "access denied",
    "사람인지 확인",
)


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
    # (2026-08-10 추가) Selenium 자동화 탐지 우회 - 아래 세 가지가 봇 탐지 스크립트들이 흔히
    # 확인하는 시그널이다: (1) --disable-blink-features=AutomationControlled로 크로미움이
    # 자동화 플래그를 노출하는 blink 기능을 꺼서 navigator.webdriver 등 관련 흔적을 줄임,
    # (2) excludeSwitches=["enable-automation"]으로 "Chrome이 자동화 소프트웨어에 의해
    # 제어되고 있습니다" 인포바/관련 시그널 제거, (3) useAutomationExtension=False로 셀레니움
    # 기본 자동화 확장을 안 씀(이것도 탐지에 잘 걸리는 흔적). Allrecipes CAPTCHA 실측 이후 추가.
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(f"--user-agent={user_agent or DEFAULT_USER_AGENT}")
    # (2026-08-11 추가 - 위치/통화 강제 영미권 고정) Chrome이 기본적으로 시스템 로케일을
    # 그대로 따라가는 게 한국어/원화로 보이는 원인 중 하나라, 브라우저 UI 언어 자체를
    # en-US로 고정한다. 나머지(Accept-Language 헤더, navigator.language, 타임존/위치)는
    # 드라이버 생성 후 _apply_locale_overrides()에서 CDP로 마저 고정한다.
    options.add_argument("--lang=en-US")
    options.add_experimental_option("prefs", {"intl.accept_languages": "en-US,en"})
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(w, h)
    # (2026-08-10 추가) navigator.webdriver 프로퍼티를 자바스크립트 레벨에서 지운다 -
    # Options만으로는 안 지워지는 잔여 시그널이라, 새 문서가 로드될 때마다(모든 페이지 이동에
    # 대해) 이 스크립트가 먼저 실행되도록 CDP로 등록해둔다. 흔히 알려진 셀레니움 탐지 우회
    # 트릭(navigator.webdriver === true면 자동화로 간주하는 사이트가 많음).
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
        )
    except Exception as e:  # noqa: BLE001 - 이 우회가 실패해도 브라우저 자체는 계속 쓸 수 있어야 함
        print(f"[env_webvoyager.py] navigator.webdriver 우회 스크립트 등록 실패(무시하고 진행): {e}")
    _apply_locale_overrides(driver)
    return driver


def _apply_locale_overrides(driver, accept_language=DEFAULT_ACCEPT_LANGUAGE,
                             timezone_id=DEFAULT_TIMEZONE_ID, geolocation=DEFAULT_GEOLOCATION):
    """
    (2026-08-11 추가 - 위치/통화 강제 영미권 고정) --lang=en-US 옵션만으로는 안 잡히는
    시그널들을 CDP로 추가 고정한다. navigator.webdriver 우회와 같은 패턴으로 각 CDP 명령을
    개별 try/except로 감싼다 - Chrome/chromedriver 버전에 따라 일부 명령(특히
    Emulation.setLocaleOverride는 비교적 최근에 추가된 CDP 메서드라 구버전엔 없을 수 있음)이
    실패해도 나머지는 계속 적용되고 브라우저 자체는 계속 쓸 수 있어야 한다.

    - Network.setExtraHTTPHeaders: 모든 요청에 Accept-Language: en-US를 강제(서버가 언어를
      판단하는 가장 흔한 경로).
    - Page.addScriptToEvaluateOnNewDocument: navigator.language/navigator.languages를
      JS 레벨에서 en-US로 고정(헤더 대신 이 값을 읽는 사이트 대비).
    - Emulation.setLocaleOverride: 있으면 Intl API 등 브라우저 엔진 로케일까지 통째로
      en-US로 고정(가장 강력하지만 CDP 버전 의존적).
    - Emulation.setTimezoneOverride / setGeolocationOverride: 타임존/위경도로 지역을
      판단하는 사이트에 대한 추가 방어선(뉴욕 기준 좌표).

    [한계] 전부 브라우저 쪽 시그널이라, 접속 IP 자체로 지역/통화를 판단하는 서버까지는 못
    이긴다(실측: Amazon이 이 케이스로 보임 - 이 헤더/스크립트 오버라이드만으론 여전히 원화가
    나올 수 있음) - IP 지리위치 자체를 바꾸려면 실제 미국 리전 프록시가 필요하고 이건 이
    함수의 책임 범위 밖이다. 쿠키로 언어/통화를 우선 판단하는 사이트(Amazon 확인됨)는
    WebVoyagerEnv._force_site_locale()에서 사이트별 쿠키로 한 번 더 보정한다.
    """
    try:
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {"headers": {"Accept-Language": accept_language}})
    except Exception as e:  # noqa: BLE001
        print(f"[env_webvoyager.py] Accept-Language 헤더 강제 실패(무시하고 진행): {e}")

    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": (
                    "Object.defineProperty(navigator, 'language', {get: () => 'en-US'});"
                    "Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});"
                )
            },
        )
    except Exception as e:  # noqa: BLE001
        print(f"[env_webvoyager.py] navigator.language 오버라이드 등록 실패(무시하고 진행): {e}")

    try:
        driver.execute_cdp_cmd("Emulation.setLocaleOverride", {"locale": "en-US"})
    except Exception as e:  # noqa: BLE001
        print(f"[env_webvoyager.py] Emulation.setLocaleOverride 실패(CDP 버전 이슈일 수 있음, 무시): {e}")

    try:
        driver.execute_cdp_cmd("Emulation.setTimezoneOverride", {"timezoneId": timezone_id})
    except Exception as e:  # noqa: BLE001
        print(f"[env_webvoyager.py] Emulation.setTimezoneOverride 실패(무시하고 진행): {e}")

    try:
        driver.execute_cdp_cmd("Emulation.setGeolocationOverride", dict(geolocation))
    except Exception as e:  # noqa: BLE001
        print(f"[env_webvoyager.py] Emulation.setGeolocationOverride 실패(무시하고 진행): {e}")


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
        {"action": "back"}                                      # (2026-08-11 추가) 브라우저 뒤로가기
    terminate는 execute_action으로 보내지 말 것 - agent_loop가 처리(driver에 안 보냄).
    """

    def __init__(self, window_size=DEFAULT_WINDOW_SIZE, headless=True, page_load_timeout=20, user_agent=None,
                 captcha_reset_retries=0, reuse_driver=True, manual_captcha_wait=False, wait_fn=None):
        """
        captcha_reset_retries: (2026-08-11 추가) reset() 직후 detect_bot_check()에
            걸리면 driver.get(url)을 다시 시도하는 횟수. CAPTCHA는 자동화로 못 풀지만,
            reset 시점 감지는 가끔 "아직 페이지가 다 안 뜬 상태에서의 순간적인 로딩
            인터스티셜"일 수도 있어서(Cloudflare "Just a moment..." 류) 재시도 여지를
            둔다 - 그래도 계속 감지되면 진짜 CAPTCHA로 보고 포기한다(task_info에
            "_bot_check_at_reset" 마킹, eval_webvoyager_v2.run_episode()가 이 신호를
            보고 첫 스텝도 안 밟고 바로 blocked 처리).

        reuse_driver: (2026-08-11 추가 - 태스크 간 지연 단축) 실측으로 태스크 사이 텀이
            길다는 지적이 있어서 확인해보니, 매 reset()마다 Chrome 드라이버를 통째로
            close()+재생성하고 있었다(브라우저 프로세스 재기동 자체가 태스크당 몇 초씩
            추가됨). True(기본)면 첫 reset()에서만 드라이버를 만들고 이후로는 같은
            드라이버를 재사용하며 delete_all_cookies()로 이전 태스크의 세션/로케인 쿠키만
            지운다 - CDP 로케일 오버라이드(_apply_locale_overrides)는 드라이버(브라우저
            세션) 단위 설정이라 _make_driver() 안에서 한 번만 걸면 재사용해도 계속
            유지된다(재적용 불필요). False로 주면 예전처럼 매 태스크마다 완전히 새
            브라우저를 띄운다 - 태스크 간 격리를 더 강하게 보장하고 싶을 때(예: 드라이버가
            불안정해지는 게 의심될 때) 쓸 것.

        manual_captcha_wait: (2026-08-11 추가 - 실측 요청) captcha_reset_retries를 다 써도
            여전히 bot-check가 감지되면, 자동으로 포기하는 대신 잠깐 멈춰서 사람이 직접
            (headless=False로 띄운 실제 브라우저 창에서) CAPTCHA를 풀 시간을 준다 - "내가
            직접 눌러주면 될 것 같다"는 요청 대응. wait_fn()이 반환하면(기본은 콘솔에서
            Enter 입력 대기) bot-check를 한 번 더 확인해서, 실제로 풀렸으면 정상 진행하고
            여전히 감지되면 그때는 진짜 포기(_bot_check_at_reset)한다. headless=True(기본)
            상태에서 이 옵션을 켜봤자 사람이 볼 화면이 없으니 반드시 headless=False와 같이
            써야 의미가 있다 - 이 클래스가 그 조합을 강제하진 않는다(휴리스틱으로 막으면
            테스트/CI 환경에서 오히려 방해될 수 있어서, 판단은 호출부에 맡김).
        wait_fn: manual_captcha_wait=True일 때 "사람이 다 풀 때까지 기다리는" 방법을 주입.
            기본값 None이면 builtins.input()을 써서 콘솔에서 Enter 누를 때까지 블로킹한다.
            테스트에서는 실제로 멈추면 안 되므로 즉시 반환하는 가짜 함수를 넣어서 검증한다.
        """
        self.window_size = window_size
        self.headless = headless
        self.page_load_timeout = page_load_timeout
        self.user_agent = user_agent
        self.captcha_reset_retries = captcha_reset_retries
        self.reuse_driver = reuse_driver
        self.manual_captcha_wait = manual_captcha_wait
        self.wait_fn = wait_fn or (lambda msg: input(msg))
        self.driver = None
        self.task_info = None

    # ------------------------------------------------------------------
    def reset(self, task):
        if self.driver is None:
            self.driver = _make_driver(self.window_size, headless=self.headless, user_agent=self.user_agent)
            self.driver.set_page_load_timeout(self.page_load_timeout)
        elif self.reuse_driver:
            try:
                self.driver.delete_all_cookies()
            except Exception as e:  # noqa: BLE001 - 쿠키 초기화 실패해도 나머지 흐름은 계속되어야 함
                print(f"[env_webvoyager.py] 태스크 간 쿠키 초기화 실패(무시하고 진행): {e}")
        else:
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

        # (2026-08-11 추가 - 위치/통화 강제 영미권 고정) _apply_locale_overrides()의 브라우저
        # 시그널(Accept-Language 등)만으론 IP 기반 지역판단을 못 이기는 사이트(Amazon 확인됨)에
        # 대한 2차 방어선 - 사이트가 우선적으로 신뢰하는 쿠키를 직접 심고 한 번 더 로드한다.
        self._force_site_locale(url)

        self.task_info = {"instruction": instruction, "url": url, **extra}

        # (2026-08-11 추가) reset 직후 bot-check 감지되면 captcha_reset_retries만큼 재로드
        # 시도. 다 써도 여전히 감지되면 포기하고 "_bot_check_at_reset" 마커를 남긴다 - CAPTCHA를
        # 이 코드가 풀어주지는 않는다(그건 이 프로젝트 범위 밖), 정직하게 막혔다고 보고할 뿐.
        bot_check = self.detect_bot_check()
        retries_left = self.captcha_reset_retries
        while bot_check and retries_left > 0:
            print(
                f"[env_webvoyager.py] reset 직후 bot-check 감지({bot_check['reason']}) -> "
                f"재로드 재시도(남은 횟수={retries_left})"
            )
            retries_left -= 1
            try:
                self.driver.get(url)
            except Exception as e:
                print(f"[env_webvoyager.py] 재시도 중 driver.get({url!r}) 예외(무시하고 진행): {e}")
            time.sleep(1.5)
            bot_check = self.detect_bot_check()

        # (2026-08-11 추가 - 수동 CAPTCHA 통과) 자동 재시도를 다 쓰고도 여전히 막혀 있으면,
        # manual_captcha_wait가 켜져 있는 경우에 한해 사람이 직접 풀 시간을 준다.
        if bot_check and self.manual_captcha_wait:
            print(
                f"[env_webvoyager.py] bot-check 감지됨({bot_check['reason']}) - 자동 재시도 소진. "
                "headless=False로 띄운 브라우저 창에서 직접 CAPTCHA를 풀고 나서 Enter를 눌러주세요."
            )
            self.wait_fn(
                "CAPTCHA를 다 풀었으면 Enter를 누르세요 (풀지 못했으면 그냥 Enter를 눌러도 "
                "이 태스크는 blocked로 기록됩니다): "
            )
            bot_check = self.detect_bot_check()
            if not bot_check:
                print("[env_webvoyager.py] 수동 CAPTCHA 해결 확인됨 - 정상 진행")
            else:
                print("[env_webvoyager.py] 여전히 bot-check 감지됨 - 이 태스크는 blocked로 기록")

        if bot_check:
            self.task_info["_bot_check_at_reset"] = bot_check

        return self._screenshot(), dict(self.task_info)

    # ------------------------------------------------------------------
    def detect_bot_check(self):
        """
        (2026-08-11 추가) 현재 페이지가 CAPTCHA/bot-check 화면인지 title/URL 기준으로
        저렴하게 확인하는 보수적인 휴리스틱(완벽한 탐지 아님 - _BOT_CHECK_KEYWORDS 참고).
        eval_webvoyager_v2.run_episode()가 reset 직후와 매 스텝 이후 duck-typing으로
        호출한다(구버전 env나 mock처럼 이 메서드가 없어도 호출부는 정상 동작함).

        Returns: None(정상으로 보임) 또는 {"reason": str}(감지됨)
        """
        if self.driver is None:
            return None
        try:
            title = (self.driver.title or "").lower()
            url = (self.driver.current_url or "").lower()
        except Exception:
            # driver가 죽었거나 페이지 전환 중이라 title/url을 못 읽는 경우 - bot-check
            # 여부를 판단할 수 없으니 안전하게 "모르겠다"(None)로 처리, 호출부의 stuck-repeat
            # 안전장치가 결국 잡아낼 것.
            return None
        for kw in _BOT_CHECK_KEYWORDS:
            if kw in title:
                return {"reason": f"title contains {kw!r}"}
        for kw in _BOT_CHECK_KEYWORDS:
            if kw in url:
                return {"reason": f"url contains {kw!r}"}
        return None

    # ------------------------------------------------------------------
    def wait_for_manual_captcha(self) -> bool:
        """
        (2026-08-11 추가 - 수동 CAPTCHA 통과) reset() 시점의 재시도 로직과 별개로, 에피소드가
        이미 진행 중인 상태(스텝 실행 후)에서 bot-check가 뜬 경우를 위한 버전.
        eval_webvoyager_v2.run_episode()가 매 스텝 후 detect_bot_check()로 뭔가 감지하면
        이 메서드를 호출한다(duck-typing - 이 메서드가 없는 구버전 env/mock이면 그냥
        건너뛰고 기존처럼 즉시 blocked 처리됨).

        manual_captcha_wait=False(기본)면 아무것도 안 하고 바로 False를 반환한다 - 호출부는
        이걸 "사람이 개입 안 했다"는 뜻으로 받아서 기존처럼 즉시 blocked 처리하면 된다.

        Returns: bool - 사람이 직접 풀어서 실제로 통과됐으면 True(호출부는 blocked 처리하지
            않고 계속 진행), 옵션이 꺼져 있거나 풀리지 않았으면 False(기존과 동일하게 blocked).
        """
        if not self.manual_captcha_wait:
            return False
        bot_check = self.detect_bot_check()
        if not bot_check:
            return True
        print(
            f"[env_webvoyager.py] 스텝 진행 중 bot-check 감지됨({bot_check['reason']}) - "
            "headless=False 브라우저 창에서 직접 풀고 나서 Enter를 눌러주세요."
        )
        self.wait_fn(
            "CAPTCHA를 다 풀었으면 Enter를 누르세요 (풀지 못했으면 그냥 Enter를 눌러도 "
            "이 태스크는 blocked로 기록됩니다): "
        )
        still_blocked = self.detect_bot_check()
        if still_blocked:
            print("[env_webvoyager.py] 여전히 bot-check 감지됨 - 이 태스크는 blocked로 기록")
            return False
        print("[env_webvoyager.py] 수동 CAPTCHA 해결 확인됨 - 정상 진행")
        return True

    # ------------------------------------------------------------------
    def _force_site_locale(self, url):
        """
        (2026-08-11 추가 - 위치/통화 강제 영미권 고정) _apply_locale_overrides()가 건 Accept-
        Language/navigator.language/타임존/위치 오버라이드는 전부 브라우저 쪽 시그널이라,
        서버가 접속 IP로 지역/통화를 판단하면 못 이긴다 - 실측으로 Amazon이 이 케이스임을
        확인했다(한국 IP에서 접속하면 저 오버라이드를 다 걸어도 여전히 원화(KRW)로 가격이
        표시됨. WebVoyager 결과 jsonl에서 "50~75달러 사이" 태스크의 final_answer가 "KRW
        28,343"으로 나와서 judge가 달러 조건과 대조를 못 하고 오탐한 사례로 발견).

        Amazon은 i18n-prefs(통화)/lc-main(언어) 쿠키를 IP 기반 추정보다 우선적으로 읽는
        것으로 확인되어, 해당 쿠키를 직접 심고 페이지를 한 번 더 로드해서 반영시킨다(쿠키는
        새로고침 전까지는 적용 안 됨). _SITE_LOCALE_COOKIES에 등록된 사이트만 대상 - 지금은
        Amazon만 확인된 문제라 그것만 있고, 다른 WebVoyager 사이트에서 같은 증상이 확인되면
        거기 추가하면 이 함수는 그대로 재사용된다.

        driver.add_cookie()는 Selenium 특성상 "현재 그 도메인의 페이지에 있어야" 동작해서,
        reset()이 최초 driver.get(url) + sleep을 마친 직후(= 이미 해당 도메인 페이지에 있는
        상태)에만 호출해야 한다 - 호출 순서를 reset()이 보장한다.
        """
        try:
            current = (self.driver.current_url or url or "").lower()
        except Exception:
            current = (url or "").lower()

        for site_kw, cookies in _SITE_LOCALE_COOKIES.items():
            if site_kw not in current:
                continue
            try:
                for cookie in cookies:
                    self.driver.add_cookie(dict(cookie))
                self.driver.get(url)
                time.sleep(1.0)
            except Exception as e:  # noqa: BLE001 - 쿠키 보정 실패해도 나머지 흐름은 계속되어야 함
                print(f"[env_webvoyager.py] {site_kw!r} locale 쿠키 강제 실패(무시하고 진행): {e}")
            break

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
            "back": self._back,
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

    def _back(self, action):
        # (2026-08-11 추가) 브라우저 히스토리 뒤로가기. 지금까지 액션 스키마에 없어서, 잘못된
        # 페이지로 들어갔을 때 화면 안에서 뒤로가기 버튼/로고를 못 찾으면 그 자리에서 뺑뺑이
        # 돌 수밖에 없었다(뺑뺑이 조기종료/경고 로직은 이걸 감지는 하지만 대안을 주진 못했다).
        # driver.back()은 CDP가 아니라 Selenium의 표준 WebDriver 히스토리 내비게이션 - 브라우저
        # 크롬(주소창) 레벨 동작이라 CDP Input 이벤트로는 못 흉내내던 것.
        self.driver.back()
        time.sleep(0.5)  # 페이지 전환 렌더 여유(다른 액션들과 달리 즉시 스크린샷을 찍으면
        # 이전 페이지가 찍힐 수 있어서 짧게 대기)

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

    # (2026-08-11 추가) back -> driver.back() 호출됨(CDP 아닌 Selenium 표준 히스토리 내비게이션)
    orig_sleep_back = time.sleep
    time.sleep = lambda *a, **k: None
    try:
        env.execute_action({"action": "back"})
        check("back -> driver.back() 호출됨", env.driver.back.called)
    finally:
        time.sleep = orig_sleep_back

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

    # --- (2026-08-11 추가) detect_bot_check() ---
    env.driver.title = "Just a moment..."
    env.driver.current_url = "http://example.com/"
    r = env.detect_bot_check()
    check("detect_bot_check -> title 매칭시 감지됨", r is not None and "just a moment" in r["reason"])

    env.driver.title = "Example Domain"
    env.driver.current_url = "http://example.com/recaptcha/challenge"
    r2 = env.detect_bot_check()
    # "recaptcha"는 "captcha"를 부분문자열로 포함하므로("captcha"가 키워드 목록에서 먼저
    # 매치될 수 있음) 정확히 어느 키워드로 잡혔는지보다 "url 쪽에서 잡혔다"는 것만 확인.
    check("detect_bot_check -> url 매칭도 감지됨", r2 is not None and "url contains" in r2["reason"] and "captcha" in r2["reason"])

    env.driver.title = "Example Domain"
    env.driver.current_url = "http://example.com/"
    r3 = env.detect_bot_check()
    check("detect_bot_check -> 정상 페이지는 None", r3 is None)

    env_no_driver = WebVoyagerEnv.__new__(WebVoyagerEnv)
    env_no_driver.driver = None
    check("detect_bot_check -> driver 없으면 예외 없이 None", env_no_driver.detect_bot_check() is None)

    # task 파싱 (dict / tuple 둘 다)
    dummy = WebVoyagerEnv.__new__(WebVoyagerEnv)
    url, instr, extra = dummy._parse_task({"web": "http://x.com", "ques": "do X", "web_name": "X", "id": "X--0"})
    check("task dict 파싱", url == "http://x.com" and instr == "do X" and extra == {"web_name": "X", "id": "X--0"})
    url2, instr2, extra2 = dummy._parse_task(("http://y.com", "do Y"))
    check("task tuple 파싱", url2 == "http://y.com" and instr2 == "do Y" and extra2 == {})

    # --- (2026-08-11 추가) _apply_locale_overrides(): CDP 호출 배선 + 개별 실패 시 예외 안 남 ---
    locale_driver = MagicMock()
    _apply_locale_overrides(locale_driver)
    cdp_calls = {c.args[0]: c.args[1] for c in locale_driver.execute_cdp_cmd.call_args_list}
    check(
        "_apply_locale_overrides -> Accept-Language 헤더 강제",
        cdp_calls.get("Network.setExtraHTTPHeaders", {}).get("headers", {}).get("Accept-Language")
        == DEFAULT_ACCEPT_LANGUAGE,
    )
    check(
        "_apply_locale_overrides -> navigator.language/languages 오버라이드 스크립트 등록",
        any(
            c.args[0] == "Page.addScriptToEvaluateOnNewDocument"
            and "navigator, 'language'" in c.args[1]["source"]
            and "navigator, 'languages'" in c.args[1]["source"]
            for c in locale_driver.execute_cdp_cmd.call_args_list
        ),
    )
    check(
        "_apply_locale_overrides -> Emulation.setLocaleOverride(en-US)",
        cdp_calls.get("Emulation.setLocaleOverride", {}).get("locale") == "en-US",
    )
    check(
        "_apply_locale_overrides -> Emulation.setTimezoneOverride",
        cdp_calls.get("Emulation.setTimezoneOverride", {}).get("timezoneId") == DEFAULT_TIMEZONE_ID,
    )
    check(
        "_apply_locale_overrides -> Emulation.setGeolocationOverride(뉴욕 좌표)",
        cdp_calls.get("Emulation.setGeolocationOverride") == DEFAULT_GEOLOCATION,
    )

    # 일부 CDP 명령이 구버전이라 실패해도(예: Emulation.setLocaleOverride 미지원) 나머지는
    # 계속 적용되고 예외가 밖으로 안 새야 함(navigator.webdriver 우회와 같은 원칙).
    flaky_driver = MagicMock()
    def _flaky_execute_cdp_cmd(cmd, params):
        if cmd == "Emulation.setLocaleOverride":
            raise Exception("이 Chrome 버전엔 없는 CDP 메서드")
        return MagicMock()
    flaky_driver.execute_cdp_cmd.side_effect = _flaky_execute_cdp_cmd
    try:
        _apply_locale_overrides(flaky_driver)
        check("_apply_locale_overrides -> 일부 CDP 명령 실패해도 예외 안 남", True)
    except Exception:
        check("_apply_locale_overrides -> 일부 CDP 명령 실패해도 예외 안 남", False)
    check(
        "_apply_locale_overrides -> 실패한 명령 이후({}) 명령도 계속 시도됨".format("Emulation.setLocaleOverride"),
        any(c.args[0] == "Emulation.setTimezoneOverride" for c in flaky_driver.execute_cdp_cmd.call_args_list),
    )

    # --- (2026-08-11 추가) _force_site_locale(): Amazon 쿠키 강제 + 비-Amazon 사이트는 no-op ---
    amazon_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    amazon_env.driver = MagicMock()
    amazon_env.driver.current_url = "https://www.amazon.com/s?k=xbox+controller"
    amazon_env._force_site_locale("https://www.amazon.com/s?k=xbox+controller")
    cookie_calls = [c.args[0] for c in amazon_env.driver.add_cookie.call_args_list]
    check(
        "_force_site_locale -> amazon.com에서 i18n-prefs/lc-main 쿠키 심음",
        {"name": "i18n-prefs", "value": "USD"} in cookie_calls
        and {"name": "lc-main", "value": "en_US"} in cookie_calls,
    )
    check(
        "_force_site_locale -> 쿠키 반영 위해 같은 url로 재로드",
        amazon_env.driver.get.call_args_list[-1].args == ("https://www.amazon.com/s?k=xbox+controller",),
    )

    non_amazon_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    non_amazon_env.driver = MagicMock()
    non_amazon_env.driver.current_url = "https://en.wikipedia.org/wiki/Python"
    non_amazon_env._force_site_locale("https://en.wikipedia.org/wiki/Python")
    check(
        "_force_site_locale -> 등록 안 된 사이트는 쿠키/재로드 둘 다 안 함(no-op)",
        not non_amazon_env.driver.add_cookie.called and not non_amazon_env.driver.get.called,
    )

    # 쿠키 심기 자체가 실패해도(예: 도메인 불일치로 InvalidCookieDomainException) 예외가 밖으로
    # 새면 안 됨 - reset() 흐름 전체를 죽이면 안 되는 부수적인 보정 로직이라서.
    failing_cookie_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    failing_cookie_env.driver = MagicMock()
    failing_cookie_env.driver.current_url = "https://www.amazon.com/"
    failing_cookie_env.driver.add_cookie.side_effect = Exception("InvalidCookieDomainException")
    try:
        failing_cookie_env._force_site_locale("https://www.amazon.com/")
        check("_force_site_locale -> 쿠키 심기 실패해도 예외 안 남", True)
    except Exception:
        check("_force_site_locale -> 쿠키 심기 실패해도 예외 안 남", False)

    # --- (2026-08-11 추가 - 태스크 간 지연 단축) reset()의 드라이버 재사용 로직 ---
    # 실측 지적: 태스크 사이 텀이 길다 -> 원인은 매 reset()마다 Chrome을 통째로 재기동하던
    # 것. reuse_driver=True(기본)면 기존 드라이버를 유지하고 쿠키만 지우는지, False면 예전
    # 처럼 close()+새 드라이버로 교체하는지 확인.
    import sys

    orig_sleep = time.sleep
    time.sleep = lambda *a, **k: None  # reset() 안의 고정 sleep들 때문에 테스트가 느려지는 것 방지
    try:
        reuse_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        reuse_env.window_size = (1280, 800)
        reuse_env.headless = True
        reuse_env.page_load_timeout = 20
        reuse_env.user_agent = None
        reuse_env.captcha_reset_retries = 0
        reuse_env.reuse_driver = True
        first_driver = MagicMock()
        first_driver.title = "Example Domain"
        first_driver.current_url = "http://example.com/"
        first_driver.get_screenshot_as_png.return_value = _fake_png_bytes()
        reuse_env.driver = first_driver
        reuse_env.task_info = None

        reuse_env.reset({"web": "http://example.com/a", "ques": "do A"})
        check("reuse_driver=True -> 기존 드라이버 객체를 그대로 재사용(재생성 안 함)", reuse_env.driver is first_driver)
        check("reuse_driver=True -> delete_all_cookies 호출됨", first_driver.delete_all_cookies.called)
        check("reuse_driver=True -> 기존 드라이버 quit() 호출 안 됨", not first_driver.quit.called)

        # reuse_driver=False -> 예전처럼 매번 close()+새 드라이버로 교체돼야 함
        no_reuse_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        no_reuse_env.window_size = (1280, 800)
        no_reuse_env.headless = True
        no_reuse_env.page_load_timeout = 20
        no_reuse_env.user_agent = None
        no_reuse_env.captcha_reset_retries = 0
        no_reuse_env.reuse_driver = False
        old_driver = MagicMock()
        old_driver.title = "Example Domain"
        old_driver.current_url = "http://example.com/"
        no_reuse_env.driver = old_driver
        no_reuse_env.task_info = None

        new_driver = MagicMock()
        new_driver.title = "Example Domain"
        new_driver.current_url = "http://example.com/"
        new_driver.get_screenshot_as_png.return_value = _fake_png_bytes()

        this_module = sys.modules[__name__]
        orig_make_driver = this_module._make_driver
        this_module._make_driver = lambda *a, **k: new_driver
        try:
            no_reuse_env.reset({"web": "http://example.com/b", "ques": "do B"})
        finally:
            this_module._make_driver = orig_make_driver

        check("reuse_driver=False -> 기존 드라이버 quit() 호출됨(close)", old_driver.quit.called)
        check("reuse_driver=False -> 새 드라이버 객체로 교체됨", no_reuse_env.driver is new_driver)
        check(
            "reuse_driver=False -> 새 드라이버라 delete_all_cookies는 호출 안 함",
            not new_driver.delete_all_cookies.called,
        )

        # driver가 아예 None인 최초 1회는 reuse_driver 값과 무관하게 새로 만들어져야 함
        first_reset_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        first_reset_env.window_size = (1280, 800)
        first_reset_env.headless = True
        first_reset_env.page_load_timeout = 20
        first_reset_env.user_agent = None
        first_reset_env.captcha_reset_retries = 0
        first_reset_env.reuse_driver = True
        first_reset_env.driver = None
        first_reset_env.task_info = None

        fresh_driver = MagicMock()
        fresh_driver.title = "Example Domain"
        fresh_driver.current_url = "http://example.com/"
        fresh_driver.get_screenshot_as_png.return_value = _fake_png_bytes()
        this_module._make_driver = lambda *a, **k: fresh_driver
        try:
            first_reset_env.reset({"web": "http://example.com/c", "ques": "do C"})
        finally:
            this_module._make_driver = orig_make_driver
        check("driver=None인 최초 reset -> _make_driver로 새로 생성됨", first_reset_env.driver is fresh_driver)
        check(
            "driver=None인 최초 reset -> delete_all_cookies는 호출 안 함(재사용할 기존 드라이버가 없음)",
            not fresh_driver.delete_all_cookies.called,
        )

        # --- (2026-08-11 추가 - 수동 CAPTCHA 통과) reset()의 manual_captcha_wait ---
        def _make_captcha_env(manual_captcha_wait, detect_responses, wait_fn):
            env_c = WebVoyagerEnv.__new__(WebVoyagerEnv)
            env_c.window_size = (1280, 800)
            env_c.headless = True
            env_c.page_load_timeout = 20
            env_c.user_agent = None
            env_c.captcha_reset_retries = 0  # 재시도 없이 바로 manual 분기로 가도록
            env_c.reuse_driver = True
            env_c.manual_captcha_wait = manual_captcha_wait
            env_c.wait_fn = wait_fn
            d = MagicMock()
            d.title = "Example Domain"
            d.current_url = "http://example.com/"
            d.get_screenshot_as_png.return_value = _fake_png_bytes()
            env_c.driver = d
            env_c.task_info = None
            responses = list(detect_responses)
            env_c.detect_bot_check = lambda: (responses.pop(0) if responses else None)
            return env_c

        # (a) manual_captcha_wait=True, wait 이후 실제로 풀림 -> _bot_check_at_reset 안 남음
        wait_calls_a = []
        env_a = _make_captcha_env(
            True, [{"reason": "title contains 'captcha'"}, None], lambda msg: wait_calls_a.append(msg)
        )
        _, task_info_a = env_a.reset({"web": "http://example.com/x", "ques": "do X"})
        check("manual_captcha_wait -> 사람이 풀면 wait_fn이 호출됨", len(wait_calls_a) == 1)
        check("manual_captcha_wait -> 풀린 경우 _bot_check_at_reset 안 남음", "_bot_check_at_reset" not in task_info_a)

        # (b) manual_captcha_wait=True인데 wait 이후에도 여전히 감지됨 -> 결국 blocked 마킹
        wait_calls_b = []
        env_b = _make_captcha_env(
            True,
            [{"reason": "title contains 'captcha'"}, {"reason": "title contains 'captcha'"}],
            lambda msg: wait_calls_b.append(msg),
        )
        _, task_info_b = env_b.reset({"web": "http://example.com/y", "ques": "do Y"})
        check("manual_captcha_wait -> 여전히 안 풀리면 wait_fn은 호출됐지만", len(wait_calls_b) == 1)
        check("manual_captcha_wait -> 여전히 안 풀리면 결국 _bot_check_at_reset 남음", "_bot_check_at_reset" in task_info_b)

        # (c) manual_captcha_wait=False(기본) -> wait_fn 호출 안 되고 예전처럼 즉시 blocked
        wait_calls_c = []
        env_c = _make_captcha_env(
            False, [{"reason": "title contains 'captcha'"}], lambda msg: wait_calls_c.append(msg)
        )
        _, task_info_c = env_c.reset({"web": "http://example.com/z", "ques": "do Z"})
        check("manual_captcha_wait=False -> wait_fn 호출 안 됨(하위 호환)", len(wait_calls_c) == 0)
        check("manual_captcha_wait=False -> 예전과 동일하게 즉시 _bot_check_at_reset 남음", "_bot_check_at_reset" in task_info_c)

        # --- wait_for_manual_captcha() 단독 테스트 (스텝 진행 중 감지 케이스) ---
        wait_calls_d = []
        env_d = _make_captcha_env(True, [{"reason": "url contains 'recaptcha'"}, None], lambda msg: wait_calls_d.append(msg))
        resolved = env_d.wait_for_manual_captcha()
        check("wait_for_manual_captcha -> 풀리면 True 반환", resolved is True)
        check("wait_for_manual_captcha -> wait_fn 호출됨", len(wait_calls_d) == 1)

        wait_calls_e = []
        env_e = _make_captcha_env(
            True, [{"reason": "url contains 'recaptcha'"}, {"reason": "url contains 'recaptcha'"}],
            lambda msg: wait_calls_e.append(msg),
        )
        resolved_e = env_e.wait_for_manual_captcha()
        check("wait_for_manual_captcha -> 안 풀리면 False 반환", resolved_e is False)

        env_f = _make_captcha_env(False, [{"reason": "url contains 'recaptcha'"}], lambda msg: None)
        check("wait_for_manual_captcha -> manual_captcha_wait=False면 즉시 False(감지 확인조차 안 함)", env_f.wait_for_manual_captcha() is False)

        env_g = _make_captcha_env(True, [None], lambda msg: None)
        check("wait_for_manual_captcha -> 애초에 bot-check가 없으면 바로 True", env_g.wait_for_manual_captcha() is True)
    finally:
        time.sleep = orig_sleep

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
    ap.add_argument(
        "--out_image", default=None,
        help="reset 스크린샷 저장 경로. 미지정시 실행 위치(cwd) 기준 './webvoyager_reset.png' - "
        "(2026-08-10 수정 x2) 처음엔 '/tmp/...'로 하드코딩돼 있어서 Windows에서 FileNotFoundError가 "
        "났었고(실측), tempfile.gettempdir()로 바꿨더니 이번엔 이 값이 OS 기본 TEMP가 아니라 사용자 "
        "PC에 깔린 ESTsoft(알집/반디집류) 프로그램이 TEMP 환경변수를 자기 폴더로 덮어써놓은 값이 "
        "그대로 나와서 경로가 이상하게 나왔다(실측: C:\\Users\\Public\\...\\ESTsoft\\CreatorTemp\\...) - "
        "예측 불가능한 시스템 환경변수에 기대는 대신, 그냥 이 스크립트를 실행한 위치 기준으로 저장하게 "
        "바꿔서 항상 어디에 저장됐는지 바로 알 수 있게 함.",
    )
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

        out_image = args.out_image or os.path.join(os.getcwd(), "webvoyager_reset.png")
        screenshot.save(out_image)
        print(f"[env_webvoyager.py] reset 완료. screenshot 저장: {out_image}")
        env.close()
        env.close()