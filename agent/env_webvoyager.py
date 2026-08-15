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
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
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
    # (2026-08 추가 - Apple 한국어/원화 렌더링 문제) 실측(devtools Application 탭)으로 확인:
    # apple.com이 "geo" 쿠키(값 "KR")로 스토어프론트 지역을 판단하고 있었고, 이 값을 "US"로
    # 바꾸고 새로고침하니 가격/언어가 실제로 USD/영어로 바뀌는 것까지 사용자가 직접 확인함
    # (Amazon의 i18n-prefs/lc-main과 동일한 성격의 단일 지역 판단 쿠키).
    "apple": (
        {"name": "geo", "value": "US"},
    ),
}

# (2026-08 추가 - Google Search/Maps 한국어 렌더링 문제) 실측: Google Search 홈/결과 화면과
# Google Maps가 한국 IP 기준으로 한국어 UI를 보여주는 게 확인됨(Google 검색 버튼이 "Google
# 검색"으로 뜨는 등). Google은 Amazon과 달리 언어/지역 판단에 쿠키보다 URL 쿼리 파라미터
# (hl=인터페이스 언어, gl=지역/국가)를 우선적으로 신뢰하는 것으로 알려져 있어서, 쿠키 대신
# 이 방식으로 사이트별 강제 로케일을 처리한다. 값은 dict(파라미터명 -> 값) - 기존 쿼리에
# hl/gl이 이미 있으면 덮어쓰고, 없으면 추가한다. _force_site_locale()이 _SITE_LOCALE_COOKIES
# 처리 후 이쪽도 확인한다.
_SITE_LOCALE_URL_PARAMS = {
    "google.com": {"hl": "en", "gl": "us"},
    # (2026-08 추가 - Booking 원화 렌더링 문제) 실측(사용자 직접 확인): URL에
    # &selected_currency=USD를 붙이면 그 페이지는 USD로 뜨는데, Google의 hl/gl과 달리
    # 세션에 저장되지 않아서 링크를 클릭해 다른 페이지로 이동하면(새 URL이라 파라미터가
    # 없으니) 다시 원화(KRW)/한국어로 되돌아가는 것까지 확인됨 - 즉 Booking은 "한 번
    # 고치면 계속 유지"가 아니라 "매 페이지 이동마다 다시 붙여줘야" 하는 사이트. 이 문제
    # 때문에 execute_action()에 매 액션 후 URL이 바뀌었으면 이 파라미터를 다시 강제하는
    # 로직을 추가했다(_reapply_url_locale_params_if_navigated 참고).
    "booking.com": {"selected_currency": "USD"},
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
        {"action": "left_click_drag", "start_coordinate": [x1, y1], "coordinate": [x2, y2]}  # (2026-08 추가)
    terminate는 execute_action으로 보내지 말 것 - agent_loop가 처리(driver에 안 보냄).

    [새 탭 자동 전환 - 2026-08-14 추가] 광고 팝업이 아니라 사이트 자체가 링크를 target=_blank로
    열어서 실제 콘텐츠가 새 탭에 뜨는 경우가 있다(Booking 실측). Selenium은 새 탭이 열려도
    포커스를 자동으로 안 옮기고 get_screenshot_as_png()는 항상 현재 포커스된 탭만 찍기 때문에,
    가만히 두면 에이전트가 계속 예전 탭(내용 그대로)만 보고 "페이지가 안 뜬다"고 오판한다.
    execute_action()이 매 액션 후 새 탭이 열렸는지 확인해서, 열렸으면 그리로 스위치하고 이전
    탭은 닫아 "항상 탭 1개만 유지"한다(_switch_to_new_tab_if_opened 참고).
    """

    def __init__(self, window_size=DEFAULT_WINDOW_SIZE, headless=True, page_load_timeout=20, user_agent=None,
                 captcha_reset_retries=0, reuse_driver=True, manual_captcha_wait=False, wait_fn=None,
                 auto_switch_new_tab=True):
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
        auto_switch_new_tab: (2026-08-15 추가 - ESPN 등 광고 리다이렉트 대응) True(기본)면
            새 탭이 열렸을 때 자동으로 그 탭으로 포커스를 옮기고 이전 탭들은 닫는다
            (_switch_to_new_tab_if_opened 참고 - Booking처럼 사이트가 target=_blank로 실제
            콘텐츠를 새 탭에 여는 경우를 위해 추가된 동작). 그런데 ESPN 등 일부 사이트는 광고가
            새 탭으로 리다이렉트되는 경우가 있어서, 이 정책을 그대로 적용하면 광고 탭으로
            옮겨가버리는 역효과가 남 - False로 주면 새 탭 감지/전환 로직 자체를 완전히 건너뛰고
            기존 탭에 포커스를 유지한다(이 기능이 추가되기 전의 동작과 동일 - 새 탭은 열린
            채로 방치되고 무시됨).
        """
        self.window_size = window_size
        self.headless = headless
        self.page_load_timeout = page_load_timeout
        self.user_agent = user_agent
        self.captcha_reset_retries = captcha_reset_retries
        self.reuse_driver = reuse_driver
        self.manual_captcha_wait = manual_captcha_wait
        self.wait_fn = wait_fn or (lambda msg: input(msg))
        self.auto_switch_new_tab = auto_switch_new_tab
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

        (2026-08-15 추가 - 구글 검색 실측) 구글의 "Our systems have detected unusual
        traffic..." 차단 페이지는 title/URL 어디에도 _BOT_CHECK_KEYWORDS 문구가 안 남는다
        (URL은 google.com/sorry/index... 형태, title은 보통 그대로거나 비어있음) - 실제
        문구는 페이지 본문(body)에만 있어서 title/URL만 보는 기존 로직은 이걸 통과시켰다.
        그래서 title/URL이 둘 다 깨끗해도, 마지막으로 body innerText 앞부분(전체를 다 읽으면
        느려질 수 있는 긴 페이지 대비 8000자로 제한)까지 한 번 더 검사한다. body 텍스트
        조회 자체가 실패해도(페이지 전환 중 등) title/URL 검사 결과는 그대로 유지한다.

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
        try:
            body_text = (
                self.driver.execute_script(
                    "return document.body ? document.body.innerText.slice(0, 8000) : ''"
                )
                or ""
            ).lower()
        except Exception:
            # body 텍스트 조회 실패해도 title/URL 기준 판단(정상)은 그대로 유효 - 조용히 포기.
            return None
        for kw in _BOT_CHECK_KEYWORDS:
            if kw in body_text:
                return {"reason": f"body contains {kw!r}"}
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

        # (2026-08 추가 - Google Search/Maps, Booking) 쿠키 대신 URL 쿼리 파라미터로 로케일을
        # 강제하는 사이트 - _SITE_LOCALE_COOKIES와 별개 매핑이라 독립적으로 순회한다(한 url이
        # 이론상 둘 다에 해당할 수도 있어 break 없이 각자 처리).
        self._apply_url_locale_params(url, current)

    def _apply_url_locale_params(self, url, current_lower=None):
        """
        (2026-08 추가 - _force_site_locale에서 분리) _SITE_LOCALE_URL_PARAMS에 등록된 사이트면
        url에 그 쿼리 파라미터를 강제로 덮어써서 재로드한다. reset() 시점(_force_site_locale)과
        매 액션 후 재적용(_reapply_url_locale_params_if_navigated) 둘 다에서 재사용하기 위해
        별도 메서드로 뺐다 - Google은 한 번만 걸어도 세션에 유지되는 것으로 보이지만, Booking은
        URL 파라미터가 그 페이지 렌더링에만 적용되고 세션에 저장되지 않아(실측 확인: 링크 클릭
        -> 새 URL로 이동하면 다시 원화/한국어로 돌아감) 매 이동마다 다시 걸어줘야 한다.

        반환값: 실제로 매칭돼서 재로드를 시도했으면 True, 매칭되는 사이트가 없었으면 False.
        """
        if current_lower is None:
            try:
                current_lower = (self.driver.current_url or url or "").lower()
            except Exception:
                current_lower = (url or "").lower()

        for site_kw, params in _SITE_LOCALE_URL_PARAMS.items():
            if site_kw not in current_lower:
                continue
            try:
                parsed = urlsplit(url)
                query = dict(parse_qsl(parsed.query))
                query.update(params)
                new_url = urlunsplit(parsed._replace(query=urlencode(query)))
                self.driver.get(new_url)
                time.sleep(1.0)
            except Exception as e:  # noqa: BLE001 - 로케일 파라미터 보정 실패해도 나머지 흐름은 계속되어야 함
                print(f"[env_webvoyager.py] {site_kw!r} locale URL 파라미터 강제 실패(무시하고 진행): {e}")
            return True
        return False

    def _reapply_url_locale_params_if_navigated(self, prev_url):
        """
        (2026-08 추가 - Booking 등 세션에 유지 안 되는 사이트 대응) execute_action()이 매 액션
        직후 호출한다. 액션 전/후 URL을 비교해서 실제로 다른 페이지로 이동했을 때만(클릭이
        모달/드롭다운만 열고 URL이 그대로인 경우는 건드리지 않음) _apply_url_locale_params()를
        다시 태운다. 매 액션마다 무조건 재로드하면 느려지고 진행 중인 입력/상태를 날릴 수
        있어서, "URL이 실제로 바뀐 경우"로 조건을 좁혔다.

        (2026-08-14 수정 - 레이스 컨디션 버그 픽스) 실측(Booking 2태스크 런, debug 덤프 확인):
        _click() 등은 CDP 이벤트만 쏘고 리턴하지, 브라우저 자체 네비게이션이 끝나길 기다리지
        않는다. Search 버튼 클릭처럼 진짜 페이지 이동을 유발하는 액션 직후에 곧바로
        driver.current_url을 읽어서 거기다 파라미터만 얹어 driver.get()으로 재로드하면, 그
        시점의 URL이 아직 브라우저가 스스로 이동시키는 중인 "중간/미확정 상태"일 수 있다 -
        이러면 우리가 쏘는 수동 재로드가 브라우저 자체 네비게이션과 경합(race)해서 그걸
        가로채버리고, 결과적으로 검색폼이 초기화되고(destination 빈칸, 날짜 기본값으로
        리셋) 검색 결과 페이지에 끝내 도달하지 못한 채 같은 입력을 무한 반복하는 게 확인됨
        (Booking--0/1 둘 다 성공률 0%, "Enter a destination to start searching" 경고와 함께
        폼이 리셋된 스크린샷으로 확증). 등록된 사이트(_SITE_LOCALE_URL_PARAMS)로 이동한
        경우에 한해서만, driver.current_url이 연속으로 안정될 때까지 짧게 폴링한 뒤(최대
        2초, _wait_for_url_settle 참고) 그 최종 URL로 비교/재적용한다 - 등록 안 된 사이트는
        기존처럼 폴링 없이 즉시 1회 비교(속도 영향 없음).
        """
        prev_lower = (prev_url or "").lower()
        if not any(site_kw in prev_lower for site_kw in _SITE_LOCALE_URL_PARAMS):
            # 로케일 파라미터를 강제하는 사이트가 아니면 굳이 안정화를 기다릴 필요 없음 -
            # 기존 동작 그대로 즉시 1회 비교.
            try:
                new_url = self.driver.current_url or ""
            except Exception:
                return
            if not new_url or new_url == prev_url:
                return
            self._apply_url_locale_params(new_url)
            return

        new_url = self._wait_for_url_settle(prev_url)
        if not new_url or new_url == prev_url:
            return
        self._apply_url_locale_params(new_url)

    def _wait_for_url_settle(self, prev_url, timeout=2.0, interval=0.25):
        """
        (2026-08-14 추가 - 레이스 컨디션 버그 픽스) driver.current_url을 interval초 간격으로
        폴링하다가, 연속 두 번의 읽기 값이 같으면(=브라우저 자체 네비게이션이 끝나서 URL이
        더 이상 안 바뀜) 그 값을 안정된 최종 URL로 보고 반환한다. timeout초 안에 안정되지
        않아도 무한 대기하지 않고 마지막으로 읽은 값을 그냥 반환한다(아주 느린 페이지에서
        영원히 블로킹되면 execute_action() 전체가 멈추니까).
        """
        try:
            last = self.driver.current_url or ""
        except Exception:
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(interval)
            try:
                cur = self.driver.current_url or ""
            except Exception:
                return last
            if cur == last:
                return cur
            last = cur
        return last

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

        # (2026-08 추가 - Booking 등 URL 파라미터 로케일이 세션에 안 남는 사이트 대응) 액션
        # 실행 전 URL을 기록해뒀다가, 액션 후 실제로 다른 페이지로 이동했으면 로케일 파라미터를
        # 다시 강제한다(_reapply_url_locale_params_if_navigated 참고). try/except로 감싸서
        # current_url 조회 자체가 실패해도(드라이버 초기 상태 등) 액션 실행 자체는 막지 않는다.
        try:
            prev_url = self.driver.current_url
        except Exception:
            prev_url = None

        # (2026-08-14 추가 - 새 탭 대응) 새 탭이 열리는지 감지하려면 액션 전 탭 목록을 알아야
        # 하므로 fn(action) 전에 먼저 읽어둔다(_switch_to_new_tab_if_opened 참고).
        try:
            prev_handles = list(self.driver.window_handles)
        except Exception:
            prev_handles = []

        fn(action)

        # (2026-08-14 추가 - 광고 팝업이 아니라 사이트 자체가 target=_blank로 실제 콘텐츠를 새
        # 탭에 여는 경우 대응, Booking 실측) get_screenshot_as_png()는 항상 "현재 포커스된
        # 탭"만 찍는데 Selenium은 새 탭이 열려도 포커스를 자동으로 안 옮겨서, 에이전트가
        # 계속 예전 탭(내용 그대로인 화면)만 보고 "페이지가 안 뜬다"고 오판하는 문제가
        # 확인됨 - fn(action) 직후, 로케일 재적용보다 먼저 처리해서 이후 스크린샷/URL
        # 비교가 전부 새 탭 기준으로 이뤄지게 한다.
        if self.auto_switch_new_tab:
            try:
                self._switch_to_new_tab_if_opened(prev_handles)
            except Exception as e:  # noqa: BLE001 - 탭 전환 실패해도 액션 자체 결과 반환은 막지 않음
                print(f"[env_webvoyager.py] 새 탭 전환 처리 중 예외(무시하고 진행): {e}")

        if prev_url is not None:
            self._reapply_url_locale_params_if_navigated(prev_url)

        # (2026-08-15 추가 - planner 스크린샷이 간혹 흰 화면으로 찍히는 문제 대응) 실측: 클릭이
        # 실제 페이지 네비게이션을 유발하는 경우, fn(action)이 CDP 이벤트만 쏘고 바로 리턴하기
        # 때문에(대기 없음) 곧바로 _screenshot()을 찍으면 새 페이지가 아직 렌더링되기 전(흰
        # 화면/로딩 중)일 수 있다 - 로케일 강제 대상 사이트(Booking/Google)에만 걸려있던
        # "네비게이션 끝날 때까지 대기" 로직을 모든 액션/사이트로 일반화한다. document.readyState
        # 가 이미 'complete'면(대부분의 클릭/타이핑/스크롤처럼 네비게이션이 없는 액션) 첫 확인에서
        # 바로 통과하므로 추가 지연이 거의 없고, 실제 페이지 이동이 있었을 때만 짧게 기다린다.
        try:
            self._wait_for_page_ready()
        except Exception as e:  # noqa: BLE001 - 대기 로직 실패해도 스크린샷/반환 자체는 막지 않음
            print(f"[env_webvoyager.py] 페이지 로딩 대기 중 예외(무시하고 진행): {e}")

        # reward/terminated는 이 wrapper의 책임이 아님(파일 상단 docstring 참고) - 항상 None/False.
        return self._screenshot(), None, False, False, dict(self.task_info)

    # ------------------------------------------------------------------
    def _wait_for_page_ready(self, timeout=1.5, interval=0.15):
        """
        (2026-08-15 추가 - 흰 화면 스크린샷 버그 픽스) document.readyState가 'complete'가 될
        때까지 최대 timeout초 짧게 폴링한다. 이미 'complete'면(네비게이션이 없었던 액션) 첫
        확인에서 바로 리턴해서 지연이 거의 없다. readyState 조회 자체가 실패하면(예: 페이지
        전환 중이라 컨텍스트가 잠깐 무효화된 경우) 무한정 못 기다리게 조용히 포기하고 리턴한다
        - CAPTCHA/느린 사이트에서 영원히 블로킹되면 안 되니까.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                state = self.driver.execute_script("return document.readyState")
            except Exception:
                return
            if state == "complete":
                return
            time.sleep(interval)

    # ------------------------------------------------------------------
    def _switch_to_new_tab_if_opened(self, prev_handles):
        """
        (2026-08-14 추가 - 새 탭으로 실제 콘텐츠가 열리는 경우 대응) 액션 전/후 window_handles
        개수를 비교해서 새 탭이 열렸으면 그리로 포커스를 옮기고, 남은 이전 탭들은 닫아서
        "항상 탭 1개만 유지"하는 단순한 흐름을 지킨다. WebVoyager 태스크들이 여러 탭을
        오가며 조작해야 하는 경우는 확인된 바 없어서, 멀티탭 상태를 그대로 유지하는 대신
        새 탭으로 완전히 옮겨가는 단순한 정책을 택했다(광고 팝업으로 새 탭이 열리는 경우도
        결과적으로 광고 탭을 닫고 원래 탭에 남는 게 아니라 광고 탭으로 옮겨가게 되는데,
        광고 팝업은 보통 about:blank나 별도 도메인이라 planner가 다음 스텝에서 이상한
        화면임을 보고 back/닫기를 시도할 수 있음 - 완벽하진 않지만, 최소한 "포커스가 다른
        곳에 가 있어서 실제 진행 상황이 하나도 안 보이는" 원래 문제보다는 낫다).
        """
        try:
            new_handles = list(self.driver.window_handles)
        except Exception:
            return
        if len(new_handles) <= len(prev_handles):
            return
        extra_handles = [h for h in new_handles if h not in prev_handles]
        target = extra_handles[-1] if extra_handles else new_handles[-1]
        for h in new_handles:
            if h == target:
                continue
            try:
                self.driver.switch_to.window(h)
                self.driver.close()
            except Exception as e:  # noqa: BLE001 - 이전 탭 하나 닫기 실패해도 나머지는 계속
                print(f"[env_webvoyager.py] 이전 탭 닫기 실패(무시하고 진행): {e}")
        self.driver.switch_to.window(target)

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
        # (2026-08 추가 - 구현) ComputerUseTool 스키마엔 시작점 필드가 원래 없어서 그동안
        # NotImplementedError였는데, planner.py의 drag 액션(target_description=시작점,
        # text=끝점)을 실제 픽셀 좌표로 변환해주는 쪽(_convert_planner_action_to_env)에서
        # start_coordinate를 채워서 넘겨주는 걸로 스키마를 확장했다 - 여기서 그 필드를 받아
        # mousedown(시작점) -> 중간 지점 여러 번 mousemove -> mouseup(끝점) 순으로 CDP 합성
        # 마우스 이벤트를 보낸다. 시작->끝으로 한 번에 점프시키면 dragover/mousemove 이벤트
        # 시퀀스를 기대하는 위젯(HTML5 draggable, 슬라이더 등)이 드래그를 인식 못 할 수 있어서,
        # 중간 지점을 보간해서 순서대로 이동시킨다.
        start = action.get("start_coordinate")
        if start is None or len(start) != 2:
            raise ValueError(f"start_coordinate=[x,y](드래그 시작점)가 필요함, 받은 값: {start!r}")
        end = action.get("coordinate")
        if end is None or len(end) != 2:
            raise ValueError(f"coordinate=[x,y](드래그 끝점)가 필요함, 받은 값: {end!r}")

        x1, y1 = float(start[0]), float(start[1])
        x2, y2 = float(end[0]), float(end[1])

        self._cdp_mouse("mouseMoved", x1, y1)
        self._cdp_mouse("mousePressed", x1, y1, click_count=1)

        steps = 5
        for i in range(1, steps + 1):
            ix = x1 + (x2 - x1) * i / steps
            iy = y1 + (y2 - y1) * i / steps
            self._cdp_mouse("mouseMoved", ix, iy)

        self._cdp_mouse("mouseReleased", x2, y2, click_count=1)

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
    env.auto_switch_new_tab = True
    env.driver = MagicMock()
    env.driver.get_screenshot_as_png.return_value = _fake_png_bytes()
    env.driver.execute_script.return_value = "complete"  # readyState 즉시 통과(대기 없이) - 아래 대량의
    # execute_action() 호출들이 _wait_for_page_ready()에서 실제로 1.5초씩 블로킹되는 걸 방지

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

    # (2026-08 추가 - drag 구현) start_coordinate/coordinate 둘 다 있으면 mousedown(시작) ->
    # 중간 지점 보간 mousemove들 -> mouseup(끝) 순으로 CDP 이벤트가 나가야 함
    env.execute_action({"action": "left_click_drag", "start_coordinate": [10, 20], "coordinate": [110, 20]})
    mouse_calls = [c.args[1] for c in env.driver.execute_cdp_cmd.call_args_list if c.args[0] == "Input.dispatchMouseEvent"]
    check("drag -> 첫 이벤트는 시작점으로 mouseMoved", mouse_calls[0]["type"] == "mouseMoved" and mouse_calls[0]["x"] == 10)
    check("drag -> 두번째는 시작점에서 mousePressed", mouse_calls[1]["type"] == "mousePressed" and mouse_calls[1]["x"] == 10)
    check("drag -> 마지막은 끝점에서 mouseReleased", mouse_calls[-1]["type"] == "mouseReleased" and mouse_calls[-1]["x"] == 110)
    mid_moves = [c for c in mouse_calls[2:-1] if c["type"] == "mouseMoved"]
    check("drag -> 시작/끝 사이에 보간된 mouseMoved가 여러 번 있음(순간이동 아님)", len(mid_moves) >= 3)
    check(
        "drag -> 보간된 x좌표가 시작<중간<끝 순으로 단조 증가",
        all(mid_moves[i]["x"] < mid_moves[i + 1]["x"] for i in range(len(mid_moves) - 1)),
    )
    env.driver.reset_mock()

    # start_coordinate 누락 -> ValueError (coordinate만으론 드래그 방향을 알 수 없음)
    try:
        env.execute_action({"action": "left_click_drag", "coordinate": [0, 0]})
        check("drag -> start_coordinate 누락 시 ValueError", False)
    except ValueError:
        check("drag -> start_coordinate 누락 시 ValueError", True)

    # coordinate(끝점) 누락 -> ValueError
    try:
        env.execute_action({"action": "left_click_drag", "start_coordinate": [0, 0]})
        check("drag -> coordinate(끝점) 누락 시 ValueError", False)
    except ValueError:
        check("drag -> coordinate(끝점) 누락 시 ValueError", True)

    # (2026-08-11 추가) back -> driver.back() 호출됨(CDP 아닌 Selenium 표준 히스토리 내비게이션)
    orig_sleep_back = time.sleep
    time.sleep = lambda *a, **k: None
    try:
        env.execute_action({"action": "back"})
        check("back -> driver.back() 호출됨", env.driver.back.called)
    finally:
        time.sleep = orig_sleep_back

    # --- (2026-08-14 추가 - 새 탭으로 실제 콘텐츠가 열리는 경우 대응) _switch_to_new_tab_if_opened ---
    class _FakeTabDriver:
        """window_handles/switch_to.window/close를 MagicMock보다 사실적으로 흉내내는 가짜
        드라이버 - 실제 Selenium처럼 close()가 "현재 switch_to된 핸들"을 닫는 것까지 재현해서,
        _switch_to_new_tab_if_opened()가 엉뚱한 탭을 닫지 않는지 검증한다."""

        def __init__(self, handles):
            self.window_handles = list(handles)
            self.current_handle = handles[0] if handles else None
            self.switch_to = MagicMock()
            self.switch_to.window.side_effect = self._do_switch
            self.close_calls = []

        def _do_switch(self, handle):
            self.current_handle = handle

        def close(self):
            self.close_calls.append(self.current_handle)
            self.window_handles = [h for h in self.window_handles if h != self.current_handle]

    # 새 탭 1개가 열린 경우: 새 탭으로 스위치하고, 이전 탭은 닫아야 함
    tab_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    tab_env.driver = _FakeTabDriver(["h1", "h2"])  # h1=기존 탭, h2=새로 열린 탭
    tab_env._switch_to_new_tab_if_opened(["h1"])
    check("_switch_to_new_tab_if_opened -> 새 탭(h2)으로 최종 스위치됨", tab_env.driver.current_handle == "h2")
    check("_switch_to_new_tab_if_opened -> 이전 탭(h1)은 닫힘", tab_env.driver.close_calls == ["h1"])
    check("_switch_to_new_tab_if_opened -> 탭 개수가 다시 1개로 유지됨", tab_env.driver.window_handles == ["h2"])

    # 새 탭이 안 열린 경우: 아무것도 안 해야 함(스위치/닫기 둘 다 호출 안 됨)
    no_new_tab_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    no_new_tab_env.driver = _FakeTabDriver(["h1"])
    no_new_tab_env._switch_to_new_tab_if_opened(["h1"])
    check(
        "_switch_to_new_tab_if_opened -> 새 탭 없으면 스위치 안 함",
        not no_new_tab_env.driver.switch_to.window.called,
    )
    check("_switch_to_new_tab_if_opened -> 새 탭 없으면 아무 탭도 안 닫힘", no_new_tab_env.driver.close_calls == [])

    # 새 탭이 여러 개 열린 경우(드문 케이스): 가장 나중 핸들로 스위치하고 나머지는 전부 닫음
    multi_tab_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    multi_tab_env.driver = _FakeTabDriver(["h1", "h2", "h3"])
    multi_tab_env._switch_to_new_tab_if_opened(["h1"])
    check("_switch_to_new_tab_if_opened -> 여러 새 탭 중 마지막 것으로 스위치", multi_tab_env.driver.current_handle == "h3")
    check(
        "_switch_to_new_tab_if_opened -> 나머지 탭들(h1,h2)은 전부 닫힘",
        sorted(multi_tab_env.driver.close_calls) == ["h1", "h2"],
    )

    # window_handles 조회 자체가 실패해도(드라이버 상태 이상 등) 예외가 밖으로 새면 안 됨
    failing_tab_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    failing_tab_env.driver = MagicMock()
    type(failing_tab_env.driver).window_handles = property(lambda self: (_ for _ in ()).throw(Exception("탭 조회 실패")))
    try:
        failing_tab_env._switch_to_new_tab_if_opened(["h1"])
        check("_switch_to_new_tab_if_opened -> window_handles 조회 실패해도 예외 안 남", True)
    except Exception:
        check("_switch_to_new_tab_if_opened -> window_handles 조회 실패해도 예외 안 남", False)

    # execute_action()이 fn(action) 직후 _switch_to_new_tab_if_opened를 호출하는지 배선 확인
    wiring_tab_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    wiring_tab_env.driver = MagicMock()
    wiring_tab_env.driver.current_url = "https://example.com/"
    wiring_tab_env.driver.window_handles = ["h1"]
    wiring_tab_env.driver.execute_script.return_value = "complete"  # readyState 즉시 통과(대기 없이)
    wiring_tab_env.driver.get_screenshot_as_png.return_value = _fake_png_bytes()
    wiring_tab_env.task_info = {}
    wiring_tab_env.auto_switch_new_tab = True
    tab_switch_calls = []
    wiring_tab_env._switch_to_new_tab_if_opened = lambda prev_handles: tab_switch_calls.append(list(prev_handles))
    wiring_tab_env.execute_action({"action": "left_click", "coordinate": [1, 1]})
    check(
        "execute_action -> 액션 실행 후 _switch_to_new_tab_if_opened가 액션 전 탭 목록과 함께 호출됨",
        tab_switch_calls == [["h1"]],
    )

    # (2026-08-15 추가 - ESPN 등 광고 새탭 리다이렉트 대응) auto_switch_new_tab=False면
    # 새 탭이 열려도 _switch_to_new_tab_if_opened 자체가 호출되지 않아야 함
    no_switch_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    no_switch_env.driver = MagicMock()
    no_switch_env.driver.current_url = "https://example.com/"
    no_switch_env.driver.window_handles = ["h1", "h2"]  # 새 탭이 실제로 열린 상태
    no_switch_env.driver.execute_script.return_value = "complete"
    no_switch_env.driver.get_screenshot_as_png.return_value = _fake_png_bytes()
    no_switch_env.task_info = {}
    no_switch_env.auto_switch_new_tab = False
    no_switch_calls = []
    no_switch_env._switch_to_new_tab_if_opened = lambda prev_handles: no_switch_calls.append(list(prev_handles))
    no_switch_env.execute_action({"action": "left_click", "coordinate": [1, 1]})
    check(
        "auto_switch_new_tab=False -> execute_action이 _switch_to_new_tab_if_opened를 아예 호출 안 함",
        no_switch_calls == [],
    )

    # --- (2026-08-15 추가 - 흰 화면 스크린샷 버그 픽스) _wait_for_page_ready() ---
    orig_sleep_ready = time.sleep
    time.sleep = lambda *a, **k: None
    try:
        # readyState가 처음부터 'complete'면 즉시 리턴(추가 폴링 없음)
        immediate_ready_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        immediate_ready_env.driver = MagicMock()
        immediate_ready_env.driver.execute_script.return_value = "complete"
        immediate_ready_env._wait_for_page_ready()
        check(
            "_wait_for_page_ready -> 이미 complete면 execute_script 딱 1번만 호출",
            immediate_ready_env.driver.execute_script.call_count == 1,
        )

        # 'loading' -> 'interactive' -> 'complete' 순으로 몇 번 폴링하다가 안정되면 리턴
        polling_ready_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        polling_ready_env.driver = MagicMock()
        polling_ready_env.driver.execute_script.side_effect = ["loading", "interactive", "complete"]
        polling_ready_env._wait_for_page_ready()
        check(
            "_wait_for_page_ready -> complete가 나올 때까지 폴링 후 리턴(3번 호출)",
            polling_ready_env.driver.execute_script.call_count == 3,
        )

        # execute_script 조회 자체가 실패하면(페이지 전환 중 컨텍스트 무효화 등) 예외 없이 조용히 리턴
        failing_ready_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        failing_ready_env.driver = MagicMock()
        failing_ready_env.driver.execute_script.side_effect = Exception("컨텍스트 무효화")
        try:
            failing_ready_env._wait_for_page_ready()
            check("_wait_for_page_ready -> execute_script 실패해도 예외 안 남", True)
        except Exception:
            check("_wait_for_page_ready -> execute_script 실패해도 예외 안 남", False)

        # timeout 안에 끝내 'complete'가 안 나와도 무한 대기하지 않고 그냥 리턴함(타임아웃 도달)
        never_ready_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        never_ready_env.driver = MagicMock()
        never_ready_env.driver.execute_script.return_value = "loading"
        fake_now = [0.0]
        orig_time_func = time.time
        time.time = lambda: fake_now[0]

        def _fake_sleep(secs, *a, **k):
            fake_now[0] += secs

        time.sleep = _fake_sleep
        try:
            never_ready_env._wait_for_page_ready(timeout=1.5, interval=0.25)
            check("_wait_for_page_ready -> timeout 안에 안 끝나도 무한 대기 안 하고 리턴", True)
        finally:
            time.time = orig_time_func
            time.sleep = lambda *a, **k: None
    finally:
        time.sleep = orig_sleep_ready

    # execute_action()이 스크린샷 찍기 전에 _wait_for_page_ready를 호출하는지 배선 확인
    wiring_ready_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    wiring_ready_env.driver = MagicMock()
    wiring_ready_env.driver.current_url = "https://example.com/"
    wiring_ready_env.driver.window_handles = ["h1"]
    wiring_ready_env.driver.get_screenshot_as_png.return_value = _fake_png_bytes()
    wiring_ready_env.task_info = {}
    wiring_ready_env.auto_switch_new_tab = True
    ready_calls = []
    wiring_ready_env._wait_for_page_ready = lambda: ready_calls.append(True)
    wiring_ready_env.execute_action({"action": "left_click", "coordinate": [1, 1]})
    check("execute_action -> 스크린샷 찍기 전에 _wait_for_page_ready가 호출됨", ready_calls == [True])

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

    # (2026-08-15 추가 - 구글 봇 차단 페이지 실측) title/URL이 둘 다 깨끗해도 본문(body)에
    # bot-check 문구가 있으면 잡아야 함 - 구글의 "Our systems have detected unusual
    # traffic..." 페이지가 정확히 이 경우(title/URL엔 아무 흔적 없음).
    env.driver.title = "Google Search"
    env.driver.current_url = "http://www.google.com/sorry/index"
    env.driver.execute_script.return_value = (
        "Our systems have detected unusual traffic from your computer network."
    )
    r_body = env.detect_bot_check()
    check(
        "detect_bot_check -> title/URL이 깨끗해도 body 문구로 감지됨(구글 sorry 페이지 대응)",
        r_body is not None and "body contains" in r_body["reason"] and "unusual traffic" in r_body["reason"],
    )

    # body 텍스트 조회 자체가 실패해도(페이지 전환 중 등) title/URL 기준 결과(정상)는 유지되고
    # 예외가 밖으로 새면 안 됨.
    env.driver.execute_script.side_effect = Exception("컨텍스트 무효화")
    r_body_fail = env.detect_bot_check()
    check("detect_bot_check -> body 조회 실패해도 예외 없이 None", r_body_fail is None)
    env.driver.execute_script.side_effect = None
    env.driver.execute_script.return_value = "complete"  # 이후 다른 테스트를 위해 원복

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

    # (2026-08 추가) Apple -> geo=US 쿠키 심음(실측 검증됨 - devtools에서 geo=KR을 US로 바꾸니
    # 실제로 가격/언어가 USD/영어로 바뀌는 것까지 확인된 쿠키)
    apple_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    apple_env.driver = MagicMock()
    apple_env.driver.current_url = "https://www.apple.com/shop/buy-iphone"
    apple_env._force_site_locale("https://www.apple.com/shop/buy-iphone")
    apple_cookie_calls = [c.args[0] for c in apple_env.driver.add_cookie.call_args_list]
    check(
        "_force_site_locale -> apple.com에서 geo=US 쿠키 심음",
        {"name": "geo", "value": "US"} in apple_cookie_calls,
    )
    check(
        "_force_site_locale -> apple 쿠키 반영 위해 같은 url로 재로드",
        apple_env.driver.get.call_args_list[-1].args == ("https://www.apple.com/shop/buy-iphone",),
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

    # (2026-08 추가) Google Search/Maps -> hl=en&gl=us 쿼리 파라미터 강제(쿠키 아님)
    google_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    google_env.driver = MagicMock()
    google_env.driver.current_url = "https://www.google.com/search?q=beauty+salons"
    google_env._force_site_locale("https://www.google.com/search?q=beauty+salons")
    check("_force_site_locale(google) -> 쿠키는 안 심음(URL 파라미터 방식)", not google_env.driver.add_cookie.called)
    reloaded_url = google_env.driver.get.call_args_list[-1].args[0]
    check("_force_site_locale(google) -> hl=en 파라미터 추가됨", "hl=en" in reloaded_url)
    check("_force_site_locale(google) -> gl=us 파라미터 추가됨", "gl=us" in reloaded_url)
    check("_force_site_locale(google) -> 기존 쿼리(q=beauty+salons)는 보존됨", "q=beauty" in reloaded_url)

    # 기존 쿼리에 hl/gl이 이미 있으면 덮어씀(중복 추가 아님)
    google_maps_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    google_maps_env.driver = MagicMock()
    google_maps_env.driver.current_url = "https://www.google.com/maps?hl=ko&gl=kr"
    google_maps_env._force_site_locale("https://www.google.com/maps?hl=ko&gl=kr")
    reloaded_maps_url = google_maps_env.driver.get.call_args_list[-1].args[0]
    check(
        "_force_site_locale(google maps) -> 기존 hl=ko를 en으로 덮어씀(중복 아님)",
        reloaded_maps_url.count("hl=") == 1 and "hl=en" in reloaded_maps_url,
    )
    check(
        "_force_site_locale(google maps) -> 기존 gl=kr을 us로 덮어씀(중복 아님)",
        reloaded_maps_url.count("gl=") == 1 and "gl=us" in reloaded_maps_url,
    )

    # URL 파라미터 보정 실패해도 예외 안 남
    failing_google_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    failing_google_env.driver = MagicMock()
    failing_google_env.driver.current_url = "https://www.google.com/search?q=x"
    failing_google_env.driver.get.side_effect = Exception("navigation failed")
    try:
        failing_google_env._force_site_locale("https://www.google.com/search?q=x")
        check("_force_site_locale(google) -> URL 재로드 실패해도 예외 안 남", True)
    except Exception:
        check("_force_site_locale(google) -> URL 재로드 실패해도 예외 안 남", False)

    # (2026-08 추가) Booking -> selected_currency=USD 파라미터 강제(실측 확인됨)
    booking_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    booking_env.driver = MagicMock()
    booking_env.driver.current_url = "https://www.booking.com/searchresults.html?ss=seattle"
    booking_env._force_site_locale("https://www.booking.com/searchresults.html?ss=seattle")
    booking_reloaded = booking_env.driver.get.call_args_list[-1].args[0]
    check(
        "_force_site_locale(booking) -> selected_currency=USD 파라미터 추가됨",
        "selected_currency=USD" in booking_reloaded,
    )
    check(
        "_force_site_locale(booking) -> 기존 쿼리(ss=seattle)는 보존됨",
        "ss=seattle" in booking_reloaded,
    )

    # --- (2026-08 추가) _reapply_url_locale_params_if_navigated(): Booking처럼 세션에 로케일이
    # 안 남는 사이트를 위해, URL이 실제로 바뀐 경우에만 로케일 파라미터를 다시 강제 ---
    # (2026-08-14 수정 - 폴링 대기가 생겨서 아래 테스트들은 time.sleep을 패치해서 실제로
    # 기다리지 않게 함. 이 구간만 국소적으로 패치하고 끝나면 원복한다.)
    orig_sleep_reapply = time.sleep
    time.sleep = lambda *a, **k: None
    try:
        reapply_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        reapply_env.driver = MagicMock()
        reapply_env.driver.current_url = "https://www.booking.com/hotel/kr/example.html"
        reapply_env._reapply_url_locale_params_if_navigated("https://www.booking.com/searchresults.html?ss=seattle")
        check(
            "_reapply_url_locale_params_if_navigated -> URL이 바뀌고 등록된 사이트면 다시 파라미터 강제함",
            reapply_env.driver.get.called
            and "selected_currency=USD" in reapply_env.driver.get.call_args_list[-1].args[0],
        )

        noop_reapply_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        noop_reapply_env.driver = MagicMock()
        same_url = "https://www.booking.com/searchresults.html?ss=seattle&selected_currency=USD"
        noop_reapply_env.driver.current_url = same_url
        noop_reapply_env._reapply_url_locale_params_if_navigated(same_url)
        check(
            "_reapply_url_locale_params_if_navigated -> URL이 안 바뀌었으면(모달/드롭다운 등) 재로드 안 함",
            not noop_reapply_env.driver.get.called,
        )

        unregistered_reapply_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        unregistered_reapply_env.driver = MagicMock()
        unregistered_reapply_env.driver.current_url = "https://en.wikipedia.org/wiki/Booking"
        unregistered_reapply_env._reapply_url_locale_params_if_navigated("https://en.wikipedia.org/wiki/Python")
        check(
            "_reapply_url_locale_params_if_navigated -> 등록 안 된 사이트로 이동했으면 재로드 안 함",
            not unregistered_reapply_env.driver.get.called,
        )

        # (2026-08-14 추가) 등록 안 된 사이트는 _wait_for_url_settle 폴링 없이 즉시 1회 비교만
        # 해야 함(안 그러면 매 액션마다 최대 2초씩 불필요한 지연이 생김) - sleep 호출 자체가
        # 없었는지 카운트해서 확인.
        sleep_calls = []
        no_settle_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        no_settle_env.driver = MagicMock()
        no_settle_env.driver.current_url = "https://en.wikipedia.org/wiki/Booking"
        orig_time_sleep = time.sleep
        time.sleep = lambda *a, **k: sleep_calls.append(a)
        try:
            no_settle_env._reapply_url_locale_params_if_navigated("https://en.wikipedia.org/wiki/Python")
        finally:
            time.sleep = orig_time_sleep
        check(
            "_reapply_url_locale_params_if_navigated -> 등록 안 된 사이트는 폴링 없이 즉시 비교(sleep 호출 없음)",
            len(sleep_calls) == 0,
        )

        # (2026-08-14 추가 - 레이스 컨디션 버그 픽스) _wait_for_url_settle(): 등록된 사이트로
        # 이동했을 때, URL이 몇 번의 폴링 동안 계속 바뀌다가(=브라우저가 아직 자체 네비게이션
        # 진행 중) 마지막에 안정되면 그 최종 안정된 URL을 기다렸다가 반환하는지 확인.
        class _FakeDriverSettlingUrl:
            def __init__(self, urls):
                self._urls = list(urls)
                self.get = MagicMock()

            @property
            def current_url(self):
                # 마지막 값에 도달하면 계속 그 값을 반환(연속 두 번 같아야 안정으로 판단하므로)
                if len(self._urls) > 1:
                    return self._urls.pop(0)
                return self._urls[0]

        settle_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        settle_env.driver = _FakeDriverSettlingUrl(
            [
                "https://www.booking.com/index.html",  # 액션 직후 아직 안정 안 된 중간 URL
                "https://www.booking.com/searchresults.html?ss=Mexico",  # 계속 바뀌는 중
                "https://www.booking.com/searchresults.html?ss=Mexico",  # 여기서 안정(직전과 동일)
            ]
        )
        settled = settle_env._wait_for_url_settle("https://www.booking.com/index.html")
        check(
            "_wait_for_url_settle -> 연속으로 같은 값이 나올 때까지 기다렸다가 안정된 최종 URL 반환",
            settled == "https://www.booking.com/searchresults.html?ss=Mexico",
        )

        # 액션 직후 URL이 계속 바뀌다가 안정된 뒤에야 _reapply가 그 최종 URL로 로케일 파라미터를
        # 강제하는지 end-to-end 확인 (레이스 컨디션 시나리오 재현 - 중간 URL로 잘못 재로드하면
        # 안 됨)
        race_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
        race_env.driver = _FakeDriverSettlingUrl(
            [
                "https://www.booking.com/index.html",
                "https://www.booking.com/searchresults.html?ss=Mexico",
                "https://www.booking.com/searchresults.html?ss=Mexico",
            ]
        )
        race_env._reapply_url_locale_params_if_navigated("https://www.booking.com/index.html")
        race_reload_url = race_env.driver.get.call_args_list[-1].args[0] if race_env.driver.get.call_args_list else ""
        check(
            "_reapply_url_locale_params_if_navigated -> 중간(미확정) URL이 아니라 안정된 최종 URL로 재적용됨",
            "searchresults.html" in race_reload_url and "ss=Mexico" in race_reload_url
            and "selected_currency=USD" in race_reload_url,
        )
    finally:
        time.sleep = orig_sleep_reapply

    # --- (2026-08 추가) execute_action()이 매 액션 후 _reapply_url_locale_params_if_navigated를 호출하는지 배선 확인 ---
    wiring_env = WebVoyagerEnv.__new__(WebVoyagerEnv)
    wiring_env.driver = MagicMock()
    wiring_env.driver.current_url = "https://www.booking.com/searchresults.html"
    wiring_env.driver.execute_script.return_value = "complete"  # readyState 즉시 통과(대기 없이)
    wiring_env.driver.get_screenshot_as_png.return_value = _fake_png_bytes()
    wiring_env.task_info = {}
    wiring_env.auto_switch_new_tab = True
    reapply_calls = []
    wiring_env._reapply_url_locale_params_if_navigated = lambda prev_url: reapply_calls.append(prev_url)
    wiring_env.execute_action({"action": "left_click", "coordinate": [10, 10]})
    check(
        "execute_action -> 액션 실행 후 _reapply_url_locale_params_if_navigated가 액션 전 URL과 함께 호출됨",
        reapply_calls == ["https://www.booking.com/searchresults.html"],
    )

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