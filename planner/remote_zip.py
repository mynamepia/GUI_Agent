"""
여러 파일(볼륨)로 쪼개진 zip 아카이브를 "이어붙인 하나의 파일"처럼 보이게 만드는
파일류(file-like) 어댑터.

핵심 아이디어: split zip은 zip -s로 만들 때 물리적으로 여러 파일(.z01, .z02, ...,
.zip)로 나뉘어 있을 뿐, 그 파일들을 순서대로 이어붙이면 완전히 정상적인 하나의 zip
스트림이 된다(Info-ZIP의 -s 옵션 자체가 그렇게 동작). 그래서 zip 포맷(EOCD/ZIP64/
central directory/local header/DEFLATE 등)을 직접 파싱할 필요 없이, seek/read만
올바르게 "가상의 전체 오프셋 -> 실제 볼륨+오프셋"으로 변환해주면 Python 표준
zipfile.ZipFile이 나머지(CRC32 검증 포함)를 전부 대신 해준다.

이 클래스는 두 가지 백엔드를 지원한다:
- LocalVolumeSource: 로컬 디스크의 여러 파일 (테스트/검증용)
- HttpVolumeSource: HTTP Range 요청으로 원격 파일의 일부만 받아오는 버전 (실제 사용)
"""
from __future__ import annotations

import io
import os
import random
import time
from typing import List, Sequence


def _backoff_sleep(response, attempt: int, base_delay: float = 3.0, max_delay: float = 60.0):
    """
    (2026-08 추가) HF가 429(Too Many Requests)를 던질 때 쓰는 재시도 대기 로직.
    이전 버전은 429를 받아도 그냥 즉시 재시도했는데, 그러면 이미 rate limit이 걸린
    상태로 계속 요청을 더 쏘게 돼서 오히려 상황이 악화된다(실측: workers=12로
    병렬화한 뒤 실제로 이 문제가 터져서 361/2689개가 429로 실패함). Retry-After
    헤더가 있으면 그 값을 우선 쓰고, 없으면 지수 백오프(+지터)로 대기한다.
    """
    retry_after = None
    if response is not None:
        retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            delay = base_delay * (2 ** attempt)
    else:
        delay = min(max_delay, base_delay * (2 ** attempt))
    delay += random.uniform(0, delay * 0.2)
    time.sleep(delay)


def head_with_retry(session, url: str, timeout: float = 60.0, max_retries: int = 8):
    """
    HEAD 요청 + 429 백오프 재시도가 필요한 곳(HttpVolumeSource._fetch_size와
    fetch_agentnet_images.py의 볼륨 개수 확인 로직)에서 공통으로 쓰는 헬퍼.
    404는 재시도해도 의미 없으므로(진짜로 없는 파일) 그대로 반환하고, 429/5xx만
    백오프 후 재시도한다.
    """
    last_response = None
    for attempt in range(max_retries):
        r = session.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code == 429 or r.status_code >= 500:
            last_response = r
            if attempt < max_retries - 1:
                print(f"[remote_zip] {url}: HTTP {r.status_code}, 백오프 후 재시도 "
                      f"({attempt + 1}/{max_retries})...")
                _backoff_sleep(r, attempt)
                continue
        return r
    return last_response


class VolumeSource:
    """볼륨 하나(파일 하나)에서 바이트 범위를 읽어오는 인터페이스."""

    def size(self) -> int:
        raise NotImplementedError

    def read_range(self, start: int, length: int) -> bytes:
        """[start, start+length) 바이트를 반환. length는 볼륨 크기를 넘지 않는다고 가정."""
        raise NotImplementedError


class LocalVolumeSource(VolumeSource):
    def __init__(self, path: str):
        self.path = path
        self._size = os.path.getsize(path)

    def size(self) -> int:
        return self._size

    def read_range(self, start: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(start)
            data = f.read(length)
        if len(data) != length:
            raise IOError(
                f"{self.path}: expected {length} bytes at {start}, got {len(data)}"
            )
        return data


class HttpVolumeSource(VolumeSource):
    """HTTP Range 요청으로 원격 파일 일부를 읽는다. requests 라이브러리 사용.

    HEAD로 전체 크기를 먼저 알아내고, 이후 Range: bytes=start-end 헤더로
    필요한 구간만 GET한다(HF의 LFS 백엔드는 Range 요청을 지원함).
    """

    def __init__(self, url: str, session=None, timeout: float = 60.0, max_retries: int = 8):
        import requests  # local import: 이 백엔드를 실제로 쓸 때만 필요

        self.url = url
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self._size = self._fetch_size()

    def _fetch_size(self) -> int:
        r = head_with_retry(self.session, self.url, timeout=self.timeout, max_retries=self.max_retries)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        if cl is None:
            raise IOError(f"{self.url}: no Content-Length header, cannot determine size")
        return int(cl)

    def size(self) -> int:
        return self._size

    def read_range(self, start: int, length: int) -> bytes:
        end = start + length - 1
        headers = {"Range": f"bytes={start}-{end}"}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(
                    self.url, headers=headers, allow_redirects=True, timeout=self.timeout
                )
                if r.status_code == 429 or r.status_code >= 500:
                    # (2026-08 버그 수정) 예전엔 여기서 바로 예외를 던지고 즉시 재시도했는데,
                    # 그러면 이미 rate limit 걸린 상태에서 대기 없이 계속 두드리는 꼴이라
                    # 상황이 더 나빠진다(실측: 병렬화 후 429가 361건 발생). 백오프 대기 후
                    # 재시도.
                    if attempt < self.max_retries - 1:
                        _backoff_sleep(r, attempt)
                        continue
                    r.raise_for_status()
                if r.status_code not in (200, 206):
                    raise IOError(f"{self.url}: unexpected status {r.status_code}")
                data = r.content
                if len(data) != length:
                    raise IOError(
                        f"{self.url}: expected {length} bytes, got {len(data)} "
                        f"(range {start}-{end})"
                    )
                return data
            except Exception as e:  # noqa: BLE001 - 재시도용으로 폭넓게 잡음
                last_err = e
                if attempt == self.max_retries - 1:
                    raise
        raise last_err  # pragma: no cover


class MultiVolumeFile(io.RawIOBase):
    """여러 VolumeSource를 순서대로 이어붙인 것처럼 보이는 seekable/readable 가상 파일.

    zipfile.ZipFile(fileobj=...)에 그대로 넘길 수 있다.
    """

    def __init__(self, volumes: Sequence[VolumeSource]):
        if not volumes:
            raise ValueError("volumes must be non-empty")
        self._volumes = list(volumes)
        self._sizes = [v.size() for v in self._volumes]
        # 각 볼륨의 시작 글로벌 오프셋 (누적합)
        self._starts: List[int] = []
        acc = 0
        for s in self._sizes:
            self._starts.append(acc)
            acc += s
        self._total_size = acc
        self._pos = 0

    # --- io.RawIOBase 필수 인터페이스 ---
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = self._total_size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new_pos < 0:
            raise ValueError("negative seek position")
        self._pos = new_pos
        return self._pos

    def readinto(self, b) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._total_size - self._pos
        data = self.read_at(self._pos, size)
        self._pos += len(data)
        return data

    def read_at(self, offset: int, length: int) -> bytes:
        """
        (병렬 다운로드 지원용 추가) self._pos를 건드리지 않는 stateless 버전의 read.
        seek()+read()는 self._pos라는 공유 가변 상태를 쓰기 때문에 여러 스레드가 같은
        MultiVolumeFile 인스턴스에 동시에 seek+read를 하면 서로의 위치를 덮어써서
        레이스 컨디션이 난다(한 스레드의 seek 직후 다른 스레드가 seek해버리면 앞 스레드의
        read가 엉뚱한 위치를 읽음). zip_multivol.extract_entry()가 병렬 실행될 때는 이
        메서드를 써서 오프셋을 매번 인자로 명시적으로 넘긴다 - 이러면 여러 스레드가 같은
        MultiVolumeFile/세션을 공유해도 서로 간섭하지 않는다(HTTP 요청 자체는 매번 독립적).
        list_entries()처럼 central directory를 한 번만 순차적으로 읽는 곳은 그대로
        seek()+read()를 써도 무방(병렬 구간 진입 전에 끝남).
        """
        if length <= 0 or offset >= self._total_size:
            return b""
        length = min(length, self._total_size - offset)

        vol_idx = self._volume_index_for(offset)
        out = bytearray()
        remaining = length
        cur_global = offset
        while remaining > 0 and vol_idx < len(self._volumes):
            vol_start = self._starts[vol_idx]
            vol_size = self._sizes[vol_idx]
            offset_in_vol = cur_global - vol_start
            available_in_vol = vol_size - offset_in_vol
            take = min(remaining, available_in_vol)
            if take > 0:
                chunk = self._volumes[vol_idx].read_range(offset_in_vol, take)
                out += chunk
                cur_global += take
                remaining -= take
            vol_idx += 1

        return bytes(out)

    def _volume_index_for(self, global_pos: int) -> int:
        # starts는 오름차순이므로 이분탐색 가능하지만 볼륨 수가 많지 않아 선형탐색으로 충분
        for i in range(len(self._volumes) - 1, -1, -1):
            if global_pos >= self._starts[i]:
                return i
        return 0

    @property
    def total_size(self) -> int:
        return self._total_size

    def disk_start(self, disk_number: int) -> int:
        """디스크(볼륨) 번호(0-based)에 대응하는 글로벌 시작 오프셋."""
        if not (0 <= disk_number < len(self._starts)):
            raise ValueError(
                f"disk_number={disk_number} out of range (volumes={len(self._starts)})"
            )
        return self._starts[disk_number]
