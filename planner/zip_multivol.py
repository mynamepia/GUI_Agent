"""
멀티볼륨(split) zip을 위한 최소 central-directory 파서 + 개별 엔트리 추출기.

왜 표준 zipfile을 그대로 못 쓰나:
Python stdlib zipfile은 central directory의 "local header offset"을 항상
"파일 전체를 하나로 이어붙인 스트림 안에서의 절대 오프셋"으로 해석한다. 그런데
Info-ZIP의 -s(split) 아카이브는 실제로는 진짜 멀티디스크 포맷이라, 각 엔트리의
header_offset은 "그 엔트리가 시작하는 디스크(volume) 안에서의 상대 오프셋"이고
어느 디스크인지는 central directory record의 disk_number_start 필드로 별도
지정된다. 단순히 볼륨 파일들을 순서대로 이어붙인 가상 스트림을 stdlib zipfile에
넘기면, stdlib은 이 disk_number_start를 반영하지 않기 때문에 (실측: concat 보정
로직이 오작동해서 완전히 엉뚱한 offset을 계산함 - "Bad magic number for file
header" 에러로 확인됨) local header를 잘못된 위치에서 찾으려다 실패한다.

그래서 이 모듈은 central directory / local header를 APPNOTE.TXT 스펙대로 직접
파싱하고, disk_number_start를 이용해 "해당 디스크의 글로벌 시작 오프셋 +
디스크-상대 offset"으로 직접 계산한 뒤 그 위치에서 읽는다. ZIP64(4GB 초과 아카이브)
확장 필드도 처리한다 - AgentNet의 아카이브 전체 크기가 4GB를 훌쩍 넘기 때문에
EOCD64/EOCD64 locator가 필요할 가능성이 높다.

압축 해제는 zlib(DEFLATE)만 지원한다 - 일반적인 zip 생성 도구의 기본 압축 방식이며,
STORED(비압축)도 지원한다. 그 외 방식이 나오면 명시적으로 에러를 낸다(조용히
틀린 결과를 내지 않기 위함).
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import BinaryIO, List, Optional

_EOCD_SIG = b"PK\x05\x06"
_EOCD64_SIG = b"PK\x06\x06"
_EOCD64_LOC_SIG = b"PK\x06\x07"
_CD_SIG = b"PK\x01\x02"
_LFH_SIG = b"PK\x03\x04"

_EOCD_MIN_SIZE = 22
_EOCD64_LOC_SIZE = 20
_EOCD64_MIN_SIZE = 56
_CD_FIXED_SIZE = 46
_LFH_FIXED_SIZE = 30

_ZIP64_EXTRA_ID = 0x0001


@dataclass
class ZipEntry:
    filename: str
    compress_type: int
    crc32: int
    compress_size: int
    uncompress_size: int
    disk_number_start: int
    local_header_offset: int  # 해당 디스크 내 상대 offset (원본 그대로, 아직 보정 안 됨)


def _find_eocd(mvf) -> bytes:
    """파일 끝에서부터 역방향으로 EOCD 시그니처를 찾는다.

    EOCD는 파일 맨 끝의 (가변 길이 comment 때문에) 최대 22+65535 바이트 안에
    있어야 한다는 제약을 이용해 뒤쪽 일정 구간만 읽어서 찾는다.
    """
    total = mvf.total_size
    search_size = min(total, _EOCD_MIN_SIZE + 65535 + 4)
    mvf.seek(total - search_size)
    tail = mvf.read(search_size)
    idx = tail.rfind(_EOCD_SIG)
    if idx < 0:
        raise ValueError("EOCD 시그니처를 못 찾음 - zip 파일이 아니거나 손상됨")
    return tail[idx:]


def _parse_eocd(eocd: bytes):
    (disk_num, cd_start_disk, entries_this_disk, entries_total,
     cd_size, cd_offset, comment_len) = struct.unpack("<HHHHIIH", eocd[4:22])
    return {
        "disk_num": disk_num,
        "cd_start_disk": cd_start_disk,
        "entries_this_disk": entries_this_disk,
        "entries_total": entries_total,
        "cd_size": cd_size,
        "cd_offset": cd_offset,
    }


def _maybe_upgrade_to_zip64(mvf, eocd_info: dict, eocd_global_pos: int) -> dict:
    """EOCD 필드가 0xFFFF/0xFFFFFFFF 센티널이면 ZIP64 EOCD를 찾아 진짜 값으로 교체."""
    needs64 = (
        eocd_info["cd_size"] == 0xFFFFFFFF
        or eocd_info["cd_offset"] == 0xFFFFFFFF
        or eocd_info["entries_total"] == 0xFFFF
        or eocd_info["cd_start_disk"] == 0xFFFF
    )
    if not needs64:
        return eocd_info

    # EOCD64 locator는 EOCD 바로 앞, 고정 20바이트
    loc_pos = eocd_global_pos - _EOCD64_LOC_SIZE
    if loc_pos < 0:
        raise ValueError("ZIP64 EOCD locator를 찾을 위치가 없음")
    mvf.seek(loc_pos)
    loc = mvf.read(_EOCD64_LOC_SIZE)
    if loc[:4] != _EOCD64_LOC_SIG:
        raise ValueError("ZIP64 EOCD locator 시그니처 불일치")
    _, eocd64_disk, eocd64_offset, total_disks = struct.unpack("<IIQI", loc)

    # (2026-08 버그 수정) locator가 담고 있는 eocd64_offset은 글로벌 오프셋이
    # 아니라 "eocd64_disk 번 디스크 안에서의 상대 오프셋"이다(central directory
    # local header offset과 동일한 종류의 디스크-상대 offset). 이걸 그대로 글로벌
    # seek 위치로 썼더니 실제 아카이브(24볼륨, 124GB)에서
    # "ZIP64 EOCD 시그니처 불일치"로 터졌다 - disk_start()를 더해 글로벌 오프셋으로
    # 보정해야 한다.
    eocd64_global_offset = mvf.disk_start(eocd64_disk) + eocd64_offset
    mvf.seek(eocd64_global_offset)
    header = mvf.read(_EOCD64_MIN_SIZE)
    if header[:4] != _EOCD64_SIG:
        raise ValueError("ZIP64 EOCD 시그니처 불일치")
    (_, _size_of_rec, _ver_made, _ver_needed, disk_num, cd_start_disk,
     entries_this_disk, entries_total, cd_size, cd_offset) = struct.unpack(
        "<I QHH IIQQQQ".replace(" ", ""), header
    )
    return {
        "disk_num": disk_num,
        "cd_start_disk": cd_start_disk,
        "entries_this_disk": entries_this_disk,
        "entries_total": entries_total,
        "cd_size": cd_size,
        "cd_offset": cd_offset,
    }


def _parse_zip64_extra(extra: bytes, need_size=False, need_csize=False,
                        need_offset=False, need_disk=False):
    """central directory 엔트리의 zip64 extra field(id=0x0001)에서 실제 64비트
    값을 순서대로(uncompressed_size, compressed_size, local_header_offset,
    disk_number_start 순 - 스펙상 "센티널이 걸린 필드만, 이 순서로" 들어있음)
    뽑아낸다."""
    pos = 0
    while pos + 4 <= len(extra):
        field_id, field_len = struct.unpack_from("<HH", extra, pos)
        data = extra[pos + 4: pos + 4 + field_len]
        if field_id == _ZIP64_EXTRA_ID:
            vals = {}
            off = 0
            if need_size:
                vals["uncompress_size"] = struct.unpack_from("<Q", data, off)[0]
                off += 8
            if need_csize:
                vals["compress_size"] = struct.unpack_from("<Q", data, off)[0]
                off += 8
            if need_offset:
                vals["local_header_offset"] = struct.unpack_from("<Q", data, off)[0]
                off += 8
            if need_disk:
                vals["disk_number_start"] = struct.unpack_from("<I", data, off)[0]
                off += 4
            return vals
        pos += 4 + field_len
    return {}


def list_entries(mvf) -> List[ZipEntry]:
    """멀티볼륨 zip의 central directory를 직접 파싱해 엔트리 목록을 반환한다."""
    tail_search_start = max(0, mvf.total_size - (_EOCD_MIN_SIZE + 65535 + 4))
    eocd = _find_eocd(mvf)
    eocd_global_pos = tail_search_start + _rfind_pos_hint(mvf, tail_search_start, eocd)
    info = _parse_eocd(eocd)
    info = _maybe_upgrade_to_zip64(mvf, info, eocd_global_pos)

    cd_start_disk = info["cd_start_disk"]
    cd_offset_on_disk = info["cd_offset"]
    cd_global_offset = mvf.disk_start(cd_start_disk) + cd_offset_on_disk

    mvf.seek(cd_global_offset)
    cd_bytes = mvf.read(info["cd_size"])

    entries: List[ZipEntry] = []
    pos = 0
    while pos < len(cd_bytes):
        sig = cd_bytes[pos:pos + 4]
        if sig != _CD_SIG:
            raise ValueError(f"central directory 레코드 시그니처 불일치 at pos={pos}")
        fixed = cd_bytes[pos:pos + _CD_FIXED_SIZE]
        (_, _ver_made, _ver_needed, _flags, compress_type, _time, _date,
         crc32, compress_size, uncompress_size, fname_len, extra_len, comment_len,
         disk_number_start, _int_attr, _ext_attr, local_header_offset) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", fixed
        )
        name_start = pos + _CD_FIXED_SIZE
        filename = cd_bytes[name_start: name_start + fname_len].decode("utf-8", "replace")
        extra_start = name_start + fname_len
        extra = cd_bytes[extra_start: extra_start + extra_len]

        need_size = uncompress_size == 0xFFFFFFFF
        need_csize = compress_size == 0xFFFFFFFF
        need_offset = local_header_offset == 0xFFFFFFFF
        need_disk = disk_number_start == 0xFFFF
        if need_size or need_csize or need_offset or need_disk:
            z64 = _parse_zip64_extra(extra, need_size, need_csize, need_offset, need_disk)
            uncompress_size = z64.get("uncompress_size", uncompress_size)
            compress_size = z64.get("compress_size", compress_size)
            local_header_offset = z64.get("local_header_offset", local_header_offset)
            disk_number_start = z64.get("disk_number_start", disk_number_start)

        entries.append(ZipEntry(
            filename=filename,
            compress_type=compress_type,
            crc32=crc32,
            compress_size=compress_size,
            uncompress_size=uncompress_size,
            disk_number_start=disk_number_start,
            local_header_offset=local_header_offset,
        ))

        comment_start = extra_start + extra_len
        pos = comment_start + comment_len

    return entries


def _rfind_pos_hint(mvf, tail_search_start: int, eocd_bytes: bytes) -> int:
    # _find_eocd가 읽은 tail 버퍼 안에서의 상대 위치를 다시 계산(간단히 재탐색)
    search_size = mvf.total_size - tail_search_start
    mvf.seek(tail_search_start)
    tail = mvf.read(search_size)
    idx = tail.rfind(_EOCD_SIG)
    return idx


def extract_entry(mvf, entry: ZipEntry) -> bytes:
    """엔트리 하나를 읽어서 압축 해제 + CRC32 검증까지 마친 바이트를 반환.

    (병렬 다운로드 지원용 수정) seek()+read() 대신 read_at()을 쓴다 - seek()+read()는
    mvf._pos라는 공유 가변 상태에 의존하는데, 여러 스레드가 동시에 extract_entry()를
    호출하면 서로의 seek 위치를 덮어써서 레이스 컨디션이 생긴다. read_at()은 오프셋을
    인자로 직접 받아서 상태를 안 건드리므로 스레드 안전하다.
    """
    global_offset = mvf.disk_start(entry.disk_number_start) + entry.local_header_offset
    lfh = mvf.read_at(global_offset, _LFH_FIXED_SIZE)
    if lfh[:4] != _LFH_SIG:
        raise ValueError(
            f"{entry.filename}: local file header 시그니처 불일치 "
            f"(global_offset={global_offset}, disk={entry.disk_number_start}, "
            f"disk_relative_offset={entry.local_header_offset})"
        )
    (_, _ver, _flags, _compress_type, _time, _date, _crc, _csize, _usize,
     fname_len, extra_len) = struct.unpack("<IHHHHHIIIHH", lfh)
    data_start = global_offset + _LFH_FIXED_SIZE + fname_len + extra_len

    compress_size = entry.compress_size
    raw = mvf.read_at(data_start, compress_size)

    if entry.compress_type == 0:
        data = raw
    elif entry.compress_type == 8:
        data = zlib.decompress(raw, -15)  # raw DEFLATE, zlib 헤더 없음
    else:
        raise ValueError(f"{entry.filename}: 지원하지 않는 압축 방식 {entry.compress_type}")

    if len(data) != entry.uncompress_size:
        raise ValueError(
            f"{entry.filename}: 압축 해제 후 크기 불일치 "
            f"(기대 {entry.uncompress_size}, 실제 {len(data)})"
        )
    actual_crc = zlib.crc32(data) & 0xFFFFFFFF
    if actual_crc != entry.crc32:
        raise ValueError(
            f"{entry.filename}: CRC32 불일치 (기대 {entry.crc32:#x}, 실제 {actual_crc:#x})"
        )
    return data
