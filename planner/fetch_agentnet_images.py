"""
curate_agentnet.py가 만든 manifest(jsonl)를 보고, HuggingFace의 xlangai/AgentNet
이미지 아카이브에서 "실제로 필요한 이미지만" HTTP Range 요청으로 뽑아온다.

아카이브 전체(ubuntu_images 68.7GB, win_mac_images 115.9GB, 총합 ~184GB)를
받지 않는다 - 대신 각 그룹의 마지막 볼륨(central directory가 들어있는 작은 파일,
ubuntu는 images.zip 3.7GB, win_mac은 images.zip 855MB)만 읽어서 어떤 이미지가
어느 볼륨의 어느 오프셋에 있는지 목록을 얻고, manifest에 있는 이미지 하나하나에
대해서만 그 위치의 바이트를 Range 요청으로 가져와 압축 해제한다.

원리는 remote_zip.py(여러 볼륨을 이어붙인 가상 파일)와 zip_multivol.py(멀티디스크
zip의 central directory/local header를 직접 파싱 - disk_number_start를 반영한
오프셋 계산과 CRC32 검증까지 포함) 두 모듈에 있다. 로컬에서 22개 파트로 쪼갠 테스트
zip(파일이 볼륨 경계를 넘나드는 경우 포함)과, 실제 HTTP Range 요청 경로까지
end-to-end로 검증을 마쳤다(30/30 엔트리 byte-identical + CRC32 일치).

사용 예:
  # 1) 원격 접근 없이 그냥 두 아카이브의 엔트리 수/manifest와의 매칭만 확인(가벼움, 권장 첫 실행)
  python fetch_agentnet_images.py --dry_run

  # 2) 실제 추출
  python fetch_agentnet_images.py

주의: HuggingFace 리졸브 URL에 대한 HTTP Range 요청이 필요하다. 이 스크립트는
Claude의 샌드박스에서는 huggingface.co가 아웃바운드 프록시에 막혀있어 직접
검증하지 못했다(로컬 멀티볼륨 zip으로만 검증) - 실제 사용자 환경(gpu-work)에서
--dry_run으로 먼저 확인해보길 권장.
"""
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_zip import HttpVolumeSource, MultiVolumeFile, head_with_retry
from zip_multivol import list_entries, extract_entry

_BASE_URL = "https://huggingface.co/datasets/xlangai/AgentNet/resolve/main"
# 2026-08-07 HF API로 확인한 실제 파트 개수 (ubuntu_images: z01~z13+zip,
# win_mac_images: z01~z23+zip). 데이터셋이 나중에 갱신되어 파트 수가 바뀌어도
# 동작하도록 아래 count가 안 맞으면 자동으로 순차 탐색(discovery)한다.
_KNOWN_PART_COUNTS = {"ubuntu": 13, "win_mac": 23}


def _volume_urls(group: str, session) -> list:
    urls = []
    n = _KNOWN_PART_COUNTS.get(group)
    if n:
        for i in range(1, n + 1):
            urls.append(f"{_BASE_URL}/{group}_images/images.z{i:02d}")
        # 하드코딩된 개수가 맞는지 한 번 검증 (마지막 파트가 실제로 존재하는지만 확인).
        # (2026-08 버그 수정) 예전엔 200이 아니면 무조건 "개수가 틀렸다"고 판단해서
        # 순차 discovery로 넘어갔는데, 실측해보니 이 체크 자체가 429(rate limit)를 맞아서
        # "파트 개수가 틀림"으로 오판 -> 불필요한 순차 탐색(요청 수만 더 늘림) -> 결국
        # 볼륨을 1개(images.zip)만 찾은 것처럼 되어버려서 완전히 잘못된 결과로 이어질
        # 뻔했다. 진짜로 "이 파트가 없다"는 404일 때만 discovery로 넘어가야 하고, 429/5xx는
        # head_with_retry가 백오프 재시도로 이미 처리해준다.
        last_check = head_with_retry(session, urls[-1], timeout=30)
        if last_check.status_code == 404:
            print(f"[fetch_agentnet_images] 경고: {group}의 파트 개수 하드코딩({n})이 "
                  f"실제로 안 맞는 것 같음(404) - 순차 탐색으로 재시도")
            urls = _discover_volume_urls(group, session)
        elif last_check.status_code != 200:
            last_check.raise_for_status()
    else:
        urls = _discover_volume_urls(group, session)
    urls.append(f"{_BASE_URL}/{group}_images/images.zip")
    return urls


def _discover_volume_urls(group: str, session) -> list:
    urls = []
    i = 1
    while True:
        u = f"{_BASE_URL}/{group}_images/images.z{i:02d}"
        r = head_with_retry(session, u, timeout=30)
        if r.status_code == 404:
            break
        if r.status_code != 200:
            r.raise_for_status()
        urls.append(u)
        i += 1
        if i > 200:  # 안전장치: 비정상적으로 많으면 중단
            raise RuntimeError(f"{group}: 볼륨 탐색이 200개를 넘음 - 뭔가 잘못됨")
    return urls


def _load_manifest(manifest_path: str):
    rows = []
    needs: Dict[str, Set[str]] = defaultdict(set)
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(rec)
            needs[rec["archive_group"]].add(rec["image"])
    return rows, needs


def _open_archive(group: str):
    import requests

    session = requests.Session()
    urls = _volume_urls(group, session)
    print(f"[fetch_agentnet_images] {group}: 볼륨 {len(urls)}개 ({urls[0]} ... {urls[-1]})")
    volumes = [HttpVolumeSource(u, session=session) for u in urls]
    mvf = MultiVolumeFile(volumes)
    print(f"[fetch_agentnet_images] {group}: 가상 전체 크기 {mvf.total_size/1e9:.2f}GB, "
          f"central directory 읽는 중 (마지막 볼륨만 건드림)...")
    entries = list_entries(mvf)
    print(f"[fetch_agentnet_images] {group}: 아카이브 내 전체 엔트리 {len(entries)}개")
    return mvf, entries


def dry_run(manifest_path: str):
    rows, needs = _load_manifest(manifest_path)
    for group, names in needs.items():
        mvf, entries = _open_archive(group)
        by_name = {e.filename: e for e in entries}
        by_base = {}
        for e in entries:
            by_base.setdefault(os.path.basename(e.filename), e)
        found = sum(1 for n in names if n in by_name or os.path.basename(n) in by_base)
        print(f"[fetch_agentnet_images] {group}: manifest 필요 이미지 {len(names)}개 중 "
              f"아카이브에서 매칭됨 {found}개, 못 찾음 {len(names) - found}개")


def _extract_and_write(mvf, entry, out_path: str):
    """스레드 워커 함수. extract_entry는 read_at() 기반이라 여러 스레드가 같은 mvf를
    동시에 호출해도 안전하다(remote_zip.py의 read_at() docstring 참고)."""
    data = extract_entry(mvf, entry)
    with open(out_path, "wb") as f:
        f.write(data)


def fetch_group(group: str, image_names: Set[str], out_dir: str,
                 platform_by_image: Dict[str, str], skip_existing: bool = True,
                 workers: int = 12):
    mvf, entries = _open_archive(group)
    by_name = {e.filename: e for e in entries}
    by_base = {}
    for e in entries:
        by_base.setdefault(os.path.basename(e.filename), e)

    got, missing, skipped = 0, 0, 0
    to_do = []  # (name, entry, out_path)
    for name in sorted(image_names):
        entry = by_name.get(name) or by_base.get(os.path.basename(name))
        if entry is None:
            missing += 1
            continue
        platform = platform_by_image.get(name, group)
        out_path = os.path.join(out_dir, platform, os.path.basename(entry.filename))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if skip_existing and os.path.exists(out_path):
            skipped += 1
            continue
        to_do.append((name, entry, out_path))

    print(f"[fetch_agentnet_images] {group}: 병렬 다운로드 시작 (workers={workers}, "
          f"신규 필요 {len(to_do)}개, 기존스킵 {skipped}개, 못찾음 {missing}개)")

    t0 = time.time()
    if to_do:
        # 모든 워커가 같은 mvf(=같은 requests.Session, 커넥션 풀 공유)를 참조한다 - HTTP GET
        # 요청은 스레드 간에 서로 상태를 공유하지 않으므로 안전하고, 커넥션 풀 재사용 덕에
        # 매번 새 TCP/TLS 핸드셰이크를 안 해도 돼서 순차 버전보다 빨라진다.
        with _cf.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(_extract_and_write, mvf, entry, out_path): (name, out_path)
                for name, entry, out_path in to_do
            }
            done = 0
            failed_in_batch = 0
            for fut in _cf.as_completed(future_to_task):
                name, out_path = future_to_task[fut]
                done += 1
                try:
                    fut.result()
                    got += 1
                except Exception as e:  # noqa: BLE001
                    missing += 1
                    failed_in_batch += 1
                    print(f"[fetch_agentnet_images] {group}: {name} 추출 실패 - {e}")
                if done % 200 == 0 or done == len(to_do):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0.0
                    print(f"[fetch_agentnet_images] {group}: {done}/{len(to_do)} 처리 "
                          f"(신규 {got}, 실패 {failed_in_batch}, {elapsed:.0f}s 경과, {rate:.1f}장/s)")

    print(f"[fetch_agentnet_images] {group} 완료: 신규 {got}, 기존스킵 {skipped}, 못찾음/실패 {missing}")
    return got, skipped, missing


def _cli():
    ap = argparse.ArgumentParser(description="AgentNet manifest에 필요한 이미지만 원격 zip에서 추출")
    ap.add_argument("--manifest", default="data/processed/planner_agentnet_manifest.jsonl")
    ap.add_argument("--out_dir", default="data/processed/images/agentnet")
    ap.add_argument("--no_skip_existing", action="store_true", help="이미 받은 파일도 다시 받기")
    ap.add_argument("--dry_run", action="store_true",
                     help="실제로 다운로드하지 않고 manifest와 아카이브 엔트리 매칭만 확인")
    ap.add_argument("--workers", type=int, default=6,
                     help="동시 Range 요청 스레드 수 (기본 6). 순차 실행시 장당 ~2초였던 게 "
                          "네트워크 왕복 대기 때문이라 병렬화로 크게 단축됨. (2026-08 수정) "
                          "기본값을 12->6으로 낮췄다 - 실측으로 workers=12에서 HF가 429(rate "
                          "limit)를 던지기 시작하는 걸 확인함(2689개 중 361개 실패). 이제 429는 "
                          "자동 백오프 재시도로 처리되긴 하지만, 애초에 덜 유발하는 게 더 안전함.")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[fetch_agentnet_images] manifest 없음: {args.manifest} "
              f"(curate_agentnet.py --curate 먼저 실행)")
        sys.exit(1)

    if args.dry_run:
        dry_run(args.manifest)
        return

    rows, needs = _load_manifest(args.manifest)
    platform_by_image = {r["image"]: r.get("platform", r["archive_group"]) for r in rows}
    total_got = total_skipped = total_missing = 0
    for group, names in needs.items():
        got, skipped, missing = fetch_group(
            group, names, args.out_dir, platform_by_image,
            skip_existing=not args.no_skip_existing,
            workers=args.workers,
        )
        total_got += got
        total_skipped += skipped
        total_missing += missing
    print(f"[fetch_agentnet_images] 전체 완료: 신규 {total_got}, 기존스킵 {total_skipped}, "
          f"못찾음/실패 {total_missing}")


if __name__ == "__main__":
    _cli()
