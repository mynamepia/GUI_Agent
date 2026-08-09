"""
curate_agentnet.py

planner LoRA 학습용으로 AgentNet(xlangai/AgentNet, HuggingFace)에서 필요한 만큼만
큐레이션해서 받는 스크립트. AgentNet은 total ~200GB(이미지 아카이브만 184GB+)짜리
대형 데이터셋이라, 이걸 통째로 받는 건 로컬 컴퓨터 기준 비현실적임 - 그래서 이 스크립트는
**이미지는 아직 건드리지 않고, 텍스트 궤적(jsonl)만 먼저 받아서 어떤 이미지가 실제로
필요한지부터 정하는 단계**를 담당한다. 이미지 자체를 받는 건 다음 단계 스크립트
(fetch_agentnet_images.py, 아직 없음 - curate 결과를 보고 나서 작성 예정)의 몫.

[왜 2단계로 나눴나]
AgentNet 이미지 아카이브는 멀티파트 zip(ubuntu: .z01~.z13+zip 총 68.7GB, win_mac:
.z01~.z23+zip 총 115.9GB)이라 "zip -s 0 ... --out"으로 합쳐서 풀어야 하는데, 이러면
필요한 이미지 몇 천 장만 쓸 거여도 68~116GB를 통째로 받아야 함. 반면 궤적 jsonl은
ubuntu 282MB, win_mac 1.3GB로 훨씬 작음 - 그러니까 먼저 jsonl만 받아서 "정확히 어떤
이미지 파일명이 필요한지" 리스트를 뽑아내고, 그 다음에 (이미지 아카이브 전체가 아니라)
그 리스트에 있는 파일만 골라 받는 게 합리적. 이 스크립트는 그 첫 단계.

[사용법 - 3단계로 나눠 실행]
    # 1) jsonl만 받기 (총 ~1.6GB, huggingface_hub 필요: pip install -U huggingface_hub)
    python curate_agentnet.py --download

    # 2) 실제 필드 구조 확인 (README 예시와 실제 배포본이 다를 수 있어서 - 특히
    #    windows/macos를 구분하는 필드명은 README에 명시가 안 돼 있어 직접 확인 필요)
    python curate_agentnet.py --inspect

    # 3) 큐레이션 + manifest 작성 (이미지 다운로드는 안 함 - 파일명 목록만 뽑음)
    python curate_agentnet.py --curate --total_budget 4000 --out ..\\data\\processed\\planner_agentnet_manifest.jsonl

[예산(--total_budget) 잡은 근거]
기존 grounding LoRA 학습이 700k 해상도, 약 6000개 이미지 기준 4시간 걸렸다고 했음.
planner 학습도 이미지 1장+텍스트 출력이라는 점에서 연산량 단위가 비슷할 걸로 가정하고,
일단 그보다 확실히 작게(기본 4000) 잡아서 첫 파일럿이 6000개/4시간보다 오래 걸리지
않게 함 - 부족하면 --total_budget을 올려서 재실행하면 됨(멱등적으로 다시 샘플링됨,
--seed 고정이라 재현 가능).

[플랫폼 밸런싱 - 2026-08-07 실측 확정]
--inspect로 실제 필드를 확인해보니, win_mac 궤적(agentnet_win_mac_18k.jsonl) 자체에는
windows/macos 구분 필드가 아예 없었다. 대신 meta_data_merged.jsonl에 task_id로 join되는
"system"("Windows"/"Mac" 등) 필드가 있고, screen_width/screen_height/applications 등
부가 정보도 여기 있다. 그래서 win_mac 궤적의 platform은 이 meta 파일과 task_id로
join해서 결정한다(ubuntu는 이미 파일 자체가 ubuntu 전용이라 join 불필요). meta에서
못 찾은 task_id는 "win_mac_unknown"으로 분류하고 별도 카운트만 남긴다(조용히 누락되지
않게).

[품질 필터]
- 궤적(trajectory) 레벨: task_completed=True인 것만 사용(실패한 궤적을 "정답"으로
  학습시키면 안 되니까 - verifier_v1의 "TRAIN 데이터만/누출 금지" 원칙과 같은 결의
  "깨끗한 라벨만 쓴다" 원칙)
- 스텝(step) 레벨: last_step_correct=True 이고 last_step_redundant가 False인 스텝만
  사용(오답/중복 스텝을 정답처럼 학습시키지 않기 위함)
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

_REPO_ID = "xlangai/AgentNet"
_JSONL_FILES = {
    "ubuntu": "agentnet_ubuntu_5k.jsonl",
    "win_mac": "agentnet_win_mac_18k.jsonl",
}
_META_FILE = "meta_data_merged.jsonl"

# meta_data_merged.jsonl의 "system" 필드 값을 windows/macos로 분류할 때 쓰는 키워드.
# 값 표기가 "Windows"/"Mac"/"macOS" 등으로 다를 수 있어서 소문자 부분일치로 비교.
_WINDOWS_KEYWORDS = ("windows", "win")
_MACOS_KEYWORDS = ("macos", "mac", "osx", "darwin")


def _ensure_hf_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit(
            "huggingface_hub가 필요합니다: pip install -U huggingface_hub 로 설치 후 다시 실행하세요."
        )


def download_jsonl(out_dir: str):
    """
    jsonl 3개(ubuntu, win_mac, meta_data_merged)만 받는다. 이미지 아카이브(zip/.zNN)는
    절대 건드리지 않음 - huggingface_hub의 allow_patterns로 명시적으로 제한한다.
    """
    _ensure_hf_hub()
    from huggingface_hub import snapshot_download

    os.makedirs(out_dir, exist_ok=True)
    print(f"[curate_agentnet] {_REPO_ID}에서 jsonl 파일만 받는 중 (이미지 아카이브 제외)...")
    path = snapshot_download(
        repo_id=_REPO_ID,
        repo_type="dataset",
        local_dir=out_dir,
        allow_patterns=["*.jsonl"],  # zip/.z01 등 이미지 아카이브는 패턴에 안 걸려서 자동 제외됨
    )
    print(f"[curate_agentnet] 완료: {path}")
    for name in list(_JSONL_FILES.values()) + [_META_FILE]:
        fp = os.path.join(out_dir, name)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"  - {name}: {size_mb:.1f} MB")
        else:
            print(f"  - {name}: 없음(다운로드 실패했거나 저장소에 파일명이 다를 수 있음)")


def _load_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def inspect(jsonl_dir: str, n: int = 3):
    """
    실제로 받은 jsonl의 최상위 키 목록과 샘플 레코드를 그대로 출력한다. README의
    데이터 구조 예시는 참고용일 뿐이고, windows/macos 구분 필드처럼 실제 배포본에만
    있는 필드는 이걸로 직접 확인해야 함 - curate 실행 전에 반드시 한 번 볼 것.
    """
    for tag, fname in _JSONL_FILES.items():
        path = os.path.join(jsonl_dir, fname)
        if not os.path.isfile(path):
            print(f"[{tag}] {fname} 없음 - 먼저 --download 실행하세요.")
            continue
        rows = _load_jsonl(path, limit=n)
        print(f"\n===== [{tag}] {fname} (샘플 {len(rows)}개) =====")
        if not rows:
            continue
        print(f"최상위 키: {sorted(rows[0].keys())}")
        if rows[0].get("traj"):
            step0 = rows[0]["traj"][0]
            print(f"traj[0] 키: {sorted(step0.keys())}")
            if isinstance(step0.get("value"), dict):
                print(f"traj[0]['value'] 키: {sorted(step0['value'].keys())}")
        print("--- 샘플 레코드(첫 번째, traj는 길어서 스텝 수만) ---")
        sample = dict(rows[0])
        if "traj" in sample:
            sample["traj"] = f"<{len(sample['traj'])}개 스텝, 생략>"
        print(json.dumps(sample, ensure_ascii=False, indent=2))

    meta_path = os.path.join(jsonl_dir, _META_FILE)
    if os.path.isfile(meta_path):
        rows = _load_jsonl(meta_path, limit=n)
        print(f"\n===== [meta] {_META_FILE} (샘플 {len(rows)}개) =====")
        if rows:
            print(f"최상위 키: {sorted(rows[0].keys())}")
            print(json.dumps(rows[0], ensure_ascii=False, indent=2))

        # (2026-08-07 추가) windows=17625/macos=0으로 나온 실측 이상 현상 진단용 -
        # "system" 필드에 실제로 어떤 값들이, 몇 개씩 들어있는지 전체를 다 세서 보여준다.
        # 표본 몇 개만 보고 분류 로직(_classify_win_mac)을 짰다가 실제 값 표기가 예상과
        # 다르면(예: "macOS"가 아니라 다른 표기) 전부 놓칠 수 있어서, 이건 전수 조사로 확인.
        print("\n--- meta_data_merged.jsonl 전체의 'system' 필드 값 분포 ---")
        all_rows = _load_jsonl(meta_path)
        counts = defaultdict(int)
        for r in all_rows:
            counts[r.get("system")] += 1
        for val, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {val!r}: {cnt}개")
    else:
        print(f"\n[meta] {_META_FILE} 없음 - 먼저 --download 실행하세요.")


def _load_meta_map(jsonl_dir: str) -> dict:
    """
    meta_data_merged.jsonl을 task_id -> {system, screen_width, screen_height, ...} 맵으로
    로드한다. win_mac 궤적의 windows/macos 분류와, 부가 메타(해상도 등)를 여기서 가져온다.
    """
    meta_path = os.path.join(jsonl_dir, _META_FILE)
    if not os.path.isfile(meta_path):
        return {}
    meta_map = {}
    for r in _load_jsonl(meta_path):
        tid = r.get("task_id")
        if tid:
            meta_map[tid] = r
    return meta_map


def _classify_win_mac(value: str):
    """
    (2026-08-07 버그 수정) 실측 결과 macOS 엔트리의 system 값이 "Darwin"으로 나오는데,
    "darwin" 문자열 안에 windows 키워드 "win"이 부분문자열로 들어있어서(dar**win**),
    windows를 먼저 체크하면 macOS가 전부 windows로 잘못 분류됐다(실측: windows=17625,
    macos=0으로 나온 원인). macOS 키워드(macos/mac/osx/darwin) 중에는 반대로 "win"이나
    "windows"를 부분문자열로 포함하는 게 하나도 없어서, macOS를 먼저 체크하는 것만으로
    충돌 없이 해결된다.
    """
    v = (value or "").lower()
    if any(k in v for k in _MACOS_KEYWORDS):
        return "macos"
    if any(k in v for k in _WINDOWS_KEYWORDS):
        return "windows"
    return None


def _quality_ok_traj(rec, min_alignment, min_efficiency):
    if not rec.get("task_completed"):
        return False
    align = rec.get("alignment_score")
    eff = rec.get("efficiency_score")
    if align is not None and align < min_alignment:
        return False
    if eff is not None and eff < min_efficiency:
        return False
    return True


def _quality_ok_step(step):
    v = step.get("value", {})
    if v.get("last_step_correct") is False:
        return False
    if v.get("last_step_redundant") is True:
        return False
    return True


def curate(
    jsonl_dir: str,
    out_path: str,
    total_budget: int,
    min_alignment: int,
    min_efficiency: int,
    seed: int,
):
    """
    ubuntu / windows / macos 세 그룹으로 최대한 비슷하게 나눠서 총 total_budget개
    스텝(이미지 1장 = 학습 예제 1개 기준)을 뽑는다. 이미지 자체는 안 받고, 어떤 이미지가
    필요한지(archive_group, image 파일명)만 manifest에 적어둔다.

    한 궤적(trajectory) 전체를 통째로 담거나 통째로 버림 - 스텝만 골라내면 history
    맥락이 끊겨서 planner 학습에 의미가 없어짐. 그래서 예산은 "궤적을 몇 개 쓸지"가
    아니라 "그 안의 유효한 스텝을 다 더하면 total_budget을 넘지 않는 선"으로 채운다.
    """
    random.seed(seed)

    ubuntu_path = os.path.join(jsonl_dir, _JSONL_FILES["ubuntu"])
    win_mac_path = os.path.join(jsonl_dir, _JSONL_FILES["win_mac"])
    if not os.path.isfile(ubuntu_path) or not os.path.isfile(win_mac_path):
        sys.exit("jsonl이 없습니다 - 먼저 --download를 실행하세요.")

    print("[curate_agentnet] jsonl 로딩 중 (win_mac 1.3GB라 시간이 좀 걸릴 수 있음)...")
    ubuntu_rows = _load_jsonl(ubuntu_path)
    win_mac_rows = _load_jsonl(win_mac_path)
    print(f"  ubuntu: {len(ubuntu_rows)}개 궤적, win_mac: {len(win_mac_rows)}개 궤적")

    meta_map = _load_meta_map(jsonl_dir)
    print(f"  meta_data_merged: {len(meta_map)}개 task_id 매핑 로드")
    if not meta_map:
        sys.exit(
            "meta_data_merged.jsonl이 없거나 비어 있습니다 - windows/macos 구분에 필요합니다. "
            "먼저 --download를 실행하세요."
        )

    # win_mac은 궤적 자체에 platform 필드가 없어서, meta_data_merged.jsonl의 "system" 필드를
    # task_id로 join해서 windows/macos를 분류한다(2026-08-07, --inspect 실측으로 확정).
    n_win = n_mac = n_unknown = 0
    for r in win_mac_rows:
        meta = meta_map.get(r["task_id"])
        cls = _classify_win_mac(meta.get("system") if meta else None)
        r["_platform"] = cls or "win_mac_unknown"
        r["_meta"] = meta
        if cls == "windows":
            n_win += 1
        elif cls == "macos":
            n_mac += 1
        else:
            n_unknown += 1
    print(f"  win_mac meta join으로 분류: windows={n_win}, macos={n_mac}, 미분류={n_unknown}")
    if n_unknown:
        print(f"  [참고] meta에서 못 찾았거나 system 값이 애매한 {n_unknown}개는 'win_mac_unknown'으로 분류돼 큐레이션에서 제외됩니다.")

    for r in ubuntu_rows:
        r["_platform"] = "ubuntu"
        r["_meta"] = meta_map.get(r["task_id"])

    # 궤적 품질 필터 + 셔플(플랫폼 그룹별로 독립적으로 섞어서 순서 편향 방지).
    # win_mac_unknown(meta 매칭 실패)은 windows/macos/ubuntu 3그룹 밸런싱에 넣지 않고 버린다 -
    # 어느 플랫폼인지 모르는 데이터를 "균등 분배" 대상에 넣으면 밸런싱 자체가 무의미해짐.
    groups = defaultdict(list)
    for r in ubuntu_rows + win_mac_rows:
        if r["_platform"] == "win_mac_unknown":
            continue
        if _quality_ok_traj(r, min_alignment, min_efficiency):
            groups[r["_platform"]].append(r)
    for g in groups.values():
        random.shuffle(g)

    platform_names = sorted(groups.keys())
    print(f"[curate_agentnet] 품질 필터 통과 궤적 수: " + ", ".join(f"{p}={len(groups[p])}" for p in platform_names))
    if not platform_names:
        sys.exit("품질 필터를 통과한 궤적이 하나도 없습니다 - min_alignment/min_efficiency를 낮춰보세요.")

    per_group_budget = total_budget // len(platform_names)
    print(f"[curate_agentnet] 그룹당 목표 스텝 수: 약 {per_group_budget}개 (총 {total_budget}개 목표, {len(platform_names)}개 그룹)")

    manifest = []
    stats = defaultdict(lambda: {"trajectories": 0, "steps": 0})
    for platform in platform_names:
        used_steps = 0
        for rec in groups[platform]:
            if used_steps >= per_group_budget:
                break
            valid_steps = [s for s in rec.get("traj", []) if _quality_ok_step(s)]
            if not valid_steps:
                continue
            meta = rec.get("_meta") or {}
            for s in valid_steps:
                manifest.append(
                    {
                        "task_id": rec["task_id"],
                        "platform": platform,
                        "archive_group": "ubuntu" if platform == "ubuntu" else "win_mac",
                        "instruction": rec.get("natural_language_task") or rec.get("instruction"),
                        "step_index": s.get("index"),
                        "image": s.get("image"),
                        "observation": s.get("value", {}).get("observation"),
                        "thought": s.get("value", {}).get("thought"),
                        "action": s.get("value", {}).get("action"),
                        "code": s.get("value", {}).get("code"),
                        "screen_width": meta.get("screen_width"),
                        "screen_height": meta.get("screen_height"),
                    }
                )
            used_steps += len(valid_steps)
            stats[platform]["trajectories"] += 1
            stats[platform]["steps"] += len(valid_steps)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[curate_agentnet] manifest 저장: {out_path} (총 {len(manifest)}개 스텝)")
    for platform in platform_names:
        s = stats[platform]
        print(f"  {platform}: 궤적 {s['trajectories']}개, 스텝(이미지) {s['steps']}개")
    print(
        "\n다음 단계: 이 manifest의 'archive_group'+'image' 조합이 실제로 필요한 이미지 "
        "목록입니다. 아직 이미지는 안 받았습니다 - 이 목록을 보고 이미지 추출 스크립트를 "
        "다음에 작성하면 됩니다 (멀티파트 zip에서 필요한 파일만 골라 받는 로직 필요)."
    )


def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl_dir", default="./agentnet_raw", help="jsonl을 받을/읽을 폴더")
    ap.add_argument("--download", action="store_true", help="jsonl 3개만 다운로드 (이미지 제외)")
    ap.add_argument("--inspect", action="store_true", help="실제 필드 구조를 샘플로 출력")
    ap.add_argument("--curate", action="store_true", help="큐레이션 + manifest 작성 (이미지는 안 받음)")
    ap.add_argument("--out", default="data/processed/planner_agentnet_manifest.jsonl", help="manifest 저장 경로")
    ap.add_argument("--total_budget", type=int, default=4000, help="총 스텝(이미지) 예산 - 근거는 파일 상단 docstring 참고")
    ap.add_argument("--min_alignment", type=int, default=5, help="이 값 미만인 궤적은 제외 (AgentNet alignment_score)")
    ap.add_argument("--min_efficiency", type=int, default=5, help="이 값 미만인 궤적은 제외 (AgentNet efficiency_score)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (args.download or args.inspect or args.curate):
        ap.error("--download, --inspect, --curate 중 최소 하나는 지정해야 합니다.")

    if args.download:
        download_jsonl(args.jsonl_dir)
    if args.inspect:
        inspect(args.jsonl_dir)
    if args.curate:
        curate(
            args.jsonl_dir,
            args.out,
            args.total_budget,
            args.min_alignment,
            args.min_efficiency,
            args.seed,
        )


if __name__ == "__main__":
    _cli()
