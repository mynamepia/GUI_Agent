"""
순수 파이썬 유틸 (torch 불필요).
evaluation.py, dataset_prep.py, test.py, data_utils.py가 공용으로 사용.
평가만 하고 싶을 때(torch 설치 없이) evaluation.py가 무겁게 안 돌아가도록
Dataset(torch 필요) 클래스는 data_utils.py로 분리해뒀다.
"""

import json
import re

PROMPT_TEMPLATE = (
    "You are a GUI grounding agent. Given a screenshot and an instruction, "
    "output the pixel location to click as a single point in the format "
    "(x,y), where x and y are integers from 0 to 1000 representing the "
    "relative position on the image (0,0 = top-left, 1000,1000 = bottom-right).\n"
    "Instruction: {instruction}"
)

POINT_RE = re.compile(r"\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?")


def load_jsonl(path: str):
    """
    jsonl을 읽어서 dict 리스트로 반환.

    image_path는 윈도우(gpu-work)에서 데이터셋을 만들 때 "\\" 구분자로 박혀서 저장되는데,
    리눅스/맥에서는 "\\"가 경로 구분자가 아니라 그냥 파일명 문자로 취급돼서 파일을 못 찾는
    문제가 있었음(FileNotFoundError). "/"는 윈도우에서도 정상 동작하는 경로 구분자라서,
    여기서 항상 "/"로 정규화해두면 어느 OS에서 읽어도 안전하다.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if isinstance(rec.get("image_path"), str):
                    rec["image_path"] = rec["image_path"].replace("\\", "/")
                records.append(rec)
    return records


def point_to_norm1000(point, resolution):
    x, y = point
    w, h = resolution
    nx = round(x / w * 1000)
    ny = round(y / h * 1000)
    return max(0, min(1000, nx)), max(0, min(1000, ny))


def norm1000_to_point(norm_point, resolution):
    nx, ny = norm_point
    w, h = resolution
    return nx / 1000 * w, ny / 1000 * h


def parse_point_from_text(text: str):
    """모델 생성 텍스트에서 (x,y) 형태의 첫 좌표를 파싱. 실패시 None."""
    m = POINT_RE.search(text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def build_target_text(point, resolution):
    nx, ny = point_to_norm1000(point, resolution)
    return f"({nx},{ny})"