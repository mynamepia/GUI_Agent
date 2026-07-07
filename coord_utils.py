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
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
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