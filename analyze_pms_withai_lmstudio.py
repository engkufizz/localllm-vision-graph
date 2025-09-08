import os
import uuid
import base64
from pathlib import Path
from typing import List

import requests
import pandas as pd

API_BASE = os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234").rstrip("/")
API_KEY = os.getenv("LMSTUDIO_API_KEY", "").strip()
MODEL = os.getenv("LMSTUDIO_MODEL", "local")

DIR = os.getenv("DIR", r"D:\Path\To\PMS")
OUT_XLSX = os.getenv("OUT_XLSX", str(Path(__file__).resolve().parent / "PMS_Results.xlsx"))
TIMEOUT = int(os.getenv("TIMEOUT", "300"))

def headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h

def to_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def classify_graph(path: str) -> str:
    data_url = to_data_url(path)
    url = f"{API_BASE}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a vision assistant. Reply with one word only: Normal or Abnormal."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyse this graph. Reply with one word only: Normal or Abnormal."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    r = requests.post(url, headers=headers(), json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    ans = r.json()["choices"][0]["message"]["content"].strip().lower()
    if ans == "normal":
        return "Normal"
    if ans == "abnormal":
        return "Abnormal"
    # Fallback: capitalise first letter; still keep it concise
    return ans.split()[0].capitalize() if ans else "Error"

def list_images(dir_path: str) -> List[Path]:
    p = Path(dir_path)
    return sorted([*p.glob("*.png"), *p.glob("*.jpg"), *p.glob("*.jpeg")])

def process_directory(dir_path: str, out_xlsx: str) -> str:
    files = list_images(dir_path)
    rows = []
    for fp in files:
        try:
            result = classify_graph(str(fp))
        except Exception as e:
            result = f"Error: {e.__class__.__name__}"
        rows.append({"Graph Name": fp.stem, "Result": result})
    pd.DataFrame(rows).to_excel(out_xlsx, index=False)
    return out_xlsx

if __name__ == "__main__":
    out = process_directory(DIR, OUT_XLSX)
    print("Saved to:", out)
