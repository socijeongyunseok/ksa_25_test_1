# -*- coding: utf-8 -*-
# ============================================================
# data_load.py
# ------------------------------------------------------------
import pandas as pd
from pathlib import Path
from config import DATA_DIR

def load_dialog_csv(*fnames: str) -> pd.DataFrame:
    """하나 또는 여러 CSV를 읽어 (timestamp, text) DataFrame 반환."""
    frames = [pd.read_csv(DATA_DIR / f) for f in fnames]
    df = pd.concat(frames, ignore_index=True)
    # 오직 '수정결과'만 필터링
    df = df[df["Modification"] == "수정결과"].copy()
    df["timestamp"] = df.index.astype(int)              # 시계열 인덱스
    df["text"] = df["Script"].astype(str).str.strip()  # 본문 텍스트
    return df[["timestamp", "text"]]
