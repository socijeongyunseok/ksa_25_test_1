# -*- coding: utf-8 -*-

# data_load.py
import pandas as pd
from pathlib import Path
from config import DATA_DIR

def load_dialog_csv(fname1: str, fname2: str) -> pd.DataFrame:
    """
    두 CSV(n1.csv, n2.csv) → 대사 스크립트 DataFrame 반환
    필요한 열: Script(본문), Speaker, Modification
    """
    f1, f2 = DATA_DIR / fname1, DATA_DIR / fname2
    df = pd.concat([pd.read_csv(f1), pd.read_csv(f2)], ignore_index=True)
    df = df[df['Modification'] == '수정결과'].dropna(subset=['Speaker'])
    df.reset_index(drop=True, inplace=True)
    return df
