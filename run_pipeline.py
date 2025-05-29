# -*- coding: utf-8 -*-

# run_pipeline.py

import pprint
import json
from pathlib import Path

# DATA_DIR 사용
from config import DATA_DIR
from pipeline import run

# 내가 이런 짓까지 할 줄은 몰랐다
from experiments import compute_silhouette

def find_csvs():
    # data/ 폴더에서 n1.csv, n2.csv 자동 검색
    p = Path(DATA_DIR)
    csvs = sorted(p.glob("n*.csv"))
    if len(csvs) < 2:
        raise FileNotFoundError(f"{DATA_DIR} 폴더에 n1.csv, n2.csv 두 파일이 필요합니다.")
    # 첫 두 개를 사용
    return csvs[0].name, csvs[1].name

if __name__ == "__main__":
    # 인자 없이 자동 파일 검색
    csv1, csv2 = find_csvs()
    print(f"▶ 로드할 파일: data/{csv1}, data/{csv2}")
    df, metrics = run(csv1, csv2)
    pprint.pprint(metrics, compact=True)
    # 결과 저장
    out_path = Path.cwd() / "pipeline_output.csv"
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 파이프라인 완료 — 결과는 {out_path} / exports/ 폴더 확인")


"""
import argparse, pprint, json
from pipeline import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv1", default="n1.csv")
    p.add_argument("--csv2", default="n2.csv")
    args = p.parse_args()
    df, metrics = run(args.csv1, args.csv2)
    pprint.pprint(metrics, compact=True)
    df.to_csv("pipeline_output.csv", index=False, encoding='utf-8-sig')
    print("\n✓ 파이프라인 완료 — 결과는 pipeline_output.csv / exports/ 폴더 확인")
"""
