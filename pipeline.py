"""
pipeline.py
------------
CSV → 토픽 → 그래프 → 지표·파일 저장
(컬럼명을 자동 감지하도록 수정)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx

# ───────────────────────────────────────────────────────────────
# 1) 결과 폴더 준비
# ───────────────────────────────────────────────────────────────
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)          # ← 새로 추가된 줄

# 2) 나머지 모듈 import

from topic_model import build_topic_model
from graph_build import build_cooccurrence_graph
from metrics import (
    cluster_metrics,
    graph_metrics,
    compute_coherence,
    compute_clarity,
)

# --------------------------  자동 컬럼 매핑  --------------------------
TITLE_CANDIDATES   = ["title", "title.title", "Title", "Speaker"]
CONTENT_CANDIDATES = ["content", "content.text", "Content", "body", "Script"]

def _pick_first(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            print(f"⚠️  fallback: using '{c}' column")
            return c
    raise KeyError(f"No text-like column found — columns={df.columns.tolist()}")
# --------------------------  메인 파이프라인  --------------------------
def run(posts_csv: str, comments_csv: str):
    # 1) 경로 처리
    p_posts = Path(posts_csv)
    p_comments = Path(comments_csv)
    if not p_posts.is_file():
        p_posts   = Path("data") / p_posts
        p_comments = Path("data") / p_comments

    # 2) 로드
    df_posts    = pd.read_csv(p_posts)
    df_comments = pd.read_csv(p_comments)

    # 3) 텍스트 결합
    title_col   = _pick_first(df_posts, TITLE_CANDIDATES)
    content_col = _pick_first(df_posts, CONTENT_CANDIDATES)
    texts = (
        df_posts[title_col].fillna("") + " " + df_posts[content_col].fillna("")
    ).tolist()

    # 4) 토픽 모델
    topic_model, topics, probs = build_topic_model(texts)

    # 5) 그래프
    G = build_cooccurrence_graph(texts, topics)
    nx.write_gexf(G, EXPORT_DIR / "graph_ppmi.gexf")

    # 6) 지표
    metrics = {
        **graph_metrics(G),
        **cluster_metrics(np.array(topics)),
        "coherence": float(compute_coherence(topic_model, texts)),
    }
    clarity_df = compute_clarity(probs)
    clarity_df.to_csv(EXPORT_DIR / "doc_clarity.csv", index=False)
    metrics["clarity_mean"]   = float(clarity_df.clarity.mean())
    metrics["clarity_median"] = float(clarity_df.clarity.median())

    # 7) 저장
    df_posts.to_csv(EXPORT_DIR / "pipeline_output.csv", index=False)
    topic_model.get_topic_info().to_csv(EXPORT_DIR / "topic_summary.csv", index=False)

    return df_posts, metrics


if __name__ == "__main__":
    import sys, json
    posts_csv, comments_csv = sys.argv[1:]
    _, m = run(posts_csv, comments_csv)
    print(json.dumps(m, ensure_ascii=False, indent=2))
