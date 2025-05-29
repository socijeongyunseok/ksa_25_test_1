#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
experiments.py ─────────────────────────────────────────────────────────────
BERTopic + UMAP + HDBSCAN 하이퍼파라미터 그리드 탐색 스크립트
(한국어 전처리, TF‑IDF, 대체 Coherence 계산)

Usage:
python experiments.py \
    --data data/n1.csv data/n2.csv \
    --grid_results exports/grid_results.csv
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from kiwipiepy import Kiwi

# ────────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터 그리드
GRID = {
    "embedding_model": [
        "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
    ],
    "ngram_range": [(1, 1), (1, 2)],
    "min_df": [1, 2],
    "min_cluster_size": [15, 20, 25],
    "min_samples": [5, 10],
    "n_neighbors": [15, 20],
    "min_dist": [0.05],
}
RANDOM_STATE = 42
EXPORTS_DIR = "exports"

kiwi = Kiwi()  # 형태소 분석기

# ────────────────────────────────────────────────────────────────────────────
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_texts(csv_paths):
    """
    CSV 목록에서 'Modification'=='수정결과'인 행의 'Script' 열을 가져와
    NaN/None/빈 문자열 제거 후 순수한 str 리스트로 반환
    """
    texts = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "Modification" in df.columns:
            df = df[df["Modification"] == "수정결과"]
        if "Script" not in df.columns:
            raise KeyError("CSV에 'Script' 열이 없습니다.")
        series = df["Script"].dropna().astype(str)
        for s in series.tolist():
            if s and isinstance(s, str):
                texts.append(s)
    return texts


def preprocess_korean(texts):
    """
    Korean morphological filtering: 명사(NNG, NNP)와 형용사(VA)만 추출
    """
    processed = []
    for doc in texts:
        if not isinstance(doc, str):
            continue
        token_lists = kiwi.tokenize(doc)
        tokens = [tok for sent in token_lists for tok in sent]
        words = [t.form for t in tokens if getattr(t, "tag", "") in {"NNG", "NNP", "VA"}]
        if words:
            processed.append(" ".join(words))
    return processed


def silhouette_safe(emb, labels):
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return float("nan")
    return silhouette_score(emb[mask], labels[mask])


def coherence_alt(texts, topics, top_n=10):
    """대체 coherence: top-n 단어 쌍 간 평균 log2 조건부 확률"""
    try:
        vect = CountVectorizer(binary=True)
        X = vect.fit_transform(texts)
        vocab = {t: i for i, t in enumerate(vect.get_feature_names_out())}
        scores = []
        for tid, ws in topics.items():
            if tid == -1:
                continue
            words = [w for w, _ in ws][:top_n]
            idxs = [vocab[w] for w in words if w in vocab]
            if len(idxs) < 2:
                continue
            for i in range(len(idxs)):
                for j in range(i):
                    wi, wj = idxs[i], idxs[j]
                    df_i = X[:, wi].sum()
                    df_ij = (X[:, wi].multiply(X[:, wj])).sum()
                    scores.append(np.log2((df_ij + 1) / (df_i + 1)))
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def iterate_grid():
    keys, vals = zip(*GRID.items())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def run_single(raw_texts, params):
    # Debug: show first raw_texts types
    print("DEBUG raw_texts types:", [(rt, type(rt)) for rt in raw_texts[:5]])
    # 1) Load & preprocess
    cleaned_texts = preprocess_korean(raw_texts)
    # Ensure all elements are strings
    docs = []
    for d in cleaned_texts:
        if isinstance(d, str) and d.strip():
            docs.append(d)
        else:
            print(f"DEBUG skipped doc: {d} ({type(d)})")
 for d in cleaned_texts if isinstance(d, str) and d.strip()]

    # 2) TF-IDF & embedding
    vect = TfidfVectorizer(
        ngram_range=params["ngram_range"],
        min_df=params["min_df"],
        max_df=0.95,
    )
    umap_model = UMAP(
        n_neighbors=params["n_neighbors"],
        n_components=5,
        min_dist=params["min_dist"],
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=params["embedding_model"],
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vect,
        calculate_probabilities=True,
        min_topic_size=30,
        verbose=False,
    )

    # 3) Fit & metrics
    topics, _ = topic_model.fit_transform(docs)
    sil = silhouette_safe(topic_model.umap_model.embedding_, np.array(topics))
    coh = coherence_alt(docs, topic_model.get_topics())
    k = len({t for t in topics if t != -1})
    return coh, sil, k


# ────────────────────────────────────────────────────────────────────────────
def main(args):
    ensure_dir(EXPORTS_DIR)
    texts = load_texts(args.data)
    best = {"coherence": -np.inf, "silhouette": -np.inf, "n_clusters": 0, "params": {}}
    rows = []
    grid = list(iterate_grid())
    start = time.time()
    total = len(grid)

    for i, prm in enumerate(grid, start=1):
        coh, sil, k = run_single(texts, prm)
        rows.append({**prm, "coherence": coh, "silhouette": sil, "n_clusters": k})
        print(f"[{i}/{total}] k={k} coherence={coh:.3f} sil={sil:.3f} params={prm}")
        if coh > best["coherence"] or (np.isclose(coh, best["coherence"]) and sil > best["silhouette"]):
            best.update({"coherence": coh, "silhouette": sil, "n_clusters": k, "params": prm})

    pd.DataFrame(rows).to_csv(args.grid_results, index=False, encoding="utf-8-sig")
    print("\n🏆 BEST ->", best)
    print(f"Elapsed: {(time.time() - start)/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs=2, required=True)
    ap.add_argument("--grid_results", default=f"{EXPORTS_DIR}/grid_results.csv")
    main(ap.parse_args())
