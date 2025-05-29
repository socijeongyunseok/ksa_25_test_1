#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
experiments.py ─────────────────────────────────────────────────────────────
BERTopic + UMAP + HDBSCAN 하이퍼파라미터 그리드 탐색 스크립트
Usage:
    python experiments.py \
        --data data/n1.csv data/n2.csv \
        --grid_results exports/grid_results.csv
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# ────────────────────────────────────────────────────────────────────────────
# 탐색할 하이퍼파라미터 그리드
GRID = dict(
    min_cluster_size=[25, 30, 35, 40],
    min_samples=[5, 10, 15],
    n_neighbors=[10, 15, 20],
    min_dist=[0.05, 0.1],
)

# Vectorizer 기본 설정
VECTORIZER_KWARGS = dict(
    min_df=1,
    max_df=0.95,
    ngram_range=(1, 1),
    stop_words=None,
)

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
RANDOM_STATE = 42
EXPORTS_DIR = "exports"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_texts(csv_paths):
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if 'Modification' in df.columns:
            df = df[df['Modification'] == '수정결과']
        if 'Script' in df.columns:
            frames.append(df['Script'].dropna())
        else:
            raise KeyError("CSV에 'Script' 열이 없습니다.")
    texts = pd.concat(frames, ignore_index=True).astype(str).tolist()
    return texts

def compute_silhouette(embeddings, labels):
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return float('nan')
    return silhouette_score(embeddings[mask], labels[mask])

def compute_coherence(topics_dict, texts):
    tokenized = [doc.split() for doc in texts]
    id2word = Dictionary(tokenized)
    corpus = [id2word.doc2bow(tokens) for tokens in tokenized]
    topic_word_lists = []
    for tid, word_scores in topics_dict.items():
        if tid == -1:
            continue
        words = [w for w, _ in word_scores]
        topic_word_lists.append(words)
    cm = CoherenceModel(
        topics=topic_word_lists,
        texts=tokenized,
        corpus=corpus,
        dictionary=id2word,
        coherence='c_v',
        processes=4,
    )
    return cm.get_coherence()

def iterate_grid():
    keys, vals = zip(*GRID.items())
    return [dict(zip(keys, v)) for v in itertools.product(*vals)]

def run_single(texts, params, _unused):
    umap_model = UMAP(
        n_neighbors=params['n_neighbors'],
        n_components=5,
        min_dist=params['min_dist'],
        metric='cosine',
        random_state=RANDOM_STATE,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(**VECTORIZER_KWARGS)
    topic_model = BERTopic(
        embedding_model=EMBED_MODEL_NAME,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        min_topic_size=30,
        verbose=False,
    )
    topics, _probs = topic_model.fit_transform(texts)
    emb = topic_model.umap_model.embedding_
    sil = compute_silhouette(emb, np.array(topics))
    topics_dict = topic_model.get_topics()
    coh = compute_coherence(topics_dict, texts)
    k = len({t for t in topics if t != -1})
    return coh, sil, k

def main(args):
    ensure_dir(EXPORTS_DIR)
    texts = load_texts(args.data)
    grid_list = iterate_grid()
    out_rows = []
    best = {'coherence': -np.inf, 'silhouette': -np.inf}
    t0 = time.time()

    for i, pr in enumerate(grid_list, 1):
        coh, sil, k = run_single(texts, pr, args.n_jobs)
        row = {**pr, 'coherence': coh, 'silhouette': sil, 'n_clusters': k}
        out_rows.append(row)
        print(f"[{i}/{len(grid_list)}] k={k} coh={coh:.3f} sil={sil:.3f} params={pr}")
        if coh > best['coherence'] or (np.isclose(coh, best['coherence']) and sil > best['silhouette']):
            best.update(row)

    df = pd.DataFrame(out_rows)
    df.to_csv(args.grid_results, index=False, encoding='utf-8-sig')
    print("\n🏆 BEST")
    print(f"coh={best['coherence']:.3f}, sil={best['silhouette']:.3f}, k={best['n_clusters']}")
    print("params:", {k: best[k] for k in GRID.keys()})
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs=2, required=True, help='CSV 두 개 경로')
    parser.add_argument('--grid_results', default=f'{EXPORTS_DIR}/grid_results.csv')
    parser.add_argument('--n_jobs', type=int, default=-1, help='unused for v0.17')
    main(parser.parse_args())
