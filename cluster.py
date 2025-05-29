# -*- coding: utf-8 -*-

# cluster.py
from umap import UMAP
import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score

from config import (
    UMAP_N_COMPONENT,
    # UMAP_N_NEIGHBOR,
    UMAP_MIN_DIST_LIST,      # ← pull in the list of min_dist values
    UMAP_N_NEIGHBORS_LIST,   # ← pull in the list of n_neighbors values
    HDBSCAN_MIN_CLUS,
    HDBSCAN_MIN_SAMP,
    RANDOM_SEED
)


def reduce_dim(vecs: np.ndarray) -> np.ndarray:
    # 원본
    # umap = UMAP(n_components=UMAP_N_COMPONENT, n_neighbors=UMAP_N_NEIGHBOR,
    #             metric='cosine', random_state=RANDOM_SEED)
    # return umap.fit_transform(vecs)

    # ###다음과 같이 고쳤음###
    # 이제 여러 UMAP 하이퍼파라미터 조합을 시도합니다.
    best_emb, best_score = None, -np.inf
    for n_nb in UMAP_N_NEIGHBORS_LIST:
        for m_dist in UMAP_MIN_DIST_LIST:
            umap = UMAP(
                n_components=UMAP_N_COMPONENT,
                n_neighbors=n_nb,
                min_dist=m_dist,
                metric='cosine',
                random_state=RANDOM_SEED
            )
            emb = umap.fit_transform(vecs)
            # 간단히 첫 두 차원으로만 클러스터링 테스트
            labels = hdbscan.HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUS[0],
                min_samples=HDBSCAN_MIN_SAMP[0],
                metric='euclidean',
                cluster_selection_method='eom'
            ).fit_predict(emb)
            if len(set(labels)) > 1:
                score = silhouette_score(emb, labels)
                if score > best_score:
                    best_score, best_emb = score, emb
    return best_emb
    # ###위와 같이 고쳤음###

def best_hdbscan(X_low) -> (hdbscan.HDBSCAN, float):
    best_score, best_model = -1, None
    for mcs in HDBSCAN_MIN_CLUS:
        for ms in HDBSCAN_MIN_SAMP:
            # 원본
            # model = hdbscan.HDBSCAN(min_cluster_size=mcs,
            #                         min_samples=ms,
            #                         metric='euclidean',
            #                         cluster_selection_method='eom',
            #                         prediction_data=True)
            # labels = model.fit_predict(X_low)
            # if len(set(labels)) <= 1 or (labels >= 0).sum() < 10:
            #     continue
            # score = silhouette_score(X_low, labels)
            # if score > best_score:
            #     best_score, best_model = score, model

            # ###다음과 같이 고쳤음###
            # HDBSCAN에 cluster_selection_epsilon 옵션을 추가하고, 유사도 기반 metric 사용
            model = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric='euclidean',
                cluster_selection_method='eom',
                cluster_selection_epsilon=0.01,  # 추가된 하이퍼파라미터
                prediction_data=True
            )
            labels = model.fit_predict(X_low)
            # 최소 군집 크기 조건 강화
            if len(set(labels)) <= 1 or (labels >= 0).sum() < 20:
                continue
            score = silhouette_score(X_low, labels)
            if score > best_score:
                best_score, best_model = score, model
    return best_model, best_score
    # ###위와 같이 고쳤음###
