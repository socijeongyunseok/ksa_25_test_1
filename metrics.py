"""
metrics.py
-----------
토픽/그래프/클러스터 품질 지표 모음 + (NEW) 토픽 코히런스·문서 명확도
"""

# ----------------------------------------------------------------------
# 0) 기본 import
# ----------------------------------------------------------------------
from __future__ import annotations
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1) (기존) 그래프·클러스터 지표 함수
#     ※ 이 부분은 여러분의 저장소에 이미 구현돼 있었던 내용입니다.
#       기존 코드가 있다면 그대로 두세요. 아래는 “동작만 보장”용
#       최소 구현(stub) 예시입니다. 기존 코드가 더 낫다면 덮어쓰지 마세요.
# ----------------------------------------------------------------------
def cluster_metrics(labels: np.ndarray) -> Dict[str, Any]:
    """
    클러스터 라벨 배열 → n_clusters, silhouette 등 계산
    (기존 구현이 있으면 그대로 유지)
    """
    from sklearn.metrics import silhouette_score
    valid = labels >= 0
    if valid.sum() == 0:
        return {"n_clusters": 0, "silhouette": -1}
    return {
        "n_clusters": int(len(set(labels[valid]))),
        "silhouette": float(silhouette_score(
            np.arange(len(labels))[valid].reshape(-1, 1),
            labels[valid]
        ))
    }


def graph_metrics(G) -> Dict[str, Any]:
    """
    networkx Graph → density, #nodes, #edges 반환
    (기존 구현이 있으면 그대로 유지)
    """
    import networkx as nx
    return {
        "graph_density": float(nx.density(G)),
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
    }

# ----------------------------------------------------------------------
# 2) (신규) 토픽 코히런스
# ----------------------------------------------------------------------
try:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
except ImportError:
    # Colab 런타임 첫 실행 직후 gensim import 문제 방지
    pass


def compute_coherence(
    topic_model,
    docs: List[str],
    topk: int = 10,
    measure: str = "c_v",
) -> float:
    """
    gensim.CoherenceModel 로 corpus-level coherence 계산.
    Parameters
    ----------
    topic_model : BERTopic
    docs        : 토큰화 / 띄어쓰기 분리된 문서 문자열 리스트
    topk        : 각 토픽당 상위 단어 수
    measure     : "c_v", "u_mass", "c_npmi", "c_uci"
    """
    topic_dict = topic_model.get_topics()
    topic_words = [[w for w, _ in ws[:topk]] for _, ws in topic_dict.items()]

    tokenized = [d.split() for d in docs]
    dictionary = Dictionary(tokenized)
    bow_corpus = [dictionary.doc2bow(t) for t in tokenized]

    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenized,
        corpus=bow_corpus,
        dictionary=dictionary,
        coherence=measure,
    )
    return cm.get_coherence()

# ----------------------------------------------------------------------
# 3) (신규) 문서 명확도(clarity)
# ----------------------------------------------------------------------
def compute_clarity(prob_matrix: np.ndarray) -> pd.DataFrame:
    """
    확률 행렬 → “명확도” (p1 - p2) 계산
    Returns
    -------
    DataFrame[doc_id, clarity, topic_top1, topic_top2, p_top1, p_top2]
    """
    order = np.argsort(-prob_matrix, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    p1 = prob_matrix[np.arange(len(prob_matrix)), top1]
    p2 = prob_matrix[np.arange(len(prob_matrix)), top2]
    return pd.DataFrame(
        {
            "doc_id": np.arange(len(prob_matrix)),
            "clarity": p1 - p2,
            "topic_top1": top1,
            "topic_top2": top2,
            "p_top1": p1,
            "p_top2": p2,
        }
    )
