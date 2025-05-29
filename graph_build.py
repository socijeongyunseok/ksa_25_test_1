# -*- coding: utf-8 -*-

# graph_build.py
import numpy as np, networkx as nx
from collections import Counter
from config import N_TOP_WORDS, PMI_TOP_P

# ###다음과 같이 추가했음###
from config import GRAPH_METRIC, JACCARD_THRESH  # 그래프 생성 metric 및 임계값 옵션
# ###위와 같이 추가했음###

def ppmi_matrix(token_lists):
    # 상위 N개 단어 선정
    wc = Counter()
    for toks in token_lists:
        wc.update(set(toks))
    vocab = [w for w,_ in wc.most_common(N_TOP_WORDS)]
    idx   = {w:i for i,w in enumerate(vocab)}
    M = len(vocab)
    co_mat = np.zeros((M,M), dtype=np.int32)
    for toks in token_lists:
        uniq = list(set([t for t in toks if t in idx]))
        for i,a in enumerate(uniq):
            for b in uniq[i+1:]:
                co_mat[idx[a], idx[b]] += 1
                co_mat[idx[b], idx[a]] += 1
    # PPMI 계산
    col_sum = co_mat.sum(axis=0, keepdims=True)
    row_sum = co_mat.sum(axis=1, keepdims=True)
    total   = col_sum.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2((co_mat * total) / (row_sum * col_sum + 1e-9))
    pmi[np.isnan(pmi) | (pmi < 0)] = 0
    # 상위 p% edge 보존
    thresh = np.quantile(pmi[pmi>0], 1-PMI_TOP_P)
    pmi[pmi < thresh] = 0
    return vocab, pmi

# ###다음과 같이 추가했음###
def jaccard_matrix(token_lists):
    # Jaccard 유사도 계산 행렬
    wc = Counter()
    for toks in token_lists:
        wc.update(set(toks))
    vocab = [w for w,_ in wc.most_common(N_TOP_WORDS)]
    idx   = {w:i for i,w in enumerate(vocab)}
    M = len(vocab)
    mat = np.zeros((M,M), dtype=float)
    # 단어-문서 매핑
    docs_containing = {w:set() for w in vocab}
    for doc_id, toks in enumerate(token_lists):
        for t in set(toks):
            if t in idx:
                docs_containing[t].add(doc_id)
    for i,a in enumerate(vocab):
        for j,b in enumerate(vocab):
            if i < j:
                inter = len(docs_containing[a] & docs_containing[b])
                uni   = len(docs_containing[a] | docs_containing[b])
                mat[i,j] = mat[j,i] = inter/uni if uni>0 else 0
    return vocab, mat
# ###위와 같이 추가했음###

def build_graph(token_lists):
    # ###다음과 같이 추가했음###
    # 그래프 생성 metric 선택
    if GRAPH_METRIC == 'ppmi':
        vocab, mat = ppmi_matrix(token_lists)
    else:
        vocab, mat = jaccard_matrix(token_lists)
    # ###위와 같이 추가했음###

    G = nx.Graph()
    for i,w in enumerate(vocab):
        G.add_node(w)
    for i in range(len(vocab)):
        for j in range(i+1, len(vocab)):
            # ###다음과 같이 추가했음###
            # Jaccard metric인 경우 threshold 비교
            if GRAPH_METRIC == 'jaccard':
                if mat[i,j] >= JACCARD_THRESH:
                    G.add_edge(vocab[i], vocab[j], weight=float(mat[i,j]))
            # PPMI 기존 방식
            elif mat[i,j] > 0:
                G.add_edge(vocab[i], vocab[j], weight=float(mat[i,j]))
            # ###위와 같이 추가했음###
    return G

def build_cooccurrence_graph(token_lists, topics=None):
    """파이프라인 호환용 래퍼 – topics 인자는 현재 사용하지 않음"""
    return build_graph(token_lists)
