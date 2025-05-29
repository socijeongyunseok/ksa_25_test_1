# -*- coding: utf-8 -*-

# tokenizer.py
"""
현재는 preprocess.tokenize 만 사용하므로 별도 래퍼만 유지
향후 다른 형태소기(예: Mecab)로 바꿀 때 이곳만 수정
"""
from preprocess import tokenize as kiwi_tokenize
tokenize = kiwi_tokenize
