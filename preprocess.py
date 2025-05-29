# -*- coding: utf-8 -*-

# preprocess.py
import re
from typing import List
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

# MeCab 설치 여부에 따라 표제어 추출 지원 여부 결정
try:
    from konlpy.tag import Mecab
    _mecab = Mecab()
    USE_MECAB = True
except Exception:
    _mecab = None
    USE_MECAB = False

# 형태소 분석기 초기화
_kiwi = Kiwi()

# Stopwords 객체에서 실제 리스트를 꺼내오기
_sw_obj = Stopwords()
if hasattr(_sw_obj, 'stopwords'):
    _sw_list = _sw_obj.stopwords
elif hasattr(_sw_obj, 'words'):
    _sw_list = _sw_obj.words
else:
    _sw_list = []

# 사용자 정의 추가 불용어
_extra = {"안녕", "안녕하세요", "안녕하십니까",
          "투쟁으로", "인사드리겠습니다",
          "그러나", "그리고", "그러므로", "하지만",
          "따라서", "아니", "그래", "저", "이",
          "차빼라", "열어라", "감사합니다", "감사",
          "우리","저는","있습니다","제가",
          "지금","여러분","함께"}

# 다음은 더 이상 사용하지 않음
# _stop = set(_sw_list) | _extra

# 불용어 세트를 강화: 기본 불용어 + 사용자 정의 + 형태소 분석 불용어
_additional = {token for token in _sw_list if len(token) > 1}
_stop = set(_sw_list) | _extra | _additional

# 숫자·영문·한자 제거용 정규식
_re_hanja_eng = re.compile(r'[A-Za-z0-9一-龥]+')

# 허용 품사 확장
_allowed_pos = {'NNG', 'NNP', 'SL', 'VV', 'VA'}


def tokenize(sent: str) -> List[str]:
    """
    한국어 문장 → 의미 토큰 리스트
    ※ 품사: NNG, NNP, SL(외래어) + 동사(VV), 형용사(VA)
    """
    # 한자·영문·숫자 모두 공백으로 치환
    sent = _re_hanja_eng.sub(' ', sent)

    tokens = []
    for morph, pos, _start, _end in _kiwi.tokenize(sent):
        # morph가 객체일 때 .form, 아니면 str 자체
        token = morph.form if hasattr(morph, 'form') else morph
        # 허용 품사 및 불용어 필터링
        if pos in _allowed_pos and len(token) > 1 and token not in _stop:
            # USE_MECAB 플래그에 따라 표제어 추출
            if USE_MECAB:
                try:
                    lemma = _mecab.pos(token)[0][0]
                except Exception:
                    lemma = token
            else:
                lemma = token
            tokens.append(lemma)

    return tokens


def add_tokens(df, col='Script'):
    df['Tokens'] = df[col].map(tokenize)
    df['CleanText'] = df['Tokens'].map(' '.join)
    return df
