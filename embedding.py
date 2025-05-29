# -*- coding: utf-8 -*-

# embedding.py
import torch, gc
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import numpy as np
from typing import List
from config import MODEL_DIR

_MODEL_NAME = "beomi/kcbert-large"   # :contentReference[oaicite:0]{index=0}

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, cache_dir=MODEL_DIR)
_model     = AutoModel.from_pretrained(_MODEL_NAME, cache_dir=MODEL_DIR)
_model.eval().cuda()

@torch.no_grad()
def _mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).type(torch.float32)
    return (last_hidden * mask).sum(1) / mask.sum(1)

def encode(texts: List[str], batch_size: int = 32) -> np.ndarray:
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        encoded = _tokenizer(batch, padding=True, truncation=True,
                             max_length=128, return_tensors='pt').to(_model.device)
        out = _model(**encoded, output_hidden_states=False)
        emb = _mean_pool(out.last_hidden_state, encoded['attention_mask'])
        vecs.append(emb.cpu().numpy())
        del encoded, out, emb
        gc.collect()
    return np.vstack(vecs)
