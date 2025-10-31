# src/re_ranker.py
from sentence_transformers import CrossEncoder
# model nhẹ but effective
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def __init__(self, model_name=MODEL_NAME):
    self.model = CrossEncoder(model_name)

def rerank(self, query, docs):
    # docs: list of short strings (candidates)
    pairs = [[query, d] for d in docs]
    scores = self.model.predict(pairs)  # higher = better
    # return indices sorted desc
    idxs = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return idxs, scores