import numpy as np
from data_utils import group_by_qid

def dcg_k(y, k):
    y = np.asarray(y)[:k]
    return np.sum((2**y - 1) / np.log2(np.arange(2, len(y)+2)))

def ndcg_k(y_true, y_pred, k):
    idx = np.argsort(y_pred)[::-1]
    y_sorted = y_true[idx]
    dcg = dcg_k(y_sorted, k)
    idcg = dcg_k(sorted(y_true, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0

def mrr(y_true, y_pred):
    idx = np.argsort(y_pred)[::-1]
    y_sorted = y_true[idx]
    for i, y in enumerate(y_sorted):
        if y > 0:
            return 1/(i+1)
    return 0

def average_precision(y_true, y_pred):
    idx = np.argsort(y_pred)[::-1]
    y_sorted = y_true[idx]
    hits, score = 0, 0
    for i, y in enumerate(y_sorted):
        if y > 0:
            hits += 1
            score += hits/(i+1)
    return score/hits if hits > 0 else 0

def evaluate_all(y, pred, qid):
    groups = group_by_qid(qid)

    res = {"NDCG@1":[], "NDCG@5":[], "NDCG@10":[], "MAP":[], "MRR":[]}

    for g in groups.values():
        y_g = y[g]
        p_g = pred[g]

        res["NDCG@1"].append(ndcg_k(y_g, p_g, 1))
        res["NDCG@5"].append(ndcg_k(y_g, p_g, 5))
        res["NDCG@10"].append(ndcg_k(y_g, p_g, 10))
        res["MAP"].append(average_precision(y_g, p_g))
        res["MRR"].append(mrr(y_g, p_g))

    return {k: np.mean(v) for k,v in res.items()}
