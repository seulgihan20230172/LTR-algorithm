# -*- coding: utf-8 -*-
import os, glob, json, random
from typing import List, Dict, Tuple

import lightgbm as lgb


def find_latest_jsonl(dataset_dir: str) -> str:
    # dataset_dir 아래 최신 jsonl 선택 (재귀 포함)
    files = glob.glob(os.path.join(dataset_dir, "**", "*.jsonl"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"jsonl 없음: {dataset_dir}")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_queries(jsonl_path: str) -> List[Dict]:
    # JSONL(각 줄=query)을 리스트로 로드
    queries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def split_queries(queries: List[Dict], seed: int = 7, valid_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    # query 단위 split
    rng = random.Random(seed)
    idx = list(range(len(queries)))
    rng.shuffle(idx)
    n_valid = max(1, int(len(idx) * valid_ratio))
    valid_idx = set(idx[:n_valid])
    train, valid = [], []
    for i, q in enumerate(queries):
        (valid if i in valid_idx else train).append(q)
    return train, valid


def to_lgb_dataset(queries: List[Dict]) -> Tuple[lgb.Dataset, List[str]]:
    # query JSON -> LightGBM Dataset (문서행으로 펼침)
    # feature 키는 데이터 전체 union으로 맞춤
    feat_keys = set()
    for q in queries:
        for feat in q.get("features", []) or []:
            if isinstance(feat, dict):
                feat_keys.update(feat.keys())
    feat_keys = sorted(feat_keys)

    X, y, group = [], [], []
    for q in queries:
        labels = q.get("relevance_labels", []) or []
        feats = q.get("features", []) or []
        n = min(len(labels), len(feats))
        group.append(n)
        for i in range(n):
            fi = feats[i] if isinstance(feats[i], dict) else {}
            X.append([float(fi.get(k, 0.0)) for k in feat_keys])
            y.append(int(labels[i]))

    ds = lgb.Dataset(X, label=y, group=group, feature_name=feat_keys, free_raw_data=True)
    return ds, feat_keys


def predict_scores(model: lgb.Booster, queries: List[Dict], feat_keys: List[str]) -> List[List[float]]:
    # query별 문서 점수 리스트 반환
    all_scores = []
    for q in queries:
        feats = q.get("features", []) or []
        Xq = []
        for feat in feats:
            feat = feat if isinstance(feat, dict) else {}
            Xq.append([float(feat.get(k, 0.0)) for k in feat_keys])
        scores = model.predict(Xq, num_iteration=model.best_iteration)
        all_scores.append(list(map(float, scores)))
    return all_scores


def ndcg_at_k(labels: List[int], scores: List[float], k: int) -> float:
    # graded relevance NDCG@k (labels: 0/1/2 그대로 사용)
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    def dcg(order):
        s = 0.0
        for rank, i in enumerate(order, start=1):
            rel = labels[i]
            s += (2**rel - 1) / (math.log2(rank + 1))
        return s

    import math
    dcg_val = dcg(idx)
    ideal_idx = sorted(range(len(labels)), key=lambda i: labels[i], reverse=True)[:k]
    idcg = dcg(ideal_idx)
    return 0.0 if idcg == 0 else dcg_val / idcg


def average_precision(labels_bin: List[int], scores: List[float]) -> float:
    # AP (binary relevance 기준) -> MAP은 query별 AP 평균
    # labels_bin: 1이면 relevant, 0이면 non-relevant
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hit = 0
    s = 0.0
    for rank, i in enumerate(order, start=1):
        if labels_bin[i] == 1:
            hit += 1
            s += hit / rank
    return 0.0 if hit == 0 else s / hit


def evaluate_map_ndcg(queries: List[Dict], scores_by_query: List[List[float]], ndcg_ks=(1,3,5,10), map_rel_level: int = 2) -> Dict:
    """
    - NDCG: labels(0/1/2) 그대로 graded로 계산
    - MAP: binary로 계산해야 하므로,
      map_rel_level=2면 label>=2만 relevant (정답 중심)
      map_rel_level=1이면 label>=1도 relevant (약한 양성 포함)
    """
    ndcg_sum = {k: 0.0 for k in ndcg_ks}
    ap_sum = 0.0
    qn = len(queries)

    for q, scores in zip(queries, scores_by_query):
        labels = (q.get("relevance_labels") or [])
        n = min(len(labels), len(scores))
        labels = labels[:n]
        scores = scores[:n]

        # MAP용 binary relevance 만들기
        labels_bin = [1 if y >= map_rel_level else 0 for y in labels]
        ap_sum += average_precision(labels_bin, scores)

        for k in ndcg_ks:
            ndcg_sum[k] += ndcg_at_k(labels, scores, min(k, len(labels)))

    out = {"MAP": ap_sum / qn if qn else 0.0}
    for k in ndcg_ks:
        out[f"NDCG@{k}"] = ndcg_sum[k] / qn if qn else 0.0
    return out


def train_and_eval(
    dataset_dir: str,
    params: Dict,
    model_out: str,
    seed: int = 7,
    valid_ratio: float = 0.2,
    ndcg_ks=(1,3,5,10),
    map_rel_level: int = 2,
) -> None:
    # 최신 데이터 로드
    latest = find_latest_jsonl(dataset_dir)
    queries = load_queries(latest)
    train_q, valid_q = split_queries(queries, seed=seed, valid_ratio=valid_ratio)

    dtrain, feat_keys = to_lgb_dataset(train_q)
    dvalid, _ = to_lgb_dataset(valid_q)

    # LightGBM 학습
    params = dict(params)
    params["seed"] = seed
    params["metric"] = "ndcg"
    params["eval_at"] = list(ndcg_ks)

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=int(params.get("num_boost_round", 800)),
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # 평가
    valid_scores = predict_scores(model, valid_q, feat_keys)
    metrics = evaluate_map_ndcg(valid_q, valid_scores, ndcg_ks=ndcg_ks, map_rel_level=map_rel_level)

    # 저장
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save_model(model_out)

    print("latest:", latest)
    print("saved:", model_out)
    print("metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
