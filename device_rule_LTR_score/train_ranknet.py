#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[목적]
가장 최신 LTR JSONL(각 줄=query)을 읽어서
PyTorch로 RankNet(pairwise loss) 학습 + MAP/NDCG 평가를 수행한다.

[핵심 개선]
1) label=2가 거의 없어도 학습이 되도록 "graded pairwise"로 학습
   - label_i > label_j 인 쌍 (i, j)를 만들어 i가 j보다 높게 랭킹되도록 학습
   - 2>1, 2>0, 1>0 모두 학습 가능

2) MAP은 binary relevance가 필요하므로 2가지 버전을 같이 출력
   - MAP(rel>=2): 정답 중심
   - MAP(rel>=1): 약한 양성 포함 (네 데이터 상황에서 더 유의미할 가능성 큼)

[입력]
device_rule_LTR_score/dataset/ 하위의 최신 *.jsonl
각 query 레코드에
- relevance_labels (0/1/2)
- features (dict list)
가 있어야 함

[출력]
- 모델 저장: device_rule_LTR_score/output/ranknet/<timestamp>_<device>/ranknet.pt
- feature 키 저장: .../feature_keys.json
- 메트릭 저장: .../metrics.json
"""

import os
import glob
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# CONFIG (여기만 조절)
# =========================================================
DATASET_DIR = "device_rule_LTR_score/dataset"
OUT_DIR = "device_rule_LTR_score/output/ranknet"

SEED = 7
VALID_RATIO = 0.2          # query 단위 split 80% train / 20% valid
EPOCHS = 10
LR = 1e-3 #Adam 업데이트 보폭(크면 빨리 가지만 불안정할 수 있음)
WEIGHT_DECAY = 1e-5 #L2 규제(가중치 커지는 걸 벌점 → 과적합 완화)

# RankNet pairwise 샘플링
PAIRS_PER_QUERY = 300      # query당 pair 샘플 수(너무 크면 느림)

# 평가
NDCG_KS = [1, 3, 5, 10]
MAP_REL_LEVELS = [2, 1]    # MAP을 rel>=2, rel>=1 두 가지로 출력

# device별로 따로 학습할지
TRAIN_PER_DEVICE = False   # True면 device별 모델 저장/평가


# =========================================================
# 유틸
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_latest_jsonl(dataset_dir: str) -> str:
    files = glob.glob(os.path.join(dataset_dir, "**", "*.jsonl"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"jsonl 파일 없음: {dataset_dir}")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_queries(jsonl_path: str) -> List[Dict[str, Any]]:
    queries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def split_queries(queries: List[Dict[str, Any]], seed: int, valid_ratio: float) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    idx = list(range(len(queries)))
    rng.shuffle(idx)
    n_valid = max(1, int(len(idx) * valid_ratio))
    valid_set = set(idx[:n_valid])
    train, valid = [], []
    for i, q in enumerate(queries):
        (valid if i in valid_set else train).append(q)
    return train, valid


def collect_feature_keys(queries: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for q in queries:
        for feat in (q.get("features") or []):
            if isinstance(feat, dict):
                keys.update(feat.keys())
    return sorted(keys)


def feats_to_tensor(features: List[Dict], feat_keys: List[str]) -> torch.Tensor:
    # 문서별 feature dict -> (num_docs, num_features) 텐서
    X = []
    for feat in features:
        feat = feat if isinstance(feat, dict) else {}
        X.append([float(feat.get(k, 0.0)) for k in feat_keys])
    if not X:
        return torch.zeros((0, len(feat_keys)), dtype=torch.float32)
    return torch.tensor(X, dtype=torch.float32)


# =========================================================
# 평가 지표 (MAP / NDCG)
# =========================================================
def average_precision(labels_bin: List[int], scores: List[float]) -> float:
    # AP: relevant(1) 기준 순위 기반
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hit = 0
    s = 0.0
    for rank, i in enumerate(order, start=1):
        if labels_bin[i] == 1:
            hit += 1
            s += hit / rank
    return 0.0 if hit == 0 else s / hit


def ndcg_at_k(labels: List[int], scores: List[float], k: int) -> float:
    # graded relevance NDCG@k (labels 0/1/2 그대로 사용)
    import math
    if not labels:
        return 0.0
    k = min(k, len(labels))

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    def dcg(order_idx):
        s = 0.0
        for rank, i in enumerate(order_idx, start=1):
            rel = labels[i]
            s += (2 ** rel - 1) / math.log2(rank + 1)
        return s

    dcg_val = dcg(order)
    ideal = sorted(range(len(labels)), key=lambda i: labels[i], reverse=True)[:k]
    idcg = dcg(ideal)
    return 0.0 if idcg == 0 else dcg_val / idcg


def eval_metrics(model, queries: List[Dict[str, Any]], feat_keys: List[str], device: str) -> Dict[str, float]:
    """
    - NDCG: graded label(0/1/2) 그대로 사용
    - MAP: rel>=2, rel>=1 두 버전 모두 출력
    """
    model.eval()
    qn = 0
    ndcg_sum = {k: 0.0 for k in NDCG_KS}
    ap_sum = {lvl: 0.0 for lvl in MAP_REL_LEVELS}

    with torch.no_grad():
        for q in queries:
            labels = q.get("relevance_labels") or []
            feats = q.get("features") or []
            n = min(len(labels), len(feats))
            if n == 0:
                continue

            labels = [int(x) for x in labels[:n]]
            X = feats_to_tensor(feats[:n], feat_keys).to(device)

            scores = model(X).detach().cpu().view(-1).tolist()

            # MAP 여러 버전
            for lvl in MAP_REL_LEVELS:
                labels_bin = [1 if y >= lvl else 0 for y in labels]
                ap_sum[lvl] += average_precision(labels_bin, scores)

            # NDCG
            for k in NDCG_KS:
                ndcg_sum[k] += ndcg_at_k(labels, scores, k)

            qn += 1

    out = {}
    for lvl in MAP_REL_LEVELS:
        out[f"MAP(rel>={lvl})"] = ap_sum[lvl] / qn if qn else 0.0
    for k in NDCG_KS:
        out[f"NDCG@{k}"] = ndcg_sum[k] / qn if qn else 0.0
    out["num_eval_queries"] = qn
    return out


# =========================================================
# RankNet 모델
# =========================================================
class RankNet(nn.Module):
    """
    RankNet 점수함수 s(x) 학습용 모델
    - 가장 단순하게: MLP 2층
    """
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (N, 1)


def sample_pairs_graded(labels: List[int], rng: random.Random, pairs_per_query: int) -> List[Tuple[int, int]]:
    """
    graded pair 샘플링:
      label_i > label_j 인 쌍 (i,j)를 샘플링해서
      "i가 j보다 높아야 한다"를 학습하게 함.

    - label=2가 없어도 label=1 > label=0 으로 학습 가능
    """
    idx_by_y: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        idx_by_y.setdefault(int(y), []).append(i)

    levels = sorted(idx_by_y.keys())
    if len(levels) < 2:
        return []

    pairs = []
    for _ in range(pairs_per_query):
        y_hi = rng.choice(levels)
        lower = [y for y in levels if y < y_hi]
        if not lower:
            continue
        y_lo = rng.choice(lower)

        i = rng.choice(idx_by_y[y_hi])
        j = rng.choice(idx_by_y[y_lo])
        if i != j:
            pairs.append((i, j))
    return pairs


def train_ranknet(
    train_queries: List[Dict[str, Any]],
    valid_queries: List[Dict[str, Any]],
    feat_keys: List[str],
    run_out_dir: str,
    device: str
) -> Dict[str, Any]:
    rng = random.Random(SEED)

    model = RankNet(in_dim=len(feat_keys), hidden=32).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # RankNet loss: -log(sigmoid(s_i - s_j)) = BCEWithLogitsLoss(logits, 1)
    bce = nn.BCEWithLogitsLoss()

    best_metric = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        step = 0

        for q in train_queries:
            labels = q.get("relevance_labels") or []
            feats = q.get("features") or []
            n = min(len(labels), len(feats))
            if n == 0:
                continue

            labels = [int(x) for x in labels[:n]]
            pairs = sample_pairs_graded(labels, rng, PAIRS_PER_QUERY)
            if not pairs:
                continue

            X = feats_to_tensor(feats[:n], feat_keys).to(device)
            scores = model(X).view(-1)  # (n,)

            # pairwise logits = s_i - s_j
            pi = torch.tensor([i for i, j in pairs], dtype=torch.long, device=device)
            pj = torch.tensor([j for i, j in pairs], dtype=torch.long, device=device)
            logits = scores[pi] - scores[pj]
            target = torch.ones_like(logits)

            loss = bce(logits, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            step += 1

        avg_loss = total_loss / step if step else 0.0

        # epoch 종료 후 평가
        metrics = eval_metrics(model, valid_queries, feat_keys, device)

        # best 선택 기준: NDCG@10이 있으면 그걸로, 없으면 NDCG@5, 그것도 없으면 MAP(rel>=1)
        if "NDCG@10" in metrics:
            select = metrics["NDCG@10"]
        elif "NDCG@5" in metrics:
            select = metrics["NDCG@5"]
        else:
            select = metrics.get("MAP(rel>=1)", 0.0)

        print(f"[epoch {epoch:02d}] step={step} loss={avg_loss:.4f} | valid={json.dumps(metrics, ensure_ascii=False)}")

        if select > best_metric:
            best_metric = select
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # best 모델 저장
    os.makedirs(run_out_dir, exist_ok=True)
    model_path = os.path.join(run_out_dir, "ranknet.pt")
    feat_path = os.path.join(run_out_dir, "feature_keys.json")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), model_path)
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feat_keys, f, ensure_ascii=False, indent=2)

    final_metrics = eval_metrics(model, valid_queries, feat_keys, device)

    out = {
        "model_path": model_path,
        "feature_keys_path": feat_path,
        "best_select_metric": float(best_metric),
        "valid_metrics": final_metrics,
        "train_queries": len(train_queries),
        "valid_queries": len(valid_queries),
        "num_features": len(feat_keys),
        "pairs_per_query": PAIRS_PER_QUERY,
        "epochs": EPOCHS,
        "lr": LR,
    }

    with open(os.path.join(run_out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("saved:", model_path)
    print("final metrics:", json.dumps(final_metrics, ensure_ascii=False, indent=2))
    return out


# =========================================================
# main
# =========================================================
def main():
    set_seed(SEED)
    latest = find_latest_jsonl(DATASET_DIR)
    print("✅ 최신 데이터셋:", latest)

    queries = load_queries(latest)
    if not queries:
        raise RuntimeError("JSONL이 비어있습니다.")

    # device별 학습 옵션
    if TRAIN_PER_DEVICE:
        devices = sorted({q.get("device", "GENERIC") for q in queries})
        targets = devices
    else:
        targets = ["ALL"]

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("✅ torch device:", torch_device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dev in targets:
        if dev == "ALL":
            q_dev = queries
        else:
            q_dev = [q for q in queries if q.get("device", "GENERIC") == dev]

        feat_keys = collect_feature_keys(q_dev)
        if not feat_keys:
            print(f"⚠️ device={dev} feature가 없음. 스킵")
            continue

        train_q, valid_q = split_queries(q_dev, SEED, VALID_RATIO)
        if len(train_q) < 3 or len(valid_q) < 1:
            print(f"⚠️ device={dev} query 수 부족. 스킵 (train={len(train_q)}, valid={len(valid_q)})")
            continue

        run_dir = os.path.join(OUT_DIR, f"{ts}_{dev}")
        print(f"\n=== TRAIN RankNet | device={dev} | trainQ={len(train_q)} validQ={len(valid_q)} feats={len(feat_keys)} ===")
        train_ranknet(train_q, valid_q, feat_keys, run_dir, torch_device)


if __name__ == "__main__":
    main()
