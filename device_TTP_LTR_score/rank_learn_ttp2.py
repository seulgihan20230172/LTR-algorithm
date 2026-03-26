import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1) TTP 후보 + feature 정의
#   - base_score: 네가 이미 만든 점수(그대로 사용)
#   - net_signal: 네트워크 장비(FW/IDS)에서 잘 잡히는 정도(0~1)
#   - host_signal: 호스트 장비(EDR)에서 잘 잡히는 정도(0~1)
#   - c2_signal: C2/전송/프로토콜 성격(IDS에서 특히 중요)
# -----------------------------
df = pd.DataFrame([
    # ttp,        base_score, net_signal, host_signal, c2_signal
    ["T1046",     1.00,       0.95,       0.10,        0.20],  # 스캔(네트워크 강)
    ["T1547.006", 0.90,       0.05,       0.95,        0.10],  # 커널모듈(호스트 강)
    ["T1105",     0.80,       0.60,       0.70,        0.85],  # 전송/C2 성격
    ["T1059",     0.60,       0.10,       0.85,        0.20],  # 명령 실행(호스트 강)
    ["T1071",     0.30,       0.70,       0.10,        0.90],  # 앱계층 프로토콜(C2 강)
    ["T1055",     0.20,       0.05,       0.90,        0.10],  # 프로세스 인젝션(호스트 강)
], columns=["ttp","base_score","net_signal","host_signal","c2_signal"])

FEATURES = ["base_score","net_signal","host_signal","c2_signal"]

# -----------------------------
# 2) 장비별 "선호(utility)" 정의
#   -> 여기서 장비마다 기준이 달라지므로, 학습된 w도 장비마다 달라짐
# -----------------------------
DEVICE_UTILITY_WEIGHTS = {
    "FW":  {"base_score":0.6, "net_signal":1.2, "host_signal":0.1, "c2_signal":0.3},  # FW: 네트워크 우선
    "EDR": {"base_score":0.4, "net_signal":0.1, "host_signal":1.3, "c2_signal":0.2},  # EDR: 호스트 우선
    "IDS": {"base_score":0.5, "net_signal":0.7, "host_signal":0.1, "c2_signal":1.2},  # IDS: C2/프로토콜 우선
}

def utility(row, device: str) -> float:
    w = DEVICE_UTILITY_WEIGHTS[device]
    return sum(w[f] * float(row[f]) for f in FEATURES)

# -----------------------------
# 3) Pairwise 데이터 생성 + 학습 + 랭킹 출력
# -----------------------------
def train_and_rank(df: pd.DataFrame, device: str) -> pd.DataFrame:
    X, y = [], []
    for i, j in combinations(df.index, 2):
        a, b = df.loc[i], df.loc[j]
        ua, ub = utility(a, device), utility(b, device)

        xa = a[FEATURES].to_numpy(dtype=float)
        xb = b[FEATURES].to_numpy(dtype=float)

        # winner-loser 방향을 (1), 반대 방향을 (0)로 넣어 2클래스 확보
        if ua > ub:
            X.append(xa - xb); y.append(1)
            X.append(xb - xa); y.append(0)
        else:
            X.append(xb - xa); y.append(1)
            X.append(xa - xb); y.append(0)

    model = LogisticRegression(max_iter=500).fit(X, y)

    # 개별 점수 = w^T x (상대 순위 목적이라 intercept는 생략해도 됨)
    w = model.coef_[0]
    out = df.copy()
    out["rank_score"] = out[FEATURES].to_numpy(dtype=float) @ w
    out = out.sort_values("rank_score", ascending=False).reset_index(drop=True)

    # (선택) 어떤 feature를 학습했는지 보여주기
    print(f"\n[{device}] learned weights (for {FEATURES}) = {np.round(w, 3)}")
    return out[["ttp","rank_score"]]

if __name__ == "__main__":
    # Baseline: base_score로만 정렬(항상 동일)
    print("\n=== Baseline: sort by base_score (context-invariant) ===")
    print(df.sort_values("base_score", ascending=False)[["ttp","base_score"]].reset_index(drop=True))

    # Device-aware learned rankings
    for dev in ["FW","EDR","IDS"]:
        ranked = train_and_rank(df, dev)
        print(f"\n=== Learned ranking for device={dev} (should differ) ===")
        print(ranked)
