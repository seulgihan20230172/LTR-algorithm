"""
Goal
- Show that "simple sorting by base_score" gives ONE fixed order
- But a learned ranking model can output DIFFERENT orders depending on (scenario, device)

How it works (minimal + clear):
1) We define your TTP candidates with a base_score (your current output)
2) We add context/features (category, device-applicability, cost, FP-risk, OT-impact)
3) We create 3 contexts:
   - IT-Scenario + Firewall
   - OT-Scenario + EDR
   - OT-Scenario + IDS
4) For each context, we generate pairwise preferences (A>B) from a hidden utility rule
   (this simulates "ground-truth preference" a SOC would follow)
5) Train a simple pairwise ranker (logistic regression on feature differences)
6) Compare:
   - Sorting by base_score
   - Context-aware ranking model result (changes by scenario/device)
"""

from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# 1) Your TTP candidates + extra features
# -----------------------------
ttps = [
    # id, name, base_score, category, ot_impact, fw, edr, ids, cost, fp_risk
    ("T1046",     "Network Service Scanning",        1.00, "discovery",    0.40, 1, 0, 1, 0.10, 0.20),
    ("T1547.006", "Kernel Modules and Extensions",    0.90, "persistence", 0.95, 0, 1, 0, 0.60, 0.35),
    ("T1105",     "Ingress Tool Transfer",           0.80, "command_ctrl", 0.75, 0, 1, 1, 0.35, 0.30),
    ("T1059",     "Command and Scripting Interpreter",0.60, "execution",   0.70, 0, 1, 0, 0.25, 0.40),
    ("T1071",     "Application Layer Protocol",      0.30, "command_ctrl", 0.55, 1, 0, 1, 0.20, 0.45),
    ("T1055",     "Process Injection",               0.20, "defense_evasion",0.85,0, 1, 0, 0.70, 0.60),
]

df = pd.DataFrame(
    ttps,
    columns=["ttp_id","name","base_score","category","ot_impact","app_fw","app_edr","app_ids","cost","fp_risk"]
)

# simple one-hot for category
df = pd.get_dummies(df, columns=["category"], prefix="cat")

FEATURE_COLS = [
    "base_score", "ot_impact", "cost", "fp_risk",
    "app_fw", "app_edr", "app_ids",
] + [c for c in df.columns if c.startswith("cat_")]

# -----------------------------
# 2) Define contexts (scenario, device) with different "preference weights"
#    This is what makes ranking change across contexts.
# -----------------------------
CONTEXTS = {
    # IT+Firewall: scanning/visibility important, low cost preferred
    ("IT", "FW"): dict(w_base=0.8, w_ot=0.2, w_cost=-0.4, w_fp=-0.2, w_app=0.7,
                      cat_boost={"cat_discovery": 0.25, "cat_command_ctrl": 0.10, "cat_execution": 0.05}),
    # OT+EDR: persistence/stealth/impact super important, tolerate more cost
    ("OT", "EDR"): dict(w_base=0.4, w_ot=0.9, w_cost=-0.1, w_fp=-0.25, w_app=0.9,
                       cat_boost={"cat_persistence": 0.35, "cat_defense_evasion": 0.20, "cat_execution": 0.10}),
    # OT+IDS: network C2 patterns + moderate OT impact, cost matters some
    ("OT", "IDS"): dict(w_base=0.5, w_ot=0.6, w_cost=-0.25, w_fp=-0.2, w_app=0.8,
                       cat_boost={"cat_command_ctrl": 0.30, "cat_discovery": 0.10}),
}

def hidden_utility(row: pd.Series, scenario: str, device: str) -> float:
    """Simulated 'ground-truth' utility that a SOC might follow in each context."""
    cfg = CONTEXTS[(scenario, device)]
    app_col = {"FW":"app_fw", "EDR":"app_edr", "IDS":"app_ids"}[device]
    util = 0.0
    util += cfg["w_base"] * row["base_score"]
    util += cfg["w_ot"]   * row["ot_impact"]
    util += cfg["w_cost"] * row["cost"]
    util += cfg["w_fp"]   * row["fp_risk"]
    util += cfg["w_app"]  * row[app_col]

    # category boosts (context-specific)
    for k, v in cfg["cat_boost"].items():
        if k in row.index:
            util += v * row[k]
    return util

# -----------------------------
# 3) Build pairwise training set for each context
#    We create examples of (A,B) with label 1 if A>B else 0
#    Feature for pair: x = features(A) - features(B)
# -----------------------------
def make_pairwise_data(df: pd.DataFrame, scenario: str, device: str):
    X = []
    y = []
    pairs = []

    for i, j in combinations(range(len(df)), 2):
        ri = df.iloc[i]
        rj = df.iloc[j]
        ui = hidden_utility(ri, scenario, device)
        uj = hidden_utility(rj, scenario, device)

        # if utilities equal, skip (rare)
        if abs(ui - uj) < 1e-9:
            continue

        # Create both directions for better balance
        xi = ri[FEATURE_COLS].to_numpy(dtype=float)
        xj = rj[FEATURE_COLS].to_numpy(dtype=float)

        # i > j ?
        X.append(xi - xj)
        y.append(1 if ui > uj else 0)
        pairs.append((ri["ttp_id"], rj["ttp_id"]))

        # j > i ?
        X.append(xj - xi)
        y.append(1 if uj > ui else 0)
        pairs.append((rj["ttp_id"], ri["ttp_id"]))

    return np.vstack(X), np.array(y), pairs

# -----------------------------
# 4) Train a simple pairwise ranker per context
#    (Logistic regression on pairwise differences)
# -----------------------------
def train_pairwise_ranker(df: pd.DataFrame, scenario: str, device: str):
    X, y, _ = make_pairwise_data(df, scenario, device)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))
    ])
    model.fit(X, y)
    return model

def predict_item_scores(model, df: pd.DataFrame) -> pd.Series:
    """
    Convert pairwise model to per-item scores by comparing each item vs a fixed reference.
    Simple trick: score = w·x (linear), approximated by decision_function against zero vector.
    Since our pipeline is linear LR, we can use clf.coef_ on standardized space indirectly.
    We'll just compute a consistent linear score using model's decision function
    on each item's feature vector vs "all-zero" baseline.
    """
    # Use the pipeline transform + LR decision function on X = item_features
    # We'll treat "baseline" as zeros by just feeding features through scaler then LR.
    X_item = df[FEATURE_COLS].to_numpy(dtype=float)
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    Xs = scaler.transform(X_item)
    # linear score (logit) = w·x + b
    scores = Xs @ clf.coef_.ravel() + clf.intercept_[0]
    return pd.Series(scores, index=df.index)

# -----------------------------
# 5) Compare: base_score sorting vs context ranking
# -----------------------------
def show_rankings(df: pd.DataFrame):
    # baseline sorting (same everywhere)
    baseline = df.sort_values("base_score", ascending=False)[["ttp_id","name","base_score"]].reset_index(drop=True)
    baseline.index = baseline.index + 1

    print("\n" + "="*80)
    print("Baseline: Sorting by base_score (context-invariant)")
    print("="*80)
    print(baseline.to_string())

    # context-specific learned ranking
    for (scenario, device) in CONTEXTS.keys():
        model = train_pairwise_ranker(df, scenario, device)
        scores = predict_item_scores(model, df)

        out = df.copy()
        out["rank_score_model"] = scores
        # IMPORTANT: enforce device applicability by pushing non-applicable down hard
        app_col = {"FW":"app_fw", "EDR":"app_edr", "IDS":"app_ids"}[device]
        out["rank_score_model"] = out["rank_score_model"] + (out[app_col] * 1000.0)  # strong preference if applicable

        ranked = out.sort_values("rank_score_model", ascending=False)[["ttp_id","name","base_score","rank_score_model",app_col,"ot_impact","cost","fp_risk"]].reset_index(drop=True)
        ranked.index = ranked.index + 1

        print("\n" + "-"*80)
        print(f"Context-aware Ranking Model: scenario={scenario}, device={device}")
        print("-"*80)
        print(ranked.to_string())

if __name__ == "__main__":
    show_rankings(df)
