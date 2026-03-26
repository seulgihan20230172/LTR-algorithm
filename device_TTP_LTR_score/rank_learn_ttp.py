import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame([
    ["T1046",     1.00, 0.40, 0.10],
    ["T1547.006", 0.90, 0.95, 0.60],
    ["T1105",     0.80, 0.75, 0.35],
    ["T1059",     0.60, 0.70, 0.25],
    ["T1055",     0.20, 0.85, 0.70],
], columns=["ttp","score","ot","cost"])

X, y = [], []

for i, j in combinations(df.index, 2):
    a, b = df.loc[i], df.loc[j]

    # OT 환경: ot_impact가 큰 게 먼저라는 약한 규칙
    if a.ot > b.ot:
        X.append(a[["score","ot","cost"]] - b[["score","ot","cost"]])
        y.append(1)
        X.append(b[["score","ot","cost"]] - a[["score","ot","cost"]])
        y.append(0)
    else:
        X.append(b[["score","ot","cost"]] - a[["score","ot","cost"]])
        y.append(1)
        X.append(a[["score","ot","cost"]] - b[["score","ot","cost"]])
        y.append(0)

model = LogisticRegression().fit(X, y)

df["rank_score"] = df[["score","ot","cost"]] @ model.coef_[0]
print(df.sort_values("rank_score", ascending=False)[["ttp","rank_score"]])
