import pandas as pd

df = pd.read_csv("./logging_monitoring_anomalies.csv")


# 문자형 컬럼만 선택
cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    print(f"\n===== Column: {col} =====")
    
    counts = df[col].value_counts(dropna=False)
    ratios = df[col].value_counts(normalize=True, dropna=False)
    
    result = pd.DataFrame({
        "count": counts,
        "ratio": ratios
    })
    
    print(result)