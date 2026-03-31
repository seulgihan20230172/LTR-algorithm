'''
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
'''
'''
import pandas as pd

df = pd.read_csv("./logging_monitoring_anomalies.csv")

TARGET_COL = "Severity"  # 너 데이터 컬럼명 맞게 수정

# 문자형 컬럼만 선택
cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    if col == TARGET_COL:
        continue

    print(f"\n===== Column: {col} =====")
    
    # 기존 count / ratio
    counts = df[col].value_counts(dropna=False)
    ratios = df[col].value_counts(normalize=True, dropna=False)
    
    base_result = pd.DataFrame({
        "count": counts,
        "ratio": ratios
    })
    
    print("\n[기본 분포]")
    print(base_result)

    # 🔥 추가: severity 분포 (핵심)
    print("\n[Severity 비율 (각 값별)]")
    
    sev_dist = pd.crosstab(
        df[col],
        df[TARGET_COL],
        normalize="index"   # 각 value 기준 비율
    )
    
    print(sev_dist)
    '''
import pandas as pd

df = pd.read_csv("./logging_monitoring_anomalies.csv")

TARGET_COL = "Severity"

for col in df.columns:
    if col == TARGET_COL:
        continue

    print(f"\n===== Column: {col} =====")

    series = df[col]
    print(col)
    # 🔥 숫자형인데 값이 너무 많으면 binning
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        print("(numeric → binning 적용)")
        binned = pd.qcut(series, q=10, duplicates="drop")  # 10개 구간
        use_col = binned
    else:
        use_col = series

    # 기본 분포
    counts = use_col.value_counts(dropna=False)
    ratios = use_col.value_counts(normalize=True, dropna=False)

    base_result = pd.DataFrame({
        "count": counts,
        "ratio": ratios
    })

    print("\n[기본 분포]")
    print(base_result)

    # 🔥 severity 분포
    print("\n[Severity 비율]")
    sev_ratio = pd.crosstab(
        use_col,
        df[TARGET_COL],
        normalize="index"
    )
    print(sev_ratio)
