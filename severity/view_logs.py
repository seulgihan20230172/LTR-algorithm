import os
import re
from collections import defaultdict

def analyze_importance_logs():
    # 분석할 정확한 파일 목록
    target_logs = [
        "feature_importance_train_severity_reg_decision_tree_regressor_train_score_relevance_0_3.log",
        "feature_importance_train_severity_reg_knn_regressor_train_score_relevance_0_3.log",
        "feature_importance_train_severity_reg_linear_regression_train_score_relevance_0_3.log",
        "feature_importance_train_severity_l2r_xgboost_train_score_relevance_0_3.log",
        "feature_importance_train_severity_l2r_ranknet_train_score_relevance_0_3.log",
        "feature_importance_train_severity_l2r_listnet_train_score_relevance_0_3.log",
        "feature_importance_train_severity_l2r_listmle_train_score_relevance_0_3.log",
        "feature_importance_train_severity_l2r_lambdamart_train_score_relevance_0_3.log",
        "feature_importance_train_severity_anomaly_sequence_ae_train_score_relevance_0_3.log",
        "feature_importance_train_severity_anomaly_denoising_ae_train_score_relevance_0_3.log",
        "feature_importance_train_severity_anomaly_vanilla_ae_train_score_relevance_0_3.log"
    ]

    # 그룹별 데이터를 담을 딕셔너리 {그룹명: {피처명: [중요도값들]}}
    group_data = {
        "REG": defaultdict(list),
        "L2R": defaultdict(list),
        "ANOMALY": defaultdict(list)
    }

    print("=" * 80)
    print(f"{'개별 모델별 Importance (0.1 이상)':^70}")
    print("=" * 80)

    for file_name in target_logs:
        if not os.path.exists(file_name):
            continue

        # 그룹 판별
        group_key = None
        if "_reg_" in file_name: group_key = "REG"
        elif "_l2r_" in file_name: group_key = "L2R"
        elif "_anomaly_" in file_name: group_key = "ANOMALY"

        print(f"\n[FILE] {file_name}")
        
        with open(file_name, 'r', encoding='utf-8') as f:
            start_reading = False
            for line in f:
                # 데이터 시작 지점 포착
                if "importance_norm" in line and "feature" in line:
                    start_reading = True
                    continue
                
                if start_reading:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            # rank(0), norm(1), raw(2), feature(3)
                            norm_val = float(parts[1])
                            feature_name = parts[3]
                            
                            # 그룹 통계를 위해 모든 값 저장
                            if group_key:
                                group_data[group_key][feature_name].append(norm_val)

                            # 0.1 이상인 것만 화면 출력
                            if norm_val >= 0.1:
                                print(f" - {feature_name:<20} : {norm_val:.4f}")
                        except ValueError:
                            continue

    # 그룹별 평균 계산 및 출력
    print("\n" + "=" * 80)
    print(f"{'그룹별 평균 Importance (0.1 이상)':^70}")
    print("=" * 80)

    for group_name, features in group_data.items():
        print(f"\n[{group_name} Group Average]")
        has_top_feature = False
        
        # 피처별 평균 내기
        avg_features = []
        for f_name, values in features.items():
            avg_val = sum(values) / len(values)
            avg_features.append((f_name, avg_val))
        
        # 평균값 기준 정렬
        avg_features.sort(key=lambda x: x[1], reverse=True)

        for f_name, avg_val in avg_features:
            if avg_val >= 0.1:
                print(f" - {f_name:<20} : {avg_val:.4f}")
                has_top_feature = True
        
        if not has_top_feature:
            print(" - (평균 0.1 이상의 피처가 없습니다.)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_importance_logs()
