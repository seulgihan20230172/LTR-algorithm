# -*- coding: utf-8 -*-
from ltr_utils import train_and_eval

DATASET_DIR = "device_rule_LTR_score/dataset"
MODEL_OUT = "device_rule_LTR_score/models/lambdamart.txt"

# LambdaMART는 보통 lambdarank 목적함수 + 트리부스팅 튜닝으로 구현(실무 관행)
PARAMS = {
    "objective": "lambdarank",
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 31,
    "min_data_in_leaf": 20,          # (= min_child_samples)
    "min_sum_hessian_in_leaf": 1e-3, # (= min_child_weight 느낌)
    "feature_fraction": 0.8,         # colsample_bytree
    "bagging_fraction": 0.8,         # subsample
    "bagging_freq": 1,
    "lambda_l2": 1.0,                # reg_lambda
    "lambda_l1": 0.0,                # reg_alpha
    "verbosity": -1,
    "num_boost_round": 1200,
}

if __name__ == "__main__":
    train_and_eval(DATASET_DIR, PARAMS, MODEL_OUT, map_rel_level=2)
