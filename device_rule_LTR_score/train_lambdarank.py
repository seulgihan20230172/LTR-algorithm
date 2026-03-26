# -*- coding: utf-8 -*-
from ltr_utils import train_and_eval

DATASET_DIR = "device_rule_LTR_score/dataset"
MODEL_OUT = "device_rule_LTR_score/models/lambdarank.txt"

PARAMS = {
    "objective": "lambdarank",
    "learning_rate": 0.05,
    "num_leaves": 15,         # 조금 더 단순하게
    "max_depth": 6,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "lambda_l1": 0.0,
    "verbosity": -1,
    "num_boost_round": 800,
}

if __name__ == "__main__":
    train_and_eval(DATASET_DIR, PARAMS, MODEL_OUT, map_rel_level=2)
