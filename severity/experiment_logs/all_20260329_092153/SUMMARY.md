# Severity 실험 요약
- 로그 디렉터리: `/home/work/lab_project/severity/experiment_logs/all_20260329_092153/logs`
- 로그 파일 수: 16

| 실험(로그 파일) | 상태 | Test Acc | Test Macro F1 | Val MRR | Test MRR | 전체(s) |
|---|---|---:|---:|---:|---:|---:|
| `anomaly_deep_stacked_ae_train_thresholds.log` | ok | 0.254000 | 0.240000 | 0.856287 | 0.869012 | 57.701 |
| `anomaly_denoising_ae_train_thresholds.log` | ok | 0.251700 | 0.245400 | 0.888224 | 0.842543 | 21.053 |
| `anomaly_isolation_forest_train_thresholds.log` | ok | 0.247400 | 0.233900 | 0.859381 | 0.888224 | 30.036 |
| `anomaly_sequence_ae_train_thresholds.log` | ok | 0.248700 | 0.239300 | 0.867515 | 0.867715 | 80.800 |
| `anomaly_vae_train_thresholds.log` | ok | 0.255500 | 0.254900 | 0.874251 | 0.866267 | 16.357 |
| `anomaly_vanilla_ae_train_thresholds.log` | ok | 0.246700 | 0.240100 | 0.850898 | 0.863772 | 21.317 |
| `cls_lightgbm_train_thresholds.log` | ok |  |  |  |  |  |
| `cls_random_forest_train_thresholds.log` | ok |  |  |  |  |  |
| `l2r_bm25_train_thresholds.log` | ok | 0.249500 | 0.249400 | 0.863024 | 0.856337 | 3.159 |
| `l2r_lambdamart_train_thresholds.log` | ok | 0.247100 | 0.099800 | 0.868912 | 0.855817 | 641.256 |
| `l2r_lambdarank_train_thresholds.log` | ok | 0.250400 | 0.246900 | 0.850062 | 0.864521 | 750.016 |
| `l2r_listmle_train_thresholds.log` | ok | 0.245300 | 0.244200 | 0.884581 | 0.883483 | 85.362 |
| `l2r_listnet_train_thresholds.log` | ok | 0.247500 | 0.247500 | 0.881587 | 0.877196 | 19.975 |
| `l2r_ranknet_train_thresholds.log` | ok | 0.247400 | 0.198100 | 0.848232 | 0.873353 | 3208.674 |
| `l2r_xgboost_train_thresholds.log` | ok | 0.253200 | 0.166200 | 0.854042 | 0.860030 | 26.392 |
| `reg_xgboost_regressor_train_thresholds.log` | ok | 0.251200 | 0.222800 | 0.876597 | 0.870359 | 113.369 |
