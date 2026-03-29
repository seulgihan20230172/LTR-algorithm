# Severity 실험 요약
- 로그 디렉터리: `/home/work/lab_project/severity/experiment_logs/all_20260329_074159/logs`
- 로그 파일 수: 16

| 실험(로그 파일) | 상태 | Test Acc | Test Macro F1 | Val MRR | Test MRR | 전체(s) |
|---|---|---:|---:|---:|---:|---:|
| `anomaly_deep_stacked_ae_train_thresholds.log` | ok | 0.250900 | 0.250500 | 0.878244 | 0.853293 | 59.487 |
| `anomaly_denoising_ae_train_thresholds.log` | ok | 0.255600 | 0.255100 | 0.860729 | 0.860329 | 20.899 |
| `anomaly_isolation_forest_train_thresholds.log` | ok | 0.247400 | 0.233900 | 0.859381 | 0.888224 | 6.337 |
| `anomaly_sequence_ae_train_thresholds.log` | ok | 0.248700 | 0.239300 | 0.867265 | 0.869711 | 83.932 |
| `anomaly_vae_train_thresholds.log` | ok | 0.248700 | 0.245400 | 0.868762 | 0.863024 | 15.464 |
| `anomaly_vanilla_ae_train_thresholds.log` | ok | 0.246700 | 0.245400 | 0.818313 | 0.860030 | 21.098 |
| `cls_lightgbm_train_thresholds.log` | ok | 0.251500 | 0.249100 | 0.876747 | 0.888972 | 38.967 |
| `cls_random_forest_train_thresholds.log` | ok | 0.249200 | 0.199200 | 0.877495 | 0.878842 | 31.010 |
| `l2r_bm25_train_thresholds.log` | ok | 0.249500 | 0.249400 | 0.863024 | 0.856337 | 3.282 |
| `l2r_lambdamart_train_thresholds.log` | ok | 0.247000 | 0.099700 | 0.864721 | 0.863052 | 648.040 |
| `l2r_lambdarank_train_thresholds.log` | ok | 0.249700 | 0.240400 | 0.865602 | 0.856829 | 464.299 |
| `l2r_listmle_train_thresholds.log` | ok | 0.250700 | 0.250600 | 0.864257 | 0.863897 | 167.612 |
| `l2r_listnet_train_thresholds.log` | ok | 0.250700 | 0.247500 | 0.860029 | 0.862647 | 16.941 |
| `l2r_ranknet_train_thresholds.log` | ok | 0.249900 | 0.244500 | 0.866053 | 0.857568 | 1932.884 |
| `l2r_xgboost_train_thresholds.log` | ok | 0.248600 | 0.134100 | 0.864053 | 0.867722 | 19.682 |
| `reg_xgboost_regressor_train_thresholds.log` | ok | 0.251200 | 0.222800 | 0.876597 | 0.870359 | 32.152 |
