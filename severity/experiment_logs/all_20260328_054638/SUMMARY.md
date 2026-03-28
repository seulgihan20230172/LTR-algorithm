# Severity 실험 요약
- 로그 디렉터리: `/home/work/lab_project/severity/experiment_logs/all_20260328_054638/logs`
- 로그 파일 수: 12

| 실험(로그 파일) | 상태 | Test Acc | Test Macro F1 | Val MRR | Test MRR | 전체(s) |
|---|---|---:|---:|---:|---:|---:|
| `anomaly_deep_stacked_ae_train_thresholds.log` | ok | 0.252500 | 0.244200 | 0.841427 | 0.853674 | 56.631 |
| `anomaly_denoising_ae_train_thresholds.log` | ok | 0.253400 | 0.247300 | 0.847817 | 0.867678 | 20.145 |
| `anomaly_isolation_forest_train_thresholds.log` | ok | 0.254400 | 0.239100 | 0.874281 | 0.853142 | 6.318 |
| `anomaly_sequence_ae_train_thresholds.log` | ok | 0.254200 | 0.244600 | 0.882588 | 0.841321 | 61.696 |
| `anomaly_vae_train_thresholds.log` | ok | 0.253100 | 0.247900 | 0.864483 | 0.869808 | 15.268 |
| `anomaly_vanilla_ae_train_thresholds.log` | ok | 0.254400 | 0.243300 | 0.891108 | 0.846912 | 20.255 |
| `cls_lightgbm_train_thresholds.log` | ok | 0.253500 | 0.245600 | 0.862993 | 0.865282 | 36.794 |
| `cls_random_forest_train_thresholds.log` | ok | 0.247200 | 0.211500 | 0.874157 | 0.867652 | 27.457 |
| `l2r_listmle_train_thresholds.log` | ok | 0.244900 | 0.244500 | 0.861395 | 0.844995 | 129.323 |
| `l2r_listnet_train_thresholds.log` | ok | 0.248500 | 0.248500 | 0.874840 | 0.892545 | 10.836 |
| `l2r_xgboost_train_thresholds.log` | ok | 0.250500 | 0.130200 | 0.854100 | 0.840628 | 19.483 |
| `reg_xgboost_regressor_train_thresholds.log` | ok | 0.252400 | 0.226800 | 0.876305 | 0.865921 | 28.879 |
