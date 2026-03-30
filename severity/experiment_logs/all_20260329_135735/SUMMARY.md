# Severity 실험 요약
- 로그 디렉터리: `/home/work/lab_project/severity/experiment_logs/all_20260329_135735/logs`
- 로그 파일 수: 16

| 실험(로그 파일) | 상태 | Test Acc | Test Macro F1 | Val MRR | Test MRR | 전체(s) |
|---|---|---:|---:|---:|---:|---:|
| `anomaly_deep_stacked_ae_test_oracle_ratio.log` | ok | 0.256500 | 0.256300 | 0.857635 | 0.853322 | 388.274 |
| `anomaly_denoising_ae_test_oracle_ratio.log` | ok | 0.248700 | 0.248600 | 0.841567 | 0.839271 | 123.059 |
| `anomaly_isolation_forest_test_oracle_ratio.log` | ok | 0.250200 | 0.250100 | 0.859381 | 0.888224 | 30.148 |
| `anomaly_sequence_ae_test_oracle_ratio.log` | ok | 0.246600 | 0.246500 | 0.873752 | 0.862475 | 569.952 |
| `anomaly_vae_test_oracle_ratio.log` | ok | 0.249000 | 0.249000 | 0.852645 | 0.852645 | 84.909 |
| `anomaly_vanilla_ae_test_oracle_ratio.log` | ok | 0.249200 | 0.249200 | 0.852994 | 0.861028 | 121.563 |
| `cls_lightgbm_test_oracle_ratio.log` | ok |  |  |  |  |  |
| `cls_random_forest_test_oracle_ratio.log` | ok |  |  |  |  |  |
| `l2r_bm25_test_oracle_ratio.log` | ok | 0.249900 | 0.249800 | 0.863024 | 0.856337 | 3.507 |
| `l2r_lambdamart_test_oracle_ratio.log` | ok | 0.249400 | 0.249400 | 0.867793 | 0.864970 | 2724.554 |
| `l2r_lambdarank_test_oracle_ratio.log` | ok | 0.246400 | 0.246300 | 0.869261 | 0.840918 | 3097.181 |
| `l2r_listmle_test_oracle_ratio.log` | ok | 0.246900 | 0.246900 | 0.868263 | 0.861527 | 843.777 |
| `l2r_listnet_test_oracle_ratio.log` | ok | 0.247400 | 0.247300 | 0.877495 | 0.872683 | 113.691 |
| `l2r_ranknet_test_oracle_ratio.log` | ok | 0.249000 | 0.249000 | 0.862275 | 0.881737 | 13888.274 |
| `l2r_xgboost_test_oracle_ratio.log` | ok | 0.249500 | 0.249300 | 0.854042 | 0.860030 | 26.156 |
| `reg_xgboost_regressor_test_oracle_ratio.log` | ok | 0.251400 | 0.251200 | 0.876597 | 0.870359 | 113.186 |
