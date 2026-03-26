#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[목적]
이미 생성된 LTR JSONL(가장 최신 data_*.jsonl)을 후처리해서

1) label=2 없는 query는 학습용(clean) 출력에서 제외
2) 제외된 query는 no_match_ttp 폴더에 따로 저장
3) DF(표)로 query / label1,2 TTP를 보기 좋게 저장
   - STIX에 없는 TTP는 DF에 저장하지 않음(= features/labels/candidates 상에 있지만 ttp_name이 비어있거나 없는 경우 제외)

[입력]
device_rule_LTR_score/dataset/data_*.jsonl 중 가장 최신 파일

[출력]
- clean JSONL: device_rule_LTR_score/dataset/clean/clean_<원본파일명>
- no_match JSONL: device_rule_LTR_score/dataset/no_match_ttp/no_match_<원본파일명>
- DF CSV 2개:
  - device_rule_LTR_score/dataset/df/pairs_label12_<원본파일명>.csv
  - device_rule_LTR_score/dataset/df/queries_summary_<원본파일명>.csv
"""

import os
import re
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple


# =========================
# CONFIG
# =========================
DATASET_DIR = "device_rule_LTR_score/dataset"
INPUT_GLOB_PREFIX = "data_"        # data_YYYYMMDD_....jsonl 형태
INPUT_SUFFIX = ".jsonl"

OUT_CLEAN_DIR = os.path.join(DATASET_DIR, "clean")
OUT_NO_MATCH_DIR = os.path.join(DATASET_DIR, "no_match_ttp")
OUT_DF_DIR = os.path.join(DATASET_DIR, "df")

# label=2 없는 query를 제외할지(요구사항: True)
DROP_WITHOUT_LABEL2 = True

# DF에 저장할 때 "STIX에 없는 TTP"를 어떻게 판정할지
# - 기본: ttp_name이 비어있으면 STIX에 없는 것으로 보고 제외
# - (너가 ttp_name을 record에 안 넣었다면) 아래 fallback 규칙 적용
DROP_IF_NO_TTP_NAME = True


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def find_latest_jsonl(dataset_dir: str) -> str:
    """
    dataset_dir에서 data_*.jsonl 중 가장 최신(수정시간 기준) 파일 경로 반환
    """
    candidates = []
    for fn in os.listdir(dataset_dir):
        if fn.startswith(INPUT_GLOB_PREFIX) and fn.endswith(INPUT_SUFFIX):
            path = os.path.join(dataset_dir, fn)
            if os.path.isfile(path):
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"{dataset_dir} 아래에 {INPUT_GLOB_PREFIX}*{INPUT_SUFFIX} 파일이 없습니다.")

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def safe_get_ttp_name(record: Dict, idx: int) -> str:
    """
    record에서 후보 idx에 해당하는 TTP 이름을 뽑아오려는 함수.
    너의 현재 JSONL 구조에는 ttp_name이 없을 수 있어서 몇 가지 fallback을 둠.

    우선순위:
    1) record["features"][idx] 안에 "ttp_name"이 있다면 사용 (없으면 skip)
    2) record에 "ttp_meta" 같은 게 있다면 거기서 찾기 (현재 코드는 없음)
    3) 없으면 빈 문자열 반환
    """
    try:
        feat = record.get("features", [])[idx]
        if isinstance(feat, dict):
            name = feat.get("ttp_name", "")
            if isinstance(name, str):
                return name.strip()
    except Exception:
        pass
    return ""


def write_jsonl(path: str, records: List[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ensure_dir(OUT_CLEAN_DIR)
    ensure_dir(OUT_NO_MATCH_DIR)
    ensure_dir(OUT_DF_DIR)

    in_path = find_latest_jsonl(DATASET_DIR)
    base = os.path.basename(in_path)

    clean_path = os.path.join(OUT_CLEAN_DIR, f"clean_{base}")
    no_match_path = os.path.join(OUT_NO_MATCH_DIR, f"no_match_{base}")

    pairs_csv = os.path.join(OUT_DF_DIR, f"pairs_label12_{base}.csv")
    queries_csv = os.path.join(OUT_DF_DIR, f"queries_summary_{base}.csv")

    kept_records: List[Dict] = []
    dropped_records: List[Dict] = []

    # DF용
    pair_rows: List[Dict] = []
    query_rows: List[Dict] = []

    kept = 0
    dropped = 0
    total = 0

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)

            labels = rec.get("relevance_labels", [])
            cands = rec.get("ttp_candidate_list", [])

            has2 = 2 in labels
            has1 = 1 in labels

            # label1/2 ttp 수집 (DF용)
            label2_ttps = []
            label1_ttps = []

            for i, (ttp, y) in enumerate(zip(cands, labels)):
                if y not in (1, 2):
                    continue

                # "없는 TTP는 저장 안 함" 처리
                # - 현재 레코드에 ttp_name이 없을 수 있으니,
                #   1) features[i]["ttp_name"] 같은게 있으면 그걸로 판정
                #   2) 없으면 이름은 빈칸으로 두되, DROP_IF_NO_TTP_NAME=True면 제외
                ttp_name = safe_get_ttp_name(rec, i)

                if DROP_IF_NO_TTP_NAME and not ttp_name:
                    # STIX에 없는 TTP라고 보고 DF에서 제외
                    continue

                if y == 2:
                    label2_ttps.append(ttp)
                else:
                    label1_ttps.append(ttp)

                # query-ttp 단위 row
                feat = rec.get("features", [])
                feat_i = feat[i] if i < len(feat) and isinstance(feat[i], dict) else {}
                pair_rows.append({
                    "query_id": rec.get("query_id", ""),
                    "device": rec.get("device", ""),
                    "rule_title": rec.get("rule_title", ""),
                    "label": y,
                    "ttp_id": ttp,
                    "ttp_name": ttp_name,
                    "text_sim": feat_i.get("text_sim", ""),
                    "tactic_overlap": feat_i.get("tactic_overlap", ""),
                    "is_exact_match": feat_i.get("is_exact_match", ""),
                })

            # query 단위 요약 row
            query_rows.append({
                "query_id": rec.get("query_id", ""),
                "device": rec.get("device", ""),
                "rule_title": rec.get("rule_title", ""),
                "has_label_2": int(has2),
                "has_label_1": int(has1),
                "label2_ttps": ";".join(label2_ttps),
                "label1_ttps": ";".join(label1_ttps),
            })

            # 분기 저장
            if DROP_WITHOUT_LABEL2 and not has2:
                dropped_records.append(rec)
                dropped += 1
            else:
                kept_records.append(rec)
                kept += 1

    # 저장
    write_jsonl(clean_path, kept_records)
    write_jsonl(no_match_path, dropped_records)

    write_csv(
        pairs_csv,
        pair_rows,
        fieldnames=["query_id", "device", "rule_title", "label", "ttp_id", "ttp_name", "text_sim", "tactic_overlap", "is_exact_match"]
    )
    write_csv(
        queries_csv,
        query_rows,
        fieldnames=["query_id", "device", "rule_title", "has_label_2", "has_label_1", "label2_ttps", "label1_ttps"]
    )

    print("✅ 입력(가장 최신):", in_path)
    print("✅ clean 저장:", clean_path, f"(kept={kept}/{total})")
    print("🗂️ no_match 저장:", no_match_path, f"(dropped={dropped}/{total})")
    print("📄 DF(pairs) 저장:", pairs_csv, f"(rows={len(pair_rows)})")
    print("📄 DF(queries) 저장:", queries_csv, f"(rows={len(query_rows)})")
    print("\n주의: 현재 레코드에 ttp_name이 없으면, DF에 TTP가 빠질 수 있습니다.")
    print("      (원하면 STIX를 다시 읽어서 ttp_id->name 매핑을 붙여주는 버전으로 확장 가능)")


if __name__ == "__main__":
    main()
