#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

[목적]
LLM이 생성한 보안 룰(query)을 기준으로,
ATT&CK TTP(document) 후보들을 랭킹하는
Learning-to-Rank(LTR) 실험용 데이터셋을 자동 생성한다.

[입력]
1) Sigma 룰 디렉토리
   - 룰 텍스트 (title, description, detection)
   - tags (attack.tXXXX 형태)
   - logsource (장비 추론용)

2) MITRE ATT&CK STIX JSON
   - 전체 TTP(Technique) 메타데이터

[기능]
Sigma 룰(YAML) + MITRE ATT&CK STIX(JSON) → LTR 데이터셋(JSONL) 자동 생성

[사용법]
1) 코드 상단의 CONFIG 섹션만 수정
2) 실행: python 260203_data.py
   - out 파일은 자동으로 outputs/ltr_dataset_YYYYMMDD_HHMMSS.jsonl 로 생성됨

[출력(JSONL)]
각 줄 = 룰 1개(query 1개)
{
  "device": "FW"/"EDR"/"IDS"/...,
  "query_id": "...",
  "rule_title": "...",
  "rule_text": "...",
  "positive_ttps": ["Txxxx", ...],
  "ttp_candidate_list": ["Txxxx", ...],
  "relevance_labels": [2,0,1,...],
  "features": [{"text_sim":..., ...}, ...],
  "random_doc_scores": [0.12, 0.98, ...]
}
"""

# =========================================================
# CONFIG (여기만 바꾸면 됨)
# =========================================================

# Sigma 룰 폴더(와일드카드 X, 폴더로 지정)
SIGMA_DIR = "sigma/rules"

# ATT&CK STIX JSON 경로 (ics-bundle.json 등)
ATTACK_STIX_PATH = "mitreattack-python/tests/resources/enterprise-bundle.json"

# 출력 폴더 및 파일 prefix
OUT_DIR = "device_rule_LTR_score/dataset"
OUT_PREFIX = "data"

# 데이터 생성 파라미터
CANDIDATES_PER_QUERY = 80   # query(룰)당 후보 TTP 수
SEED = 7                    # 랜덤 시드
MIN_POS = 1                 # tags에서 TTP가 최소 몇 개 이상 있어야 포함할지 (1이면 정답이 있는 룰만 사용)

# 약한 라벨 생성 옵션
SIM_THRESHOLD = 0.18        # rule_text vs ttp_text 유사도가 이 값 이상이면 relevance=1 (exact match는 2)

# (참고) LightGBM 튜닝 파라미터 모음(데이터 생성에는 사용 X)
# 나중에 LambdaMART 학습 코드에서 그대로 가져다 쓰면 됨.
LIGHTGBM_TUNING = {
    # 트리 구조
    "max_depth": 8,
    "num_leaves": 31,
    "min_child_samples": 20,     # (= min_data_in_leaf)
    "min_child_weight": 1e-3,    # leaf를 나눌 때 “그 leaf에 쌓인 weight”가 너무 작으면 분할을 막는 역할

    # 샘플링 비율
    "subsample": 0.8,            # 트리 하나 만들 때 학습 데이터의 80%만 랜덤 사용
    "colsample_bytree": 0.8,     # 트리 하나 만들 때 feature의 80%만 랜덤 사용

    # 규제 (과적합 방지)
    "reg_lambda": 1.0,           # L2 가중치가 너무 커지는 걸 벌줌
    "reg_alpha": 0.0,            # L1
}

# =========================================================
# 아래부터는 보통 수정할 필요 없음
# =========================================================

import os
import re
import json
import random
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # PyYAML
except ImportError:
    raise SystemExit("PyYAML이 필요합니다. 설치: pip install --user pyyaml")


def safe_read_text(path: str) -> str:
    # 파일 첫 바이트로 BOM(인코딩) 감지
    with open(path, "rb") as f:
        raw = f.read()

    # BOM 기반 감지 (가장 확실)
    if raw.startswith(b"\xff\xfe"):
        # UTF-16 LE
        return raw.decode("utf-16")
    if raw.startswith(b"\xfe\xff"):
        # UTF-16 BE
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        # UTF-8 with BOM
        return raw.decode("utf-8-sig")

    # 그 외는 UTF-8로 시도 (안되면 ignore로라도 읽게)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="ignore")


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def simple_tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", (s or "").lower())


def cosine_bow(a_tokens: List[str], b_tokens: List[str]) -> float:
    from collections import Counter
    a = Counter(a_tokens)
    b = Counter(b_tokens)
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


# ---------------------------
# ATT&CK STIX 파싱
# ---------------------------

@dataclass
class Technique:
    ttp_id: str
    name: str
    description: str
    tactics: List[str]
    platforms: List[str]


def load_attack_stix(stix_path: str) -> Dict[str, Technique]:
    """
    STIX bundle에서 attack-pattern만 추출하고,
    external_references[].external_id 가 Txxxx 형태인 것만 Technique로 사용
    """
    data = json.loads(safe_read_text(stix_path))
    objects = data.get("objects", [])

    ttp_map: Dict[str, Technique] = {}

    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue

        ttp_id = None
        for ref in (obj.get("external_references") or []):
            ext = ref.get("external_id")
            if isinstance(ext, str) and re.match(r"^T\d{4}(\.\d{3})?$", ext):
                ttp_id = ext
                break
        if not ttp_id:
            continue

        tactics = []
        for kcp in (obj.get("kill_chain_phases") or []):
            phase = kcp.get("phase_name")
            if isinstance(phase, str) and phase:
                tactics.append(phase)

        platforms = obj.get("x_mitre_platforms") or []
        if not isinstance(platforms, list):
            platforms = []

        ttp_map[ttp_id] = Technique(
            ttp_id=ttp_id,
            name=obj.get("name", "") or "",
            description=obj.get("description", "") or "",
            tactics=sorted(set(tactics)),
            platforms=sorted(set([p for p in platforms if isinstance(p, str)])),
        )

    if not ttp_map:
        raise ValueError(f"STIX에서 Technique를 못 찾았습니다: {stix_path}")

    return ttp_map


# ---------------------------
# Sigma 룰 파싱
# ---------------------------

@dataclass
class SigmaRule:
    query_id: str
    title: str
    rule_text: str
    tags: List[str]
    logsource: Dict
    device: str
    positive_ttps: List[str]


def infer_device(logsource: Dict, tags: List[str]) -> str:
    """
    Sigma YAML의 logsource와 tags를 보고 FW/IDS/EDR/HOST/GENERIC 중 하나로 분류
    """
    product = str((logsource or {}).get("product", "")).lower()
    service = str((logsource or {}).get("service", "")).lower()
    category = str((logsource or {}).get("category", "")).lower()
    tags_l = " ".join([str(t).lower() for t in (tags or [])])

    if any(k in service for k in ["kernel", "iptables", "ufw"]) or "firewall" in tags_l:
        return "FW"
    if any(k in category for k in ["network", "proxy"]) or any(k in tags_l for k in ["network", "dns", "http"]):
        return "IDS"
    if product == "windows" or "process_creation" in category or "sysmon" in service or "edr" in tags_l:
        return "EDR"
    if product == "linux" or "auditd" in service:
        return "HOST"
    return "GENERIC"


def extract_ttps_from_tags(tags: List[str]) -> List[str]:
    """
    tags에서 attack.tXXXX / attack.tXXXX.YYY 형태 추출 → TXXXX(.YYY)
    """
    out = []
    for t in tags or []:
        t = str(t).strip().lower()
        m = re.search(r"attack\.t(\d{4})(?:\.(\d{3}))?", t)
        if m:
            base = f"T{m.group(1)}"
            if m.group(2):
                base = f"{base}.{m.group(2)}"
            out.append(base)
    return sorted(set(out))


def load_sigma_rules_from_dir(sigma_dir: str) -> List[SigmaRule]:
    """
    sigma_dir 아래의 .yml/.yaml 파일을 재귀적으로 모두 읽음
    """
    rules: List[SigmaRule] = []

    if not os.path.isdir(sigma_dir):
        raise ValueError(f"SIGMA_DIR이 폴더가 아닙니다: {sigma_dir}")

    for root, _, files in os.walk(sigma_dir):
        for fn in files:
            if not fn.lower().endswith((".yml", ".yaml")):
                continue
            path = os.path.join(root, fn)
            try:
                doc = yaml.safe_load(safe_read_text(path))
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue

            title = normalize_text(doc.get("title", "")) or os.path.splitext(fn)[0]
            description = normalize_text(doc.get("description", ""))

            tags = doc.get("tags") or []
            if not isinstance(tags, list):
                tags = []

            logsource = doc.get("logsource") or {}
            if not isinstance(logsource, dict):
                logsource = {}

            positive_ttps = extract_ttps_from_tags(tags)
            device = infer_device(logsource, tags)

            detection = doc.get("detection")
            try:
                det_text = json.dumps(detection, ensure_ascii=False)
            except Exception:
                det_text = str(detection)

            rule_text = normalize_text(" | ".join([title, description, det_text]))

            rid = doc.get("id")
            if not isinstance(rid, str) or not rid.strip():
                rid = f"{device}:{os.path.relpath(path, sigma_dir)}"

            rules.append(SigmaRule(
                query_id=rid,
                title=title,
                rule_text=rule_text,
                tags=[str(x) for x in tags],
                logsource=logsource,
                device=device,
                positive_ttps=positive_ttps
            ))

    return rules


# ---------------------------
# 후보 생성 + feature/label 생성
# ---------------------------

def choose_candidates(all_ttps: List[str], positive_ttps: List[str], k: int, rng: random.Random) -> List[str]:
    positives = [t for t in positive_ttps if t in set(all_ttps)]
    positives = sorted(set(positives))

    if k <= len(positives):
        return positives[:k]

    remaining = [t for t in all_ttps if t not in set(positives)]
    rng.shuffle(remaining)
    return positives + remaining[: (k - len(positives))]
    #정답을 먼저 넣고 나머지는 랜덤으로 섞어서 k개 될 때까지 채움

def build_features_and_labels(
    rule: SigmaRule,
    candidates: List[str],
    ttp_map: Dict[str, Technique],
    sim_threshold: float
) -> Tuple[List[int], List[Dict]]:
    """
    약한 라벨 규칙:
      - 정확히 tags로 매칭된 TTP: 2
      - (같은 tactic) 또는 (텍스트 유사도 >= threshold): 1
      - 그 외: 0
    """
    rule_tokens = simple_tokenize(rule.rule_text)

    # sigma tags에 attack.execution 같은 tactic tag가 있을 수 있음
    rule_tactics = []
    for t in rule.tags:
        tl = str(t).lower().strip()
        m = re.search(r"attack\.([a-z_]+)$", tl)
        if m and not m.group(1).startswith("t"):
            rule_tactics.append(m.group(1))
    rule_tactics = sorted(set(rule_tactics))

    positives_set = set(rule.positive_ttps)

    labels: List[int] = []
    feats: List[Dict] = []

    for ttp in candidates:
        tech = ttp_map.get(ttp)
        if not tech:
            labels.append(0)
            feats.append({"text_sim": 0.0, "tactic_overlap": 0, "is_exact_match": 0})
            continue

        tech_text = f"{tech.name} {tech.description}"
        sim = cosine_bow(rule_tokens, simple_tokenize(tech_text))
        tactic_overlap = len(set(rule_tactics) & set(tech.tactics))
        is_exact = 1 if ttp in positives_set else 0

        if is_exact:
            y = 2
        elif tactic_overlap > 0 or sim >= sim_threshold:
            y = 1
        else:
            y = 0

        labels.append(y)
        feats.append({
            "text_sim": float(sim),
            "tactic_overlap": int(tactic_overlap),
            "is_exact_match": int(is_exact),
        })

    return labels, feats


def make_out_path(out_dir: str, prefix: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{prefix}_{ts}.jsonl")


def main():
    # 선택: 그래도 out만 바꾸고 싶을 때를 대비해 아주 최소 옵션만 남김
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="", help="원하면 직접 출력 파일명 지정. 기본은 timestamp 자동 생성.")
    args = parser.parse_args()

    rng = random.Random(SEED)

    print("[1/3] ATT&CK STIX 로딩...")
    ttp_map = load_attack_stix(ATTACK_STIX_PATH)
    all_ttps = sorted(ttp_map.keys())
    print(f"  - Techniques: {len(all_ttps)}")

    print("[2/3] Sigma 룰 로딩...")
    rules = load_sigma_rules_from_dir(SIGMA_DIR)
    if MIN_POS > 0:
        rules = [r for r in rules if len(r.positive_ttps) >= MIN_POS]
    print(f"  - Rules (filtered): {len(rules)}")

    out_path = args.out.strip() or make_out_path(OUT_DIR, OUT_PREFIX)

    print("[3/3] 데이터셋 생성...")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rules:
            candidates = choose_candidates(all_ttps, r.positive_ttps, CANDIDATES_PER_QUERY, rng)
            labels, feats = build_features_and_labels(r, candidates, ttp_map, SIM_THRESHOLD)

            # 요청대로 임의의 “장비별 TTP 랭킹 점수(document score)”를 랜덤 생성
            random_doc_scores = [rng.random() for _ in candidates]

            record = {
                "device": r.device,
                "query_id": r.query_id,
                "rule_title": r.title,
                "rule_text": r.rule_text,
                "positive_ttps": r.positive_ttps,
                "ttp_candidate_list": candidates,
                "relevance_labels": labels,
                "features": feats,
                "random_doc_scores": random_doc_scores,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"완료: {out_path}")
    print("   (각 줄 = query 1개, candidates/labels/features/doc_scores는 인덱스로 정렬되어 대응)")


if __name__ == "__main__":
    main()
