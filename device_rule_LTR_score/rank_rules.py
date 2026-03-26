import re
import math
from typing import List, Dict, Any, Tuple

# =========================================================
# 1) 입력: 장비별 TTP 랭킹 결과 
# =========================================================
DEVICE_TTP_RANK = {
    "FW": {
        "T1046": 3.182926, "T1105": 2.160869, "T1071": 2.059675,
        "T1547.006": 0.777144, "T1059": 0.510713, "T1055": -0.335117
    },
    "EDR": {
        "T1547.006": 3.070326, "T1059": 2.405198, "T1105": 2.178876,
        "T1055": 1.891100, "T1046": 1.012104, "T1071": 0.263288
    },
    "IDS": {
        "T1105": 2.865791, "T1071": 2.740731, "T1046": 2.334770,
        "T1547.006": 0.868695, "T1059": 0.840435, "T1055": 0.172109
    }
}

# =========================================================
# 2) 입력: 장비별 룰
# =========================================================
FW_RULES = [
  {
    "title": "[FW] Internal Network Service Scanning via Dropped Connections (T1046)",
    "logsource": {"product": "linux", "service": "kernel"},
    "detection": {
      "selection_base": {"message|contains|all": ["fw01 kernel:", "DROP", "IN=eth1", "SRC=10.1.2.45", "DST=10.1.2.10"]},
      "selection_ssh": {"message|contains": "DPT=22"},
      "selection_http": {"message|contains": "DPT=80"},
      "condition": "selection_base and (selection_ssh or selection_http)"
    },
    "falsepositives": {"possible_sources": ["Vulnerability scanner", "Network monitoring / port check"]},
    "level": "high"
  }
]

EDR_RULES = [
  {
    "title": "[EDR] Suspicious Kernel Module Load via insmod from /tmp (bpfdoor) (T1547.006)",
    "logsource": {"product": "linux", "category": "process_creation"},
    "detection": {
      "selection_insmod": {"CommandLine|contains|all": ["/usr/bin/insmod", "/tmp/", ".ko"]},
      "selection_keyword": {"CommandLine|contains": "bpfdoor"},
      "condition": "selection_insmod or (selection_insmod and selection_keyword)"
    },
    "falsepositives": {"possible_sources": ["Legitimate kernel module testing by administrators (rare in production)", "Driver installation scripts"]},
    "level": "critical"
  },
  {
    "title": "[EDR] Remote Script Transfer and Immediate Execution via curl | sh (T1105 + T1059)",
    "logsource": {"product": "linux", "category": "process_creation"},
    "detection": {
      "selection_shell": {"CommandLine|contains|all": ["/bin/sh -c", "curl", "http://", "| sh"]},
      "condition": "selection_shell"
    },
    "falsepositives": {"possible_sources": ["DevOps bootstrap/provisioning scripts", "Admin automation using curl pipe to shell (bad practice but exists)"]},
    "level": "high"
  }
]

IDS_RULES = [
  {
    "title": "[IDS] Ingress Tool Transfer over HTTP (T1105) - unknown",
    "logsource": {"product": "unknown", "service": "unknown"},
    "detection": {"selection": "unknown", "condition": "unknown"},
    "falsepositives": {"possible_sources": ["unknown"]},
    "level": "unknown"
  },
  {
    "title": "[IDS] Application Layer Protocol C2 Candidate (T1071) - unknown",
    "logsource": {"product": "unknown", "service": "unknown"},
    "detection": {"selection": "unknown", "condition": "unknown"},
    "falsepositives": {"possible_sources": ["unknown"]},
    "level": "unknown"
  }
]

# =========================================================
# 3) (중요) 장비별 TTP rank_score를 0~1로 정규화
#    - noisy-or 집계를 위해 필요
# =========================================================
def minmax_01(scores: Dict[str, float], eps: float = 1e-9) -> Dict[str, float]:
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    # 모든 값이 같으면(드묾) 0.5로
    if abs(mx - mn) < eps:
        return {k: 0.5 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

DEVICE_TTP_SCORE_01 = {dev: minmax_01(s) for dev, s in DEVICE_TTP_RANK.items()}

# =========================================================
# 4) 룰 품질 Q(r): level/완성도/구체성/FP 위험
# =========================================================
LEVEL_W = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2, "unknown": 0.1}
FP_PENALTY_KW = ["devops", "automation", "admin", "provision", "scanner", "monitor", "testing", "bootstrap"]

def extract_ttps(rule: Dict[str, Any]) -> List[str]:
    # title의 (T1105 + T1059) 같은 패턴에서 뽑음
    return re.findall(r"T\d+(?:\.\d+)?", rule.get("title") or "")

def is_unknown(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() == "unknown"
    if isinstance(v, dict):
        return any(is_unknown(x) for x in v.values())
    if isinstance(v, list):
        return any(is_unknown(x) for x in v)
    return False

def completeness(logsource: Dict[str, Any], detection: Dict[str, Any]) -> float:
    # IDS unknown 템플릿이면 0
    if is_unknown(logsource) or is_unknown(detection):
        return 0.0
    product = (logsource.get("product") or "").strip().lower()
    has_product = (product not in ["", "unknown"])
    has_type = any((logsource.get(k) or "").strip().lower() not in ["", "unknown"] for k in ["service", "category"])
    return 1.0 if (has_product and has_type) else 0.6

def specificity(detection: Dict[str, Any]) -> float:
    # selection_* 항목이 많고 필드가 구체적일수록 가점 (0~1)
    if not isinstance(detection, dict):
        return 0.0
    n = 0
    for k, v in detection.items():
        if k.startswith("selection"):
            n += 1
            if isinstance(v, dict):
                n += len(v.keys())
    return min(1.0, n / 8.0)

def fp_risk(falsepositives: Any) -> float:
    txt = str(falsepositives).lower()
    hits = sum(1 for kw in FP_PENALTY_KW if kw in txt)
    return min(1.0, 0.15 * hits)

def rule_quality(rule: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    level = (rule.get("level") or "unknown").lower()
    sev = LEVEL_W.get(level, 0.1)
    comp = completeness(rule.get("logsource") or {}, rule.get("detection") or {})
    spec = specificity(rule.get("detection") or {})
    fp = fp_risk(rule.get("falsepositives"))
    q = 0.45 * sev + 0.30 * comp + 0.25 * spec - 0.25 * fp
    return q, {"severity": sev, "completeness": comp, "specificity": spec, "fp_risk": fp}

# =========================================================
# 5) TTP coverage 집계: device별 s_d(t) 사용
#    - noisy-or: 1 - Π(1 - s)
#    - + coverage_count_bonus (커버하는 TTP 수 보너스)
# =========================================================
def noisy_or(ttps: List[str], ttp_score01: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    known = []
    unknown = []
    prod = 1.0
    for t in ttps:
        if t in ttp_score01:
            s = float(ttp_score01[t])
            known.append((t, s))
            prod *= (1.0 - s)
        else:
            unknown.append(t)
    if not known:
        return 0.0, {"known": known, "unknown": unknown}
    return 1.0 - prod, {"known": known, "unknown": unknown}

def coverage_count_bonus(known_count: int, max_k: int = 5) -> float:
    # log(1+k) 형태로 완만하게 증가 (0~1 근처)
    if known_count <= 0:
        return 0.0
    return math.log(1 + known_count) / math.log(1 + max_k)

def unknown_penalty(rule: Dict[str, Any]) -> float:
    # IDS unknown 템플릿처럼 logsource/detection이 unknown이면 크게 패널티
    pen = 0.0
    if is_unknown(rule.get("logsource")) or is_unknown(rule.get("detection")):
        pen += 0.8
    return min(1.0, pen)

# =========================================================
# 6) 최종 룰 점수 (장비별 TTP 점수 반영)
#    RuleScore = a*Q + b*Coverage(noisy-or) + c*CountBonus - d*UnknownPenalty
# =========================================================
def rule_score(rule: Dict[str, Any], device: str,
               a=0.40, b=0.50, c=0.20, d=0.35) -> Dict[str, Any]:
    ttps = extract_ttps(rule)
    q, q_parts = rule_quality(rule)

    ttp_score01 = DEVICE_TTP_SCORE_01[device]
    cov, cov_parts = noisy_or(ttps, ttp_score01)

    k = len(cov_parts["known"])
    bonus = coverage_count_bonus(k)

    pen = unknown_penalty(rule)

    score = a*q + b*cov + c*bonus - d*pen

    return {
        "device": device,
        "title": rule.get("title"),
        "ttps": ttps,
        "rule_score": round(float(score), 4),
        "explain": {
            "Q_parts": {k: round(float(v), 4) for k, v in q_parts.items()},
            "Q_value": round(float(q), 4),
            "device_ttp_scores_01_used": cov_parts["known"],  # (TTP, normalized score)
            "coverage_noisy_or": round(float(cov), 4),
            "coverage_count_bonus": round(float(bonus), 4),
            "unknown_penalty": round(float(pen), 4),
        }
    }

def rank_rules(rules: List[Dict[str, Any]], device: str) -> List[Dict[str, Any]]:
    scored = [rule_score(r, device) for r in rules]
    scored.sort(key=lambda x: x["rule_score"], reverse=True)
    return scored

# =========================================================
# 7) 실행: 장비별 룰 랭킹
# =========================================================
if __name__ == "__main__":
    DEVICE_RULES = {"FW": FW_RULES, "EDR": EDR_RULES, "IDS": IDS_RULES}

    for dev in ["FW", "EDR", "IDS"]:
        ranked = rank_rules(DEVICE_RULES[dev], dev)
        print(f"\n=== Rule ranking for device={dev} (uses device-specific TTP ranking) ===")
        for i, r in enumerate(ranked, 1):
            print(f"{i}. score={r['rule_score']} | ttps={r['ttps']} | {r['title']}")
            print(f"   explain={r['explain']}")
