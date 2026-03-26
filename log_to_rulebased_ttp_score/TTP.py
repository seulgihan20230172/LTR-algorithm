# TTP.py
# 룰 기반으로 ATT&CK TTP 후보를 뽑는 코드 (처음엔 단순 규칙)

from typing import Any, Dict, List
from collections import defaultdict


def extract_ttp_candidates(events) -> List[Dict[str, Any]]:
    """
    return:
      [{"technique_id": "...", "name": "...", "score": 0~1, "evidence": [...]}, ...]
    """
    cand = defaultdict(lambda: {"technique_id": "", "name": "", "score": 0.0, "evidence": []})

    def add(tech_id: str, name: str, score: float, evidence: str):
        item = cand[tech_id]
        item["technique_id"] = tech_id
        item["name"] = name
        item["score"] = min(1.0, item["score"] + score)
        item["evidence"].append(evidence)

    for ev in events:
        raw = (ev.raw or "").lower()
        cmd = (ev.cmd or "").lower()

        # 내부 포트 스캔/스캔 알람
        if "portscan" in raw or (ev.action == "net_attempt" and ev.dst_port in {22, 80, 443, 445, 3389}):
            add("T1046", "Network Service Scanning", 0.6, f"[{ev.ts}] {ev.raw}")

        # curl|sh 형태 다운로드 후 실행
        if "curl" in cmd and ("| sh" in cmd or "sh -c" in cmd):
            add("T1105", "Ingress Tool Transfer", 0.8, f"[{ev.ts}] {ev.cmd}")
            add("T1059", "Command and Scripting Interpreter", 0.6, f"[{ev.ts}] {ev.cmd}")

        # insmod + .ko 로 커널 모듈 로드
        if "insmod" in cmd and ".ko" in cmd:
            add("T1547.006", "Kernel Modules and Extensions", 0.9, f"[{ev.ts}] {ev.cmd}")

        # bpfdoor 키워드(후속 확장용: 지금은 후보로만 가산)
        if "bpfdoor" in raw or "bpfdoor" in cmd:
            add("T1071", "Application Layer Protocol (Candidate)", 0.3, f"[{ev.ts}] keyword=bpfdoor")
            add("T1055", "Process Injection (Candidate)", 0.2, f"[{ev.ts}] keyword=bpfdoor")

    out = list(cand.values())
    out.sort(key=lambda x: x["score"], reverse=True)
    return out
