# behavior_graph.py
# 로그를 받아서 provenance(행동) 그래프로 바꾸는 코드

import re
from dataclasses import dataclass
from typing import List, Optional
import networkx as nx


@dataclass
class Event:
    ts: str
    host: str
    action: str
    raw: str

    # 네트워크
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    dst_port: Optional[int] = None

    # 프로세스
    pid: Optional[int] = None
    ppid: Optional[int] = None
    user: Optional[str] = None
    cmd: Optional[str] = None


# (예시 슬라이드 로그에 맞춘 정규식)
RE_FW = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*SRC=(?P<src>\d+\.\d+\.\d+\.\d+)\s+DST=(?P<dst>\d+\.\d+\.\d+\.\d+)\s+DPT=(?P<dpt>\d+)",
    re.IGNORECASE,
)

RE_EDR_PROC = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+(?P<host>[\w\-]+)\s+EDR:\s+process_create.*?\bpid=(?P<pid>\d+)\b.*?\bppid=(?P<ppid>\d+)\b.*?\buser=(?P<user>[\w\-]+)\b.*?cmdline=\"(?P<cmd>.+?)\"",
    re.IGNORECASE,
)

RE_EDR_CHAIN = re.compile(r"parent_chain:\s*(?P<chain>.+)$", re.IGNORECASE)


def parse_logs(lines: List[str], default_host: str = "unknown-host") -> List[Event]:
    """로그 텍스트를 Event 리스트로 파싱"""
    events: List[Event] = []
    last_edr_idx: Optional[int] = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = RE_FW.search(line)
        if m:
            events.append(
                Event(
                    ts=m.group("ts"),
                    host=default_host,
                    action="net_attempt",
                    raw=line,
                    src_ip=m.group("src"),
                    dst_ip=m.group("dst"),
                    dst_port=int(m.group("dpt")),
                )
            )
            continue

        m = RE_EDR_PROC.search(line)
        if m:
            ev = Event(
                ts=m.group("ts"),
                host=m.group("host"),
                action="process_create",
                raw=line,
                pid=int(m.group("pid")),
                ppid=int(m.group("ppid")),
                user=m.group("user"),
                cmd=m.group("cmd"),
            )
            events.append(ev)
            last_edr_idx = len(events) - 1
            continue

        m = RE_EDR_CHAIN.search(line)
        if m and last_edr_idx is not None:
            # 직전 EDR 이벤트에 체인 붙이기(힌트 강화)
            chain = m.group("chain").strip()
            prev = events[last_edr_idx]
            prev.cmd = (prev.cmd or "") + f" | parent_chain: {chain}"
            prev.raw = prev.raw + " " + line
            continue

        # 모르는 로그도 일단 기록
        events.append(Event(ts="unknown", host=default_host, action="unknown", raw=line))

    return events


def _nid(kind: str, value: str) -> str:
    return f"{kind}:{value}"#노드 아이디 통일


def build_provenance_graph(events: List[Event]) -> nx.MultiDiGraph:
    """Event 리스트를 provenance 그래프로 변환"""
    G = nx.MultiDiGraph()

    for ev in events:
        host_n = _nid("host", ev.host)
        G.add_node(host_n, type="host")

        if ev.action == "net_attempt" and ev.src_ip and ev.dst_ip and ev.dst_port is not None:
            src_n = _nid("ip", ev.src_ip)
            dst_n = _nid("ip", ev.dst_ip)
            port_n = _nid("port", str(ev.dst_port))

            G.add_node(src_n, type="ip")
            G.add_node(dst_n, type="ip")
            G.add_node(port_n, type="port")

            # 네트워크 시도(스캔/접속 등)
            G.add_edge(src_n, dst_n, action="net_attempt", ts=ev.ts, dpt=ev.dst_port, raw=ev.raw)
            G.add_edge(dst_n, port_n, action="target_port", ts=ev.ts, raw=ev.raw)
            continue

        if ev.action == "process_create" and ev.pid is not None:
            proc_n = _nid("process", f"{ev.pid}@{ev.host}")
            G.add_node(proc_n, type="process", user=ev.user)

            # 호스트에서 프로세스 생성
            G.add_edge(host_n, proc_n, action="process_create", ts=ev.ts, raw=ev.raw)

            # 부모-자식 프로세스 관계
            if ev.ppid is not None:
                parent_n = _nid("process", f"{ev.ppid}@{ev.host}")
                G.add_node(parent_n, type="process")
                G.add_edge(parent_n, proc_n, action="spawn", ts=ev.ts, raw=ev.raw)

            # 명령 실행 노드
            if ev.cmd:
                cmd_n = _nid("cmd", ev.cmd)
                G.add_node(cmd_n, type="cmd")
                G.add_edge(proc_n, cmd_n, action="exec_cmd", ts=ev.ts, raw=ev.raw)

                # 커맨드에서 파일 경로/모듈 힌트 뽑기
                for path in re.findall(r"(/[\w\-/\.]+)", ev.cmd):
                    if len(path) > 3:
                        file_n = _nid("file", path)
                        G.add_node(file_n, type="file")
                        G.add_edge(proc_n, file_n, action="touches_file", ts=ev.ts, raw=ev.raw)
                        if "insmod" in ev.cmd and path.endswith(".ko"):
                            G.add_edge(proc_n, file_n, action="loads_module", ts=ev.ts, raw=ev.raw)

                # 커맨드에 IP 있으면 connect 힌트
                for ip in re.findall(r"(\d+\.\d+\.\d+\.\d+)", ev.cmd):
                    ip_n = _nid("ip", ip)
                    G.add_node(ip_n, type="ip")
                    G.add_edge(proc_n, ip_n, action="connects_to", ts=ev.ts, raw=ev.raw)

            continue

        # 기타 이벤트는 host에 연결만
        unk_n = _nid("event", ev.raw[:60])
        G.add_node(unk_n, type="event")
        G.add_edge(host_n, unk_n, action="observed", ts=ev.ts, raw=ev.raw)

    return G
