from __future__ import annotations

import argparse
import ctypes
import os
import sys
from ctypes import wintypes

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from typing import Optional

from metrics import evaluate_all


def group_by(data, qid_index):
    qid_doc_map = {}
    idx = 0
    for record in data:
        qid_doc_map.setdefault(record[qid_index], [])
        qid_doc_map[record[qid_index]].append(idx)
        idx += 1
    return qid_doc_map


def _rss_bytes() -> int:
    """프로세스 RSS(바이트). Windows는 ctypes로 조회."""
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo  # type: ignore[attr-defined]
        GetProcessMemoryInfo.restype = wintypes.BOOL
        GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD]

        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess  # type: ignore[attr-defined]
        GetCurrentProcess.restype = wintypes.HANDLE
        GetCurrentProcess.argtypes = []

        ok = bool(GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb))
        return int(counters.WorkingSetSize) if ok else 0
    try:
        import resource  # noqa: PLC0415

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r if sys.platform == "darwin" else r * 1024)
    except Exception:
        return 0


def _mem(tag: str) -> None:
    """메모리 로그(짧은 1줄). env RANKNET_MEMLOG=1 일 때만 출력."""
    if os.environ.get("RANKNET_MEMLOG", "0") not in ("1", "true", "TRUE", "yes", "YES"):
        return
    rss = _rss_bytes()
    mb = rss / (1024 * 1024) if rss else 0.0
    print(f"[RANKNET_MEM] {tag} rss={mb:.1f}MB", flush=True)


def _iter_pairs_indices(scores: np.ndarray):
    """원본 get_pairs(scores)와 동일한 ordered pairs를 스트리밍 생성 (샘플링 없음)."""
    s = np.asarray(scores)
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(n):
            if si > s[j]:
                yield i, j


def _stream_pair_batches(scores: np.ndarray, *, batch_size: int):
    """(i,j) pair를 batch_size씩 묶어 yield (리스트를 크게 쌓지 않음)."""
    a: list[int] = []
    b: list[int] = []
    for i, j in _iter_pairs_indices(scores):
        a.append(i)
        b.append(j)
        if len(a) >= batch_size:
            yield a, b
            a, b = [], []
    if a:
        yield a, b


class Model(torch.nn.Module):
    def __init__(self, n_feature, h1_units, h2_units):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_feature, h1_units),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(h1_units, h2_units),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(h2_units, 1),
        )
        self.output_sig = torch.nn.Sigmoid()

    def forward(self, input_1, input_2):
        s1 = self.model(input_1)
        s2 = self.model(input_2)
        out = self.output_sig(s1 - s2)
        return out


class RankNet:
    def __init__(
        self,
        n_feature,
        h1_units,
        h2_units,
        epoch,
        learning_rate,
        plot=True,
        *,
        batch_size: int = 4096,
        eval_every: int = 1,
    ):
        self.n_feature = n_feature
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.model = Model(n_feature, h1_units, h2_units)
        self.epoch = epoch
        self.plot = plot
        self.learning_rate = learning_rate
        self.batch_size = int(batch_size)
        self.eval_every = int(eval_every)

    def decay_learning_rate(self, optimizer, epoch, decay_rate):
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * decay_rate

    #def fit(self, training_data, qids_per_chunk: int | None = None):
    def fit(self, training_data, qids_per_chunk: Optional[int] = None):
        """샘플링 0, pair 전부 유지. 메모리 절감:

        - qid별로 Xq만 올리고, pair는 스트리밍(batch)으로 생성
        - batch마다 backward만 누적하고(gradient accumulation),
          에폭당 optimizer.step() 1회(full-batch 평균 loss gradient에 가깝게)

        training_data 행은 ``sort_ltr_rows_by_qid`` 와 동일 순서(qid 오름차순)일 것.
        ``qids_per_chunk``: 0 또는 None이면 모든 qid를 한 청크로 처리한다.
        """
        net = self.model
        net.train()

        qid_doc_map = group_by(training_data, 1)
        query_idx = sorted(qid_doc_map.keys(), key=lambda k: (float(np.asarray(k).item()),))
        chunk_n = len(query_idx) if not qids_per_chunk or int(qids_per_chunk) <= 0 else int(qids_per_chunk)
        _mem("after_group_by")

        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fun = torch.nn.BCELoss(reduction="sum")
        loss_list: list[float] = []

        if self.plot:
            plt.ion()

        best_score = -1.0
        best_state = None

        print("Traning………………\n")
        for epoch in range(self.epoch):
            self.decay_learning_rate(optimizer, epoch, 0.95)
            optimizer.zero_grad()

            total_pairs = 0
            total_loss_sum = 0.0

            for c0 in range(0, len(query_idx), chunk_n):
                c1 = min(c0 + chunk_n, len(query_idx))
                for qid in query_idx[c0:c1]:
                    idxs = qid_doc_map[qid]
                    if len(idxs) < 2:
                        continue

                    yq = np.asarray(training_data[idxs, 0], dtype=np.float64)
                    Xq = training_data[idxs, 2:].astype(np.float32, copy=False)
                    Xq_t = torch.from_numpy(Xq)

                    for a_list, b_list in _stream_pair_batches(yq, batch_size=self.batch_size):
                        a = torch.tensor(a_list, dtype=torch.int64)
                        b = torch.tensor(b_list, dtype=torch.int64)
                        X1 = Xq_t.index_select(0, a)
                        X2 = Xq_t.index_select(0, b)
                        y = torch.ones((len(a_list), 1), dtype=torch.float32)

                        y_pred = net(X1, X2)
                        loss_sum = loss_fun(y_pred, y)
                        loss_sum.backward()
                        total_pairs += int(len(a_list))
                        total_loss_sum += float(loss_sum.item())

            if total_pairs > 0:
                inv = 1.0 / float(total_pairs)
                for p in net.parameters():
                    if p.grad is not None:
                        p.grad.mul_(inv)
                optimizer.step()

            avg_loss = total_loss_sum / float(total_pairs) if total_pairs else float("nan")
            loss_list.append(avg_loss)
            _mem(f"after_epoch_step e={epoch} pairs={total_pairs}")

            # 평가 (기본은 매 epoch, 필요하면 eval_every로 줄일 수 있음)
            score = float("nan")
            if self.eval_every > 0 and (epoch % self.eval_every == 0 or epoch == self.epoch - 1):
                net.eval()
                with torch.no_grad():
                    X_all = torch.from_numpy(training_data[:, 2:].astype(np.float32, copy=False))
                    pred = net.model(X_all).numpy().flatten()
                y_true = training_data[:, 0]
                qid_all = training_data[:, 1]
                metrics = evaluate_all(y_true, pred, qid_all)
                score = float(metrics["NDCG@10"])

                if self.plot:
                    plt.cla()
                    plt.plot(range(epoch + 1), loss_list, "r-", lw=3)
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss(avg over pairs)")
                    plt.pause(0.1)

                if epoch % 10 == 0:
                    print("Epoch:{}, loss(avg) : {}, Metrics: {}".format(epoch, avg_loss, metrics))

                if score > best_score:
                    best_score = score
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    torch.save(net.state_dict(), "./save_checkpoint/ranknet/ranknet.pt")
                net.train()

        if best_state is not None:
            net.load_state_dict(best_state)

        if self.plot:
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    print("Load training data...")
    training_data = np.load("./dataset/train.npy")
    print("Load done.\n\n")

    model1 = RankNet(46, 512, 256, 100, 0.01, True)
    model1.fit(training_data)

    print("Validate... (이 파일의 데모 블록은 생략)")

