from typing import Optional

from data_utils import load_data, scale_data, group_by_qid
from metrics import evaluate_all
from ListNet import ListNet
import torch.optim as optim
import torch


def listmle_loss(pred, y):
    idx = torch.argsort(y, descending=True)
    pred = pred[idx]

    loss = 0
    for i in range(len(pred)):
        loss += torch.logsumexp(pred[i:], dim=0) - pred[i]
    return loss

def train_listmle(
    X,
    y,
    qid,
    X_val,
    y_val,
    qid_val,
    *,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 10**9,
    verbose: bool = True,
    #qids_per_chunk: Optional[int] = None,
    qids_per_chunk: Optional[int] = None,
):
    """qid는 ``sort_ltr_rows_by_qid`` 로 정렬된 것과 동일해야 XGBoost rank와 동일한 쿼리 순서다.

    ``qids_per_chunk``: 0 또는 None이면 전체 qid를 한 청크로 본다.
    """
    model = ListNet(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)

    groups = group_by_qid(qid)
    uq = np.unique(np.asarray(qid))
    chunk = len(uq) if not qids_per_chunk or int(qids_per_chunk) <= 0 else int(qids_per_chunk)
    best_score = -1.0
    wait = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for c0 in range(0, len(uq), chunk):
            c1 = min(c0 + chunk, len(uq))
            for qi in range(c0, c1):
                q = uq[qi]
                g = groups[q]
                xg = torch.tensor(X[g], dtype=torch.float32)
                yg = torch.tensor(y[g], dtype=torch.float32)

                pred = model(xg)
                loss = listmle_loss(pred, yg)

                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(torch.tensor(X_val, dtype=torch.float32)).numpy()

        metrics = evaluate_all(y_val, pred_val, qid_val)
        score = metrics["NDCG@10"]
        if verbose:
            print(f"ListMLE Epoch {epoch} | Metrics: {metrics}")

        if score > best_score:
            best_score = score
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), "./save_checkpoint/listmle/listmle.pt")
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

if __name__ == '__main__':
    X_train, y_train, qid_train = load_data('./dataset/train.npy')
    X_val, y_val, qid_val = load_data('./dataset/vali.npy')
    X_test, y_test, qid_test = load_data('./dataset/test.npy')
    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)
    listmle_model = train_listmle(X_train, y_train, qid_train, X_val, y_val, qid_val)
    
    with torch.no_grad():
        pred_lm = listmle_model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    print("ListMLE:", evaluate_all(y_test, pred_lm, qid_test))