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

def train_listmle(X, y, qid, X_val, y_val, qid_val):
    model = ListNet(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=0.001)

    groups = group_by_qid(qid)
    best = -1

    for epoch in range(50):
        for g in groups.values():
            xg = torch.tensor(X[g], dtype=torch.float32)
            yg = torch.tensor(y[g], dtype=torch.float32)

            pred = model(xg)
            loss = listmle_loss(pred, yg)

            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred_val = model(torch.tensor(X_val, dtype=torch.float32)).numpy()

        metrics = evaluate_all(y_val, pred_val, qid_val)
        score = metrics["NDCG@10"]
        print(f"ListMLE Epoch {epoch} | Metrics: {metrics}")

        if score > best:
            best = score
            torch.save(model.state_dict(), "./save_checkpoint/listmle/listmle.pt")

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