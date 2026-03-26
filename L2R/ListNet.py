#Top-1만 고려
from data_utils import load_data, scale_data, group_by_qid
from metrics import evaluate_all
import torch
import torch.nn as nn
import torch.optim as optim

class ListNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def listnet_loss(pred, y):
    P_y = torch.softmax(y, dim=0)
    P_z = torch.softmax(pred, dim=0)
    return -torch.sum(P_y * torch.log(P_z + 1e-10))

def train_listnet(X, y, qid, X_val, y_val, qid_val):
    model = ListNet(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=0.001)

    groups = group_by_qid(qid)
    best = -1
    patience, wait = 5, 0

    for epoch in range(50):
        model.train()
        for g in groups.values():
            xg = torch.tensor(X[g], dtype=torch.float32)
            yg = torch.tensor(y[g], dtype=torch.float32)

            pred = model(xg)
            loss = listnet_loss(pred, yg)

            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(torch.tensor(X_val, dtype=torch.float32)).numpy()

        
        metrics = evaluate_all(y_val, pred_val, qid_val)
        score = metrics["NDCG@10"]
        print(f"ListNet Epoch {epoch} | Metrics: {metrics}")



        if score > best:
            best = score
            wait = 0
            torch.save(model.state_dict(), "./save_checkpoint/listnet/listnet.pt")
        else:
            wait += 1
            if wait >= patience:
                break

    return model

if __name__ == '__main__':
    X_train, y_train, qid_train = load_data('./dataset/train.npy')
    X_val, y_val, qid_val = load_data('./dataset/vali.npy')
    X_test, y_test, qid_test = load_data('./dataset/test.npy')
    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)
    listnet_model = train_listnet(X_train, y_train, qid_train, X_val, y_val, qid_val)
    with torch.no_grad():
        pred_ln = listnet_model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    print("ListNet:", evaluate_all(y_test, pred_ln, qid_test))
