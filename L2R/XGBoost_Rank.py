import xgboost as xgb
import pickle
from data_utils import load_data, scale_data, group_by_qid
from metrics import evaluate_all
import numpy as np

def train_xgb(X, y, qid, X_val, y_val, qid_val):
    _, group = np.unique(qid, return_counts=True)
    _, group_val = np.unique(qid_val, return_counts=True)

    model = xgb.XGBRanker(
        objective='rank:ndcg',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(
        X, y,
        group=group,
        eval_set=[(X_val, y_val)],
        eval_group=[group_val],
        verbose=True
    )
    
    with open("./save_checkpoint/xgboost/xgb.pkl", "wb") as f:
        pickle.dump(model, f)#pytorch사용안해서

    return model

if __name__ == '__main__':
    X_train, y_train, qid_train = load_data('./dataset/train.npy')
    X_val, y_val, qid_val = load_data('./dataset/vali.npy')
    X_test, y_test, qid_test = load_data('./dataset/test.npy')
    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)
    xgb_model = train_xgb(X_train, y_train, qid_train, X_val, y_val, qid_val)
    pred_xgb = xgb_model.predict(X_test)
    print("XGB:", evaluate_all(y_test, pred_xgb, qid_test))

