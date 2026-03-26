from data_utils import load_data, scale_data, group_by_qid
from metrics import evaluate_all

def bm25_eval(X, y, qid, idx=0):
    pred = X[:, idx]#BM25 점수를 그대로 ranking score로 사용#입력 feature 중 하나
    return evaluate_all(y, pred, qid)

if __name__ == '__main__':
    X_train, y_train, qid_train = load_data('./dataset/train.npy')
    X_val, y_val, qid_val = load_data('./dataset/vali.npy')
    X_test, y_test, qid_test = load_data('./dataset/test.npy')
    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)
    print("BM25:", bm25_eval(X_test, y_test, qid_test, idx=0))
