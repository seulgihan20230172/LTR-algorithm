import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def load_data(path):
    arr = np.load(path)
    y = arr[:, 0]
    qid = arr[:, 1]
    X = arr[:, 2:]
    return X, y, qid

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test

def group_by_qid(qid):
    groups = defaultdict(list)
    for i, q in enumerate(qid):
        groups[q].append(i)
    return groups
