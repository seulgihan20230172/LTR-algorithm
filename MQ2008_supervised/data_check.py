import numpy as np

X = []
y = []
qid = []

with open('test.txt') as f:
    for line in f:
        parts = line.strip().split()
        
        y.append(float(parts[0]))
        qid.append(int(parts[1].split(':')[1]))
        
        features = []
        for item in parts[2:]:
            if item.startswith('#'):
                break
            features.append(float(item.split(':')[1]))
        
        X.append(features)

X = np.array(X)
y = np.array(y)
qid = np.array(qid)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("qid shape:", qid.shape)
print("dtype:", X.dtype)
print("num queries:", len(np.unique(qid)))
print("labels:", np.unique(y))
"""
test
X shape: (9630, 46)
y shape: (9630,)
qid shape: (9630,)
dtype: float64
num queries: 471
labels: [0. 1. 2.]

train
X shape: (2874, 46)
y shape: (2874,)
qid shape: (2874,)
dtype: float64
num queries: 156
labels: [0. 1. 2.]
"""