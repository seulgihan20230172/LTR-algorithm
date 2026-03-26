import numpy as np
'''
arr = np.load('./dataset/test.npy')
print("test", arr, arr.shape, arr.dtype)

arr1 = np.load('./dataset/train.npy')
print("train", arr1, arr1.shape, arr1.dtype)
'''
arr2 = np.load('./dataset/vali.npy')
print("vali", arr2, arr2.shape, arr2.dtype)
#[label, qid, f1, f2, ..., f46]
# (9630, 48) float64
# (2874, 48) float64
# (2707, 48) float64