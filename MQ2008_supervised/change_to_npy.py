import numpy as np

def letor_to_npy(txt_path, save_path):
    data = []

    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()

            label = float(parts[0])
            qid = float(parts[1].split(':')[1])

            features = []
            for item in parts[2:]:
                if item.startswith('#'):
                    break
                features.append(float(item.split(':')[1]))

            row = [label, qid] + features
            data.append(row)

    arr = np.array(data, dtype=np.float64)
    np.save(save_path, arr)

# 실행
#letor_to_npy("train.txt", "train.npy")
letor_to_npy("vali.txt", "vali.npy")
#letor_to_npy("test.txt", "test.npy")
