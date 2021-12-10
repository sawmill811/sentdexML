import numpy as np
from collections import Counter
import pandas as pd
import random
import math

def knn(data, predict, k=3):
    if len(data)>=k:
        print("k is too small")
    distances = []
    for group in data:
        for features in data[group]:
            distances.append([np.linalg.norm(np.array(features) - np.array(predict)), group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True)

# print((np.array(df.iloc[0])).tolist())

dataset = {}
for i in range(len(df.index)):
    row = (np.array(df.iloc[i])).tolist()
    if row[-1] in dataset:
        dataset[row[-1]].append(row[:-1])
    else:
        dataset[row[-1]] = [row[:-1]]

for group in dataset:
    for item in dataset[group]:
        item[5] = int(item[5])

random.shuffle(dataset[2])
random.shuffle(dataset[4])

test_size_2 = int(math.ceil(0.2*len(dataset[2])))
test_size_4 = int(math.ceil(0.2*len(dataset[4])))

sample_2 = dataset[2][-test_size_2:]
sample_4 = dataset[4][-test_size_4:]

dataset[2] = dataset[2][:-test_size_2]
dataset[4] = dataset[4][:-test_size_4]

# print(dataset)

# print(len(sample_2))
# print(len(sample_4))

correct = 0

for i in sample_2:
    knn_result = knn(dataset, i, 5)
    if knn_result==2:
        correct+=1

for i in sample_4:
    knn_result = knn(dataset, i, 5)
    if knn_result==4:
        correct+=1

accuracy = correct/(test_size_2 + test_size_4)*100.0

# print(test_size_4 - len(sample_4))
# print(test_size_2 - len(sample_2))

print(accuracy)




