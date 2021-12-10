import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from collections import Counter


dataset = {'k':[[1,5],[2,3],[4,2]], 'r':[[6,4],[7,8],[9,7]]}
new_features = [5,5]

def knn(data, predict, k=3):

    distances = []
    for group in data:
        for features in data[group]:
            distances.append([np.linalg.norm(np.array(features) - np.array(predict)), group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

knn_result = knn(dataset, new_features, 3)
print(knn_result)

[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=50, color = knn_result)
plt.show()
