#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <path_to_npy_file>"
    exit 1
fi

# Extract the provided argument (path to npy file)
npy_file="$1"

# Check if the provided file exists
if [ ! -f "$npy_file" ]; then
    echo "Error: File not found: $npy_file"
    exit 1
fi

# Perform the necessary computations using Python
python_script=$(cat <<END

# YOUR CODE HERE
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import time
import pandas as pd

class CustomClassifierV:
    def __init__(self, encoder='default', k=3, distmetric='manhatten',multi=1):
        self.encoder = encoder
        self.k = k
        self.distmetric = distmetric
        self.multi=multi

    def calculate_distances(self, arr1, arr2):
        if self.distmetric == 'manhatten':
            distances = np.sum(np.abs(arr1[:, np.newaxis, :] - arr2), axis=2)
        elif self.distmetric == 'euclidean':
            temparr = arr1[:, np.newaxis, :] - arr2
            distances = np.sqrt(np.sum(temparr * temparr, axis=2))
        elif self.distmetric == 'cosine':
            dotarr = np.matmul(arr1, arr2.T)
            modarr1 = np.linalg.norm(arr1, axis=1, keepdims=True)
            modarr2 = np.linalg.norm(arr2, axis=1, keepdims=True)
            distances = 1 - (dotarr / (modarr1 * modarr2.T))
        return distances

    def fit_predict_evaluate(self):
        data = np.load("$npy_file", allow_pickle=True)
        n1 = len(data)
        # print(n1)
        n=int(n1*self.multi)
        # print(n)
        li = int(0.8*n)
        # print(li)
        r = li + 1

        arr1 = np.array([data[i][self.encoder][0] for i in range(r, n - 1)])
        arr2 = np.array([data[j][self.encoder][0] for j in range(0, li)])

        distances = self.calculate_distances(arr1, arr2)

        sorted_indices = np.argsort(distances, axis=1)
        # print(n1)
        # print(n)
        # print(li)
        predlist = data[0:li, 3][sorted_indices[:, :self.k]]

        pred = []

        for row in predlist:
            element_counts = Counter(row)
            most_common_element, most_common_count = element_counts.most_common(1)[0]
            pred.append(most_common_element)

        given = data[r:n - 1, 3]

        f1 = f1_score(given, pred, average='weighted', zero_division=1)
        accuracy = accuracy_score(given, pred)
        precision = precision_score(given, pred, average='weighted', zero_division=1)
        recall = recall_score(given, pred, average='weighted', zero_division=1)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'params': (self.encoder, self.k, self.distmetric)
        }


classifier = CustomClassifierV(encoder=2, k=10, distmetric='euclidean')
result = classifier.fit_predict_evaluate()
print(result)



END
)

# Run the Python script
python -c "$python_script"