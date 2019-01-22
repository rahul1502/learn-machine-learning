import numpy as np

def softmax(z):
    exp = np.exp(z)
    return exp / np.sum(exp)


A2 = np.array([[2.3, 6.6],[2.7, 5.4], [3.1, 7.7], [2.9, 5.9], [1.1, 8.3], [2.1, 5.0], [3.1, 4.9], [4.0, 7.3], [2.9, 6.1], [2.0, 6.8]])

pred = np.zeros(A2.shape)

for i in range(A2.shape[1]):
    pred[:,i] = softmax(A2.T[i])
