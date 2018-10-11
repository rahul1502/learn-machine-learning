import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def leakey_ReLU(x):
    return np.maximum(0.01 * x, x)


x = np.array([[0.3, 0.6, 0.56, -0.23 ],
              [1, 0.34, 0.31, 0 ],
              [-0.23, -0.45, 6, 0.999 ]])

print(x)

print('Sigmoid:')
print(sigmoid(x))


print('Tanh: ')
print(tanh(x))

print('ReLU: ')
print(ReLU(x))

print('Leakey ReLU: ')
print(leakey_ReLU(x))
