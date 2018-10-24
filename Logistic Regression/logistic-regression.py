import numpy as np
import matplotlib.pyplot as plt
import csv

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

data = csv.reader(open('data/train.csv', newline=''), delimiter=' ', quotechar='|')

# User ID,Gender,Age,EstimatedSalary,Purchased
x_train = np.empty((0,4), float)
y_train = np.empty((0,1), float)
for row in data:
    if row[0].split(',')[1] == 'Male':
        gender = 1.0
    else:
        gender = 0.0

    x_train = np.append(x_train, np.array([[gender, float(row[0].split(',')[2]), float(row[0].split(',')[3]), 1.0]]), axis = 0)
    y_train = np.append(y_train, np.array([[float(row[0].split(',')[4])]]), axis = 0)

# training
print('Training with ' + str(len(x_train)) + ' tuples')

# y = w1*x1 + w2*x2 + w3*x3 + w4
# parameters = m, c
# start with the random values of m, class
n = len(y_train)
learning_rate = 0.000000001
epochs = 10

# w1, w2, w3, w4
theta = np.array([[0.0,0.0,0.0,0.0]])

for i in range(epochs):

    z = x_train @ theta.T
    y_pred = sigmoid(z)

    # cost function
    cost = - (1 / n) * np.sum((y_train * np.log(y_pred)) + ((1 - y_train) * np.log(1 - y_pred)))

    print(cost)
    # regression step
    theta[0] -= (1/n) * learning_rate * np.sum((y_pred - y_train).T @ x_train)
