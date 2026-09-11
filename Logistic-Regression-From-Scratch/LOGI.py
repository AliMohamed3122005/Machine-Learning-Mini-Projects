import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_loss(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        z = np.dot(w, x[i]) + b
        g = sigmoid(z)

        cost_sum += -y[i] * np.log(g) - (1 - y[i]) * np.log(1 - g)

    return cost_sum / m


def compute_gradient(x, y, w, b):
    m, n = x.shape

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        z = np.dot(w, x[i]) + b
        g = sigmoid(z)

        error = g - y[i]

        for j in range(n):
            dj_dw[j] += error * x[i][j]

        dj_db += error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def fit(x, y, alpha=0.01, iterations=1000):
    m, n = x.shape

    w = np.zeros(n)
    b = 0

    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % 100 == 0:
            loss = compute_loss(x, y, w, b)
            print(f"Iteration {i}: Loss = {loss}")

    return w, b


def predict(x, w, b):
    m = x.shape[0]
    y_pred = np.zeros(m)

    for i in range(m):
        z = np.dot(w, x[i]) + b
        g = sigmoid(z)

        if g >= 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    return y_pred