import numpy as np


class LinearRegression:

    def __init__(self, max_iter=1e4, lr=0.001, tol=0.001, l2_coef=1.):

        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.l2_coef = l2_coef
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):

        n, m = X_train.shape

        self.weights = np.zeros((m, 1))
        self.bias = np.mean(y_train)

        n_iter = 0
        gradient_norm = np.inf

        while n_iter < self.max_iter and gradient_norm > self.tol:

            dJdw, dJdb = self.grads(X_train, y_train)

            # L2 regularization update
            dJdw += self.l2_coef * self.weights

            gradient_norm = np.linalg.norm(np.hstack([dJdw.flatten(), [dJdb]]))

            self.weights = self.weights - self.lr * dJdw
            self.bias = self.bias - self.lr * dJdb

            n_iter += 1

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def grads(self, X, y):

        y_hat = self.predict(X)

        dJdw = (1 / X.shape[0]) * np.dot(X.T, y_hat - y)
        dJdb = np.mean(y_hat - y)

        return dJdw, dJdb


def read_input():
    n, m, k = map(int, input().split())

    x_train = np.array([input().split() for _ in range(n)]).astype(float)
    y_train = np.array([input().split() for _ in range(n)]).astype(float)
    x_test = np.array([input().split() for _ in range(k)]).astype(float)
    return x_train, y_train, x_test


def solution():

    x_train, y_train, x_test = read_input()

    model = LinearRegression(max_iter=5000, lr=1e-3, l2_coef=2.)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    result = ' '.join(map(lambda x: str(float(x)), predictions))
    print(result)


solution()
