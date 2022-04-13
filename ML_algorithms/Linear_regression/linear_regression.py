import numpy as np

class LinearRegression:
    def __init__(self, n_iterations = 100, n_learning_rate = 0.0001):
        self.n_iterations = n_iterations
        self.n_learning_rate = n_learning_rate
        self.bias = 0
        self.slope = 0

    def fit(self, X_train, y_train):
        n = X_train.shape[1]
        self.slope = np.zeros(n)

        for _ in range(self.n_iterations):
            self.slope -= self.n_learning_rate * (-2/n) * self.calc_slope(X_train.T, y_train)
            self.bias -= self.n_learning_rate * (-2/n) * self.calc_bias(X_train.T, y_train)

    def predict(self, X):
        return self.regression(X)

    def calc_bias(self, X, y):
        return np.sum(y - self.regression(X))

    def calc_slope(self, X, y):
        # return np.sum(X * (y - self.regression(X)))
        return np.dot(X, (y - self.regression(X)))

    def regression(self, X):
        # return self.slope * X + self.bias
        return np.dot(self.slope, (X + self.bias))


    