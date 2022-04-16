import numpy as np

class LogisitcRegression:
    def __init__(self, n_iterations = 100, n_learning_rate = 0.0001):
        self.n_iterations = n_iterations
        self.n_learning_rate = n_learning_rate
        self.bias = 0
        self.slope = 0

    def fit(self, X_train, y_train):
        n = X_train.shape[1]
        m = len(X_train)
        self.slope = np.zeros(n)

        for _ in range(self.n_iterations):
            self.slope -= self.n_learning_rate *self.calc_slope(X_train, y_train, m)
            self.bias -= self.n_learning_rate * self.calc_bias(X_train, y_train, m)

    def calc_slope(self, X, y, m):
        return np.dot(X.T, (self.sigmoid(X) - y)) / m

    def calc_bias(self, X, y, m):
        return np.sum(self.sigmoid(X) - y) / m
    
    def regression(self, X):
        return np.dot(X, self.slope) + self.bias

    def sigmoid(self, X):
        return 1/(1 + np.exp(- self.regression(X)))

    def predict(self, X):
        Z = self.regression(X)
        Y = np.where(Z > 0.5, 1, 0)

        return Y

    def score(self, y_predicted, y_test):
        return np.sum(y_predicted == y_test) / len(y_test)