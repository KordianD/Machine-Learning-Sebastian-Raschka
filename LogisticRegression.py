import numpy as np
import helper

X, y = helper.get_iris_data()

helper.plot_iris_data(X)


class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=50, random_state=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.weights = np.array(self.weights).reshape((3, 1))
        self.errors = []
        number_of_examples = X.shape[0]

        for _ in range(self.epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)

            difference = y.reshape((y.shape[0], 1)) - output

            error = (-y.dot(np.log(output)) -
                     ((1 - y).dot(np.log(1 - output))))

            self.errors.append(error)
            gradient = - X.transpose().dot(difference)

            self.weights = self.weights + self.learning_rate * gradient / number_of_examples

        return self

    def net_input(self, X):
        return np.dot(X, self.weights)

    def activation(self, net_input):
        return 1. / (1. + np.exp(net_input))


ppn = LogisticRegression(learning_rate=0.1, epochs=100)

ppn.fit(X, y)

helper.plot_training(ppn.errors)
