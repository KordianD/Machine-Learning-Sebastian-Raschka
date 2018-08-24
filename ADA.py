import numpy as np
import helper

X, y = helper.get_iris_data(output_values_ranges=[-1, 1])

helper.plot_iris_data(X)


class Ada:

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
            z = X.dot(self.weights)
            difference = y - z
            error = 0.5 * np.sum(difference ** 2)
            self.errors.append(error)
            gradient = - X.transpose().dot(difference)
            self.weights = self.weights - self.learning_rate * gradient / number_of_examples

        return self


ppn = Ada(learning_rate=0.1, epochs=100)
ppn.fit(X, y)

helper.plot_training(ppn.errors)
