import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

number_of_examples = 100

X = df.iloc[0:number_of_examples, [0, 2]].values

X = np.concatenate((np.ones(number_of_examples)[:, np.newaxis], X), axis=1)

X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

plt.scatter(X[:50, 1], X[:50, 2], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 1], X[50:100, 2], color='blue', marker='x', label='versicolor')

plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

print("SIZES " + str(X.shape))


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
plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Training')
plt.show()
