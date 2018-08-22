import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

number_of_examples = 100

X = df.iloc[0:number_of_examples, [0, 2]].values

X = np.concatenate((np.ones(number_of_examples)[:, np.newaxis], X), axis=1)

X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

plt.scatter(X[:50, 1], X[:50, 2], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 1], X[50:100, 2], color='blue', marker='x', label='versicolor')


plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

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

        for _ in range(self.epochs):
            z = X.dot(self.weights)
            difference = y - z
            error = 0.5 * np.sum(difference**2)
            self.errors.append(error)
            gradient = - X.transpose().dot(difference)
            self.weights = self.weights - self.learning_rate*gradient/number_of_examples

        return self


ppn = Ada(learning_rate=0.1, epochs=100)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Training')
plt.show()
