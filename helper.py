import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_iris_data(number_of_examples=100, output_values_ranges=[0, 1]):
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:number_of_examples, 4].values
    y = np.where(y == 'Iris-setosa', output_values_ranges[0], output_values_ranges[1])

    X = df.iloc[0:number_of_examples, [0, 2]].values

    X = np.concatenate((np.ones(number_of_examples)[:, np.newaxis], X), axis=1)

    X = normalize(X)

    return X, y


def normalize(X):
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    return X


def plot_iris_data(X):
    plt.scatter(X[:50, 1], X[:50, 2], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 1], X[50:100, 2], color='blue', marker='x', label='versicolor')

    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


def plot_training(training_errors):
    plt.plot(range(1, len(training_errors) + 1), training_errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Training')
    plt.show()
