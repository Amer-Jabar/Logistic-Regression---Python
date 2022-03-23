# This is logistic regression from scratch. It is created for the sake of
# the machine learning visualization web application

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from LogisticRegressionHelper import train

dataset = load_iris()
data = dataset['data']
target = dataset['target'][:100]
x = np.array(data[:100, 2])
y = np.array(target)
data_size = len(x)

w1 = 0.5
w0 = 0.5
eta = 0.01
epochs = 1000

w1, w0, cost_hist = train(x, y, data_size, eta, w1, w0, epochs)

# plt.scatter(x, y, c = y)
plt.plot(range(epochs), cost_hist)


