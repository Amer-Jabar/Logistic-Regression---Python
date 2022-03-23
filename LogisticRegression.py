# This is logistic regression from scratch. It is created for the sake of
# the machine learning visualization web application

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston
from LogisticRegressionHelper import execute_lr, predict

x = np.array([
    1, 2, 3, 4, 1, 2, 4, 3, 4, 5, 1, 2, 3, 4, 2, 3, 2, 3, 1, 4, 5, 6, 3, 1, 4, 1, 1, 2, 3, 5,
    12, 14, 13, 10, 11, 12, 11, 10, 13, 14, 12, 10, 10, 12, 10, 13, 12, 14, 12, 11, 10, 10, 14, 12, 13, 10, 11, 10, 14, 14,
])
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
data_size = len(x)

w1 = 0
w0 = 0
eta = 0.01
epochs = 20

loss_hist = []
w1, w0, loss_hist = execute_lr(x, y, len(x), eta, w1, w0, loss_hist)

# plt.scatter(x, y,  = y)
plt.plot(range(len(loss_hist)), loss_hist)

