import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection

data = pd.read_csv("cars.csv")
np.set_printoptions(suppress=True)

data = data[['mpg', 'displacement', 'weight', 'acceleration', "origin"]]
print("Dimensions:", data.shape)
print(data.describe())
print(data.groupby('origin').size())  # 1-USA, 2-Europe, 3-Japan
pd.plotting.scatter_matrix(data)
plt.show()
data.boxplot(column=['mpg',  'acceleration'])
plt.show()
data.hist(column=['mpg', 'acceleration', 'weight', 'displacement'])
plt.show()

array = data.to_numpy()
x = array[:, 1:5]
y = array[:, 0]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

model = sklearn.linear_model.LinearRegression().fit(x_train, y_train)

print("Intercept + coefficients:", model.intercept_, model.coef_)

predict_y = model.predict(x_test)

for i in range(len(predict_y)):
    print(np.round(predict_y[i], 2), y_test[i], x_test[i])

accuracy = model.score(x_test, y_test)  # R^2 = 1 -rMSE (relative mean squared error)
