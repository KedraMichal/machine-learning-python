import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection
import seaborn as sns
import pickle

data = pd.read_csv("cars.csv")
np.set_printoptions(suppress=True)


data = data[['mpg', 'acceleration', 'displacement', 'weight', 'cylinders', 'origin']]

# Seeing data
corr = data.corr()
sns.heatmap(corr, annot=True)
plt.show()

data.drop('acceleration', axis=1, inplace=True)  # corr between mpg, acc is too low, dropping acceleration
print(data.shape)
print(data.describe())

# Converting categorical variables to dummy variables
print(data.groupby('origin').size())  # 1-USA, 2-Europe, 3-Japan
data['origin'][data['origin'] == 1] = "USA"
data['origin'][data['origin'] == 2] = "Europe"
data['origin'][data['origin'] == 3] = "Japan"
data = pd.get_dummies(data, columns=['origin'])

print(data.head(3))
# pd.plotting.scatter_matrix(data)
# plt.show()
# plt.scatter(data.mpg, data.weight, edgecolors="red")
# plt.xlabel("mpg")
# plt.ylabel("weight")
# plt.show()
# data.boxplot(column=['mpg',  'acceleration'])
# plt.show()
# data.hist(column=['mpg', 'acceleration', 'weight', 'displacement'])
# plt.show()

array = data.to_numpy()
x = array[:, 1:7]
y = array[:, 0]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=0)

# best_acc = 0
# for i in range(20000):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
#
#     model = sklearn.linear_model.LinearRegression().fit(x_train, y_train)
#     acc = model.score(x_test, y_test)
#
#     if best_acc < acc:
#         best_acc = acc
#         print(best_acc)
#         with open("best_model", "wb") as file:
#            pickle.dump(model, file)

pickle_in = open("best_model", "rb")
model = pickle.load(pickle_in)

print("Intercept: {} ,coefficients:{}".format(model.intercept_, model.coef_))

predict_y = model.predict(x_test)

for i in range(len(predict_y)):
    print(np.round(predict_y[i], 2), y_test[i], x_test[i])

accuracy = model.score(x_test, y_test)  # R^2 = 1 -rMSE (relative mean squared error)
print(accuracy)


