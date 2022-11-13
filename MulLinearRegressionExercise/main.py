#some graphs needs to be printed for cmp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

dataset = fetch_california_housing()

def Visualize_DataSet():

    plt.figure(figsize=(12, 15))

    for i in range(8):
        plt.subplot(4, 2, i + 1)
        plt.scatter(X[:, i], t)
        plt.title(dataset.feature_names[i])
    plt.show()


X = dataset.data
t = dataset.target

print(X.shape)
print(t.shape)

# print(dataset.DESCR) #  describes the dataset

#Visualize_DataSet()

#linear regression
reg = LinearRegression()
reg.fit(X,t)

#prediction
y = reg.predict(X)

#mse
mse = np.mean((t-y)**2) # mean of square of t-y
print("MSE: ",mse)

plt.figure(figsize=(8, 6))
plt.scatter(t, y)
plt.plot([t.min(), t.max()], [t.min(), t.max()], c="red")
plt.xlabel("t")
plt.ylabel("y")

plt.figure(figsize=(12, 15))
for i in range(8):
    plt.subplot(4, 2 , i+1)
    plt.scatter(X[:, i], t)
    plt.scatter(X[:, i], y)
    plt.title(dataset.feature_names[i])
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(np.abs(reg.coef_))
plt.xlabel("Feature")
plt.ylabel("Weight")
plt.show()

X_normalized = (X-X.mean(axis=0)/X.std(axis=0))

print("Old mean:", X.mean(axis=0))
print("New mean:", X_normalized.mean(axis=0))
print("Old std:", X.std(axis=0))
print("New std:", X_normalized.std(axis=0))

#mul-reg after normalization
reg.fit(X_normalized,t)

#prediction
y = reg.predict(X_normalized)

mse = np.mean((t-y)**2) # mean of square of t-y
print("MSE: ",mse)

#print again to cmp the results

#regularized regression:
from sklearn.linear_model import Ridge, Lasso

#ridge : MLR with L2 regularization
reg = Ridge(alpha=10.0)

reg.fit(X_normalized,t)
y = reg.predict(X_normalized)
mse = np.mean((t-y)**2)
print(mse)
print(reg.coef_)
#PRINT GRAPHS

#lasso : MLR with L1 regularization
reg = Lasso(alpha=0.1)

reg.fit(X_normalized, t)
y = reg.predict(X_normalized)
mse = np.mean((t - y)**2)
print(mse)
#PRINT GRAPHS
