import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# least mean squares:
# 1. compute slope = m
#       n = number of elements
#       m = (n*total(x*y)-total(x)*total(y))/(n*total(x^2)-total(x)^2)
# 2. compute y intersection
# 3. substitute the values in final eq

# residual: difference between predicted Y and actual y

# Q: Implement the LMS algorithm and apply it to the generated data.
# You will use a learning rate eta = 0.1 at first, but you will vary this value later.
# Print the value of the weight and bias at the end.
def LMS(ElementNo, x, T):

    w = 0
    b = 0
    # y = w*x + b = eq of line
    eta = 0.1

    for i in range(ElementNo):
        dw = 0
        db = 0.0
        for j in range(ElementNo):
            y = w*x[j] + b
            dw = dw + (T[j]-y)*x[j]
            db = db + (T[j]-y)

        w = w + eta * dw/N
        b = b + eta * db/N

    print(w,b)
    return w,b

def LMS_with_MSE(ElementNo, x, T):
    w = 0
    b = 0
    # y = w*x + b = eq of line
    eta = 0.1
    listMSE = []

    for i in range(ElementNo):
        mse = 0.0
        dw = 0
        db = 0.0
        for j in range(ElementNo):
            y = w * x[j] + b
            dw = dw + (T[j] - y) * x[j]
            db = db + (T[j] - y)
            mse += (t[j]-y)*(t[j]-y)

        listMSE.append(mse/N)
        w = w + eta * dw / N
        b = b + eta * db / N

    print(w, b)
    return w, b,listMSE

def LMS_logETA(ElementNo, x, T):
    w = 0
    b = 0
    # y = w*x + b = eq of line
    etas = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0]
    listMSE = []

    for eta in etas:
        mse = 0.0
        for i in range(ElementNo):
            dw = 0
            db = 0.0
            for j in range(ElementNo):
                y = w * x[j] + b
                dw = dw + (T[j] - y) * x[j]
                db = db + (T[j] - y)
                mse += (t[j]-y)*(t[j]-y)

            mse /= N
            w = w + eta * dw / N
            b = b + eta * db / N
        listMSE.append(mse)
    print(w, b)
    plt.figure(figsize=(10, 5))  # create figure object
    plt.plot(numpy.log10(etas),listMSE)
    plt.xlabel("eta")
    plt.ylabel("mse")

    return w, b,listMSE

def LMS_delta(ElementNo, x, T):
    w = 0
    b = 0.0
    eta = 0.1

    for epochs in range(ElementNo):
        for i in range(ElementNo):
            a = w*x[i]+b
            w += eta*(T[i]-a)*x[i]
            b += eta*(T[i]-a)

    return w,b


N = 100
X, t = make_regression(n_samples=N, n_features=1, noise=15.0)

print(X.shape)
print(t.shape)

W,B = LMS(N,X,t)

plt.figure(figsize=(10, 5))  # create figure object
xAxis = [X.min(),X.max()]
plt.plot(xAxis,W*xAxis + B)
plt.scatter(X, t)  # create diagram
plt.xlabel("x")
plt.ylabel("t")
# plt.show()  # show diagram

# Q:Make a scatter plot where tt is the x-axis and y = w\, x + by=wx+b is the y-axis.
# How should the points be arranged in the ideal case? Also plot what this ideal relationship should be.
plt.figure(figsize=(10, 5))  # create figure object
Y = W*X + B
xAxis = [t.min(),t.max()]
plt.plot(xAxis,xAxis)
plt.scatter(t, Y)  # create diagram
plt.xlabel("t")
plt.ylabel("Y")

#W,B,LIST_MSE = LMS_with_MSE(N,X,t)
LMS_logETA(N,X,t)

# same process with Scikit-learn lib
reg = LinearRegression()
reg.fit(X,t)
# Prediction
y = reg.predict(X)
# mse
mse = numpy.mean((t - y)**2)

W,B = LMS_delta(N,X,t)

plt.figure(figsize=(10, 5))  # create figure object
xAxis = [X.min(),X.max()]
plt.plot(xAxis,W*xAxis + B)
plt.scatter(X, t)  # create diagram
plt.xlabel("x")
plt.ylabel("t")
plt.show()  # show diagram


