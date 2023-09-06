# x**2 + 10*np.sin(x)


import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2 + 10*np.sin(x)


def grad(x):
    # eps = 1e-6
    # return (f(x+eps)-f(x+eps))/(2*eps)
    return 2*x+10*np.cos(x)


def train(x, eta, gamma, iter):
    X = []
    vt = eta*grad(x)
    it = 0
    for i in range(iter):
        x -= vt
        vt = gamma*vt + eta*grad(x)
        X.append(x)
        it = i
        if abs(grad(x)) < 1e-3:
            break
    return X, it

x_axis = np.linspace(-6, 6, 50)
y_axis = f(x_axis)
plt.plot(x_axis, y_axis)

X, iter = train(5, 0.1, 0.9, 1000)
print("iter:", iter)
print("x:", X[-1])
X = np.array(X)
plt.scatter(X, f(X))
plt.show()
