# x**2 + 10*np.sin(x)


import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**2 + 10*np.sin(x)


def grad(x):
    return 2*x + 10*np.cos(x)


def train(x, gamma, eta, iter):
    X = []
    vt = eta*grad(x)
    it = 0
    for i in range(iter):
        vt = gamma*vt + eta*grad(x- gamma*vt)
        x -= vt
        X.append(x)
        if abs(grad(x)) < 1e-6 :
            break
    it = i
    return x, it, X


x_axis = np.linspace(-6, 6, 50)
y_axis = f(x_axis)
plt.plot(x_axis, y_axis)

x, iter, X = train(-5, 0.9, 0.1, 1000)
print("x:", x)
print("iter:", iter)
X = np.array(X)
plt.scatter(X, f(X))
plt.show()

