# git pull
import numpy as np
import matplotlib.pyplot as plt
# f(x) = x^2 + 5*np.sin(x)

def f(x):
    return x**2 + 5*np.sin(x)
def grad(x):
    eps = 1e-6
    return (f(x+eps)-f(x-eps))/(2*eps)

def updatePoint(x, eta, iter):
    X = []
    for i in range(iter):
        x -= eta*grad(x)
        X.append(x)
        if abs(grad(x)) < 1e-6:
            break
    return X, i

def main():
    x_axis = np.linspace(-6, 6, 50)
    y_axis = f(x_axis)
    plt.plot(x_axis, y_axis)

    X, iter = updatePoint(6, 0.05, 100)

    X = np.array(X)
    Y = f(X)
    plt.scatter(X, Y)

    print("iter: ", iter)
    plt.show()


main()
