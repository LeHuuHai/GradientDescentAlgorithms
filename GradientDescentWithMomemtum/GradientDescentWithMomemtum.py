import matplotlib.pyplot as plt
import numpy as np

# data
X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1)
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one,X), axis = 1)

def predict(Xbar, w):
    return Xbar.dot(w)

def grad(w):
    n = Xbar.shape[0]
    return 1/n * Xbar.T.dot(Xbar.dot(w) - y)

def loss_function(weight):
    n = Xbar.shape[0]
    return 0.5/n * np.linalg.norm(Xbar.dot(weight)-y) **2

def train(w, eta, gamma, iter):
    it = 0
    vt = grad(w)
    loss = []
    for i in range(iter):
        loss.append(loss_function(w))
        it = i
        w -= vt
        vt = gamma*vt + eta*grad(w)
        if np.linalg.norm(grad(w))/len(w) < 1e-2:
            break
    return w, it, loss

def main():
    w = np.array([2., 2.])
    w = np.reshape(w, (2, 1))
    w, iter, loss = train(w, 0.01, 0.9, 1000)
    print("iter:", iter)
    print("w:", w)
    print(loss)

    loss_x_axis = [i for i in range(len(loss))]
    plt.plot(loss_x_axis, loss)
    plt.show()

main()
