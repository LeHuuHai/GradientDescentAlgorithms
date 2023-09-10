import numpy as np
import matplotlib.pyplot as plt

# data
X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1)
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one,X), axis = 1)

def predict(w):
    return Xbar.dot(w)
def grad(w):
    n = Xbar.shape[0]
    return 1/n * Xbar.T.dot(Xbar.dot(w) - y)

def loss_function(w):
    n = Xbar.shape[0]
    return 0.5/n * np.linalg.norm(Xbar.dot(w) - y)**2

def train(w, eta, iter):
    loss = []
    it = 0
    for i in range(iter):
        it = i
        loss.append(loss_function(w))
        w -= eta*grad(w)
        if np.linalg.norm(grad(w))/len(w) < 1e-2 :
            break
    return w, loss, it

def main():
    w = np.array([2., 2.])
    w = np.reshape(w, (2,1))
    w, loss, iter = train(w, 0.01, 1000)
    print("weight:", w)
    print("iter", iter)
    print("loss:", loss)
    loss_x_axis = [i for i in range(len(loss))]
    plt.plot(loss_x_axis, loss)
    plt.show()

main()
