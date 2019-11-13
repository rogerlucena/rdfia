import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData

def init_params(nx, nh, ny):
    params = {}

    # TODO remplir avec les paramètres Wh, Wy, bh, by
    # params["Wh"] = ...

    n = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(0.3))

    params["Wh"] = n.sample((nx, nh))
    params["Wy"] = n.sample((nh, ny))
    params["bh"] = n.sample((nh,))
    params["by"] = n.sample((ny,))

    return params


def forward(params, X):
    outputs = {}

    # TODO remplir avec les paramètres X, htilde, h, ytilde, yhat
    # outputs["X"] = ...

    outputs["X"] = X
    outputs["htilde"] = X.mm(params["Wh"]).add(params["bh"]) # nbatch x nh
    outputs["h"] = torch.tanh(outputs["htilde"]) # nbatch x nh
    outputs["ytilde"]  = outputs["h"].mm(params["Wy"]).add(params["by"]) # nbatch x ny
    outputs["yhat"] = torch.nn.functional.softmax(outputs["ytilde"], dim=1) # nbatch x ny

    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    # TODO
    L = ((-1)*torch.log(Yhat)*Y).sum().item()
    Yprevs = torch.argmax(Yhat, dim=1) # nbatch x 1
    # print('Yprevs.shape: ', Yprevs.shape)
    # print("Y.sum():\n", Y.sum(dim=0))
    # print("Yprevs:\n", Yprevs)
    # print('Yprevs.sum(): ', Yprevs.sum())
    Yargmax = torch.argmax(Y, dim=1)
    acc = (Yprevs == Yargmax).sum().item() / Y.shape[0]
    # print("L: ", L)
    # print("acc: ", acc)
    return L, acc

def backward(params, outputs, Y):
    grads = {}

    # TODO remplir avec les paramètres Wy, Wh, by, bh
    # grads["Wy"] = ...

    # Remember: the dimensions of Wy and Wh in "formulas.pdf" transposed compared to the same dimensions here
    grad_ytilde = outputs["yhat"] - Y # nbatch x ny
    grads["Wy"] = ((grad_ytilde.T).mm(outputs["h"])).T # nh x ny
    grads["by"] = (grad_ytilde.sum(dim=0)).T # ny x 1 
    grad_htilde = (grad_ytilde.mm(params["Wy"].T)).mul(1-outputs["h"].mul(outputs["h"])) # nbatch x nh
    grads["Wh"] = (grad_htilde.T.mm(outputs["X"])).T # nx x nh
    grads["bh"] = (grad_htilde.sum(dim=0)).T # nh x 1

    return grads

def sgd(params, grads, eta):
    # TODO mettre à jour le contenu de params

    params["Wy"] -= grads["Wy"] * eta
    params["by"] -= grads["by"] * eta
    params["Wh"] -= grads["Wh"] * eta
    params["bh"] -= grads["bh"] * eta

    return params



if __name__ == '__main__':

    # init
    data = CirclesData()
    # data.plot_data()
    N = data.Xtrain.shape[0]
    Nepoch = 1500 # 3000
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.2
    print('Data set size: ', N) # 200 here
    params = init_params(nx, nh, ny)
    printInterval = 100
    trainLosses = []
    testLosses = []

    # TODO apprentissage
    for i in range(Nepoch):
        for j in range(int(N/Nbatch)):
            X = torch.from_numpy(data._Xtrain[j * Nbatch : (j+1) * Nbatch])
            Y = torch.from_numpy(data._Ytrain[j * Nbatch : (j+1) * Nbatch])
            Yhat, outputs = forward(params, X)
            L, accuracy = loss_accuracy(Yhat, Y)
            grads = backward(params, outputs, Y)
            params = sgd(params, grads, eta)

        if i % printInterval == 0:
            print('- Epoch: ', i)
            # Training set
            Yhat, _ = forward(params, torch.from_numpy(data._Xtrain))
            L, accuracy = loss_accuracy(Yhat, torch.from_numpy(data._Ytrain))
            trainLosses.append(L)
            print(' Training set:\n', '   loss: ', L, '; accuracy: ', accuracy, sep='')

            # Testing set
            Yhat, _ = forward(params, torch.from_numpy(data._Xtest))
            L, accuracy = loss_accuracy(Yhat, torch.from_numpy(data._Ytest))
            print(' Test set:\n', '   loss: ', L, '; accuracy: ', accuracy, sep='')
            testLosses.append(L)

            Ygrid, _ = forward(params, data.Xgrid)
            data.plot_data_with_grid(Ygrid)

    x = np.arange(0, Nepoch, printInterval)
    trainLosses = np.array(trainLosses)
    testLosses = np.array(testLosses)
    plt.close()
    plt.clf()
    plt.ylabel('Losses')
    plt.xlabel('Epoch')
    plt.title('Train loss and Test loss with time')
    plt.grid(True)
    plt.plot(x, trainLosses, 'b--', x, testLosses, 'r--')
    plt.legend(['Train loss', 'Test loss'], loc=1)
    plt.show()

    # attendre un appui sur une touche pour garder les figures
    # input("done")
