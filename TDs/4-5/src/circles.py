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

    params["wh"] = n.sample((nx, nh))
    params["wy"] = n.sample((nh, ny))
    params["bh"] = n.sample((nh,))
    params["by"] = n.sample((ny,))

    return params


def forward(params, X):
    outputs = {}

    # TODO remplir avec les paramètres X, htilde, h, ytilde, yhat
    # outputs["X"] = ...

    outputs["htilde"] = X.mm(params["wh"]).add(params["bh"])
    outputs["h"] = torch.tanh(outputs["htilde"])
    outputs["ytilde"]  = outputs["h"].mm(params["wy"]).add(params["by"])
    outputs["yhat"] = torch.nn.functional.softmax(outputs["ytilde"], dim=1)

    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    # TODO
    L = ((-1)*torch.log(Yhat)*Y).sum().item()
    Yprevs = torch.argmax(Yhat, dim=1)
    # print("Yprevs: \n", Yprevs)
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

    return grads

def sgd(params, grads, eta):
    # TODO mettre à jour le contenu de params

    return params



if __name__ == '__main__':

    # init
    data = CirclesData()
    data.plot_data()
    N = data.Xtrain.shape[0]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.03

    # Premiers tests, code à modifier
    params = init_params(nx, nh, xy)
    Yhat, outs = forward(params, data.Xtrain)
    L, _ = loss_accuracy(Yhat, Y)
    grads = backward(params, outputs, Y)
    params = sgd(params, grads, eta)

    # TODO apprentissage

    # attendre un appui sur une touche pour garder les figures
    input("done")
