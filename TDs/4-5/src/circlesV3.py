import torch
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData

def init_model(nx, nh, ny):
    model = torch.nn.Sequential(
        torch.nn.Linear(nx, nh),
        torch.nn.Tanh(),
        torch.nn.Linear(nh, ny)
        # torch.nn.Softmax()
    )

    loss = torch.nn.CrossEntropyLoss()
    return model, loss


def loss_accuracy(Ytilde, Y, loss):
    # TODO

    Yhat = torch.nn.Softmax()(Ytilde)
    Yprevs = torch.argmax(Yhat, dim=1) # nbatch x 1
    # print('Yprevs.shape: ', Yprevs.shape)
    # print("Y.sum():\n", Y.sum(dim=0))
    # print("Yprevs:\n", Yprevs)
    # print('Yprevs.sum(): ', Yprevs.sum())
    Yargmax = torch.argmax(Y, dim=1) # nbatch x 1
    acc = (Yprevs == Yargmax).sum().item() / Y.shape[0]
    # print("L: ", L)

    # L = ((-1)*torch.log(Yhat)*Y).sum()
    # print("acc: ", acc)
    L = loss(Ytilde, Yargmax)
    return L, acc


def sgd(model, eta):
    # TODO mettre à jour le contenu de params

    with torch.no_grad():
        for param in model.parameters():
            param -= param.grad * eta
        model.zero_grad()

    return model


if __name__ == '__main__':

    # init
    data = CirclesData()
    # data.plot_data()
    N = data.Xtrain.shape[0]
    Nepoch = 800 # 1500  # 3000
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.02
    print('Data set size: ', N)  # 200 here
    # params = init_params(nx, nh, ny)
    model, loss = init_model(nx, nh, ny)
    printInterval = 100
    trainLosses = []
    testLosses = []

    # TODO apprentissage
    for i in range(Nepoch):
        for j in range(int(N/Nbatch)):
            X = torch.from_numpy(data._Xtrain[j * Nbatch : (j+1) * Nbatch])
            Y = torch.from_numpy(data._Ytrain[j * Nbatch : (j+1) * Nbatch])

            # Yhat, outputs = forward(params, X)
            Ytilde = model(X)

            L, accuracy = loss_accuracy(Ytilde, Y, loss)

            # grads = backward(params, outputs, Y)
            L.backward()

            model = sgd(model, eta)

        if i % printInterval == 0:
            print('- Epoch: ', i)
            # Training set
            # Yhat, _ = forward(params, torch.from_numpy(data._Xtrain))
            Ytilde = model(torch.from_numpy(data._Xtrain))
            L, accuracy = loss_accuracy(Ytilde, torch.from_numpy(data._Ytrain), loss)
            trainLosses.append(L.item())
            print(' Training set:\n', '   loss: ', L.item(), '; accuracy: ', accuracy, sep='')

            # Testing set
            # Yhat, _ = forward(params, torch.from_numpy(data._Xtest))
            Ytilde = model(torch.from_numpy(data._Xtest))
            L, accuracy = loss_accuracy(Ytilde, torch.from_numpy(data._Ytest), loss)
            print(' Test set:\n', '   loss: ', L.item(), '; accuracy: ', accuracy, sep='')
            testLosses.append(L.item())

            # Ygrid, _ = forward(params, data.Xgrid)
            Ygridtilde = model(data.Xgrid)
            Ygridhat = torch.nn.Softmax()(Ygridtilde) # Softmax is not inside the "model" because "torch.nn.CrossEntropyLoss()" already has it, so you have to do this step to transform "Ygridtilde" in "Ygridhat" 
            data.plot_data_with_grid(Ygridhat.detach())

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
