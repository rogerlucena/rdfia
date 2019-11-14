import torch
import numpy as np
import random
import matplotlib.pyplot as plt
# from mlxtend.data import mnist_data
from tme5 import MNISTData

def init_model(nx, nh, ny, eta):
    model = torch.nn.Sequential(
        torch.nn.Linear(nx, nh),
        torch.nn.Tanh(),
        torch.nn.Linear(nh, ny)
        # torch.nn.Softmax()
    )

    loss = torch.nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), lr=eta)
    return model, loss, optim


def loss_accuracy(Ytilde, Y, loss):
    # TODO

    Yhat = torch.nn.Softmax(dim=1)(Ytilde)
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


if __name__ == '__main__':
    # Start:
    data = MNISTData()
    N = data._Xtrain_th.shape[0]
    Nepoch = 300 # 50
    Nbatch = 1000
    nx = data._Xtrain_th.shape[1]
    nh = 10
    ny = data._Ytrain_th.shape[1]
    eta = 0.02
    print('Data set size: ', N)
    model, loss, optim = init_model(nx, nh, ny, eta)
    printInterval = 50
    trainLosses = []
    testLosses = []

    # Seeing a bit of the dataset:
    print('Glimpse of the dataset:')
    print('Xtrain.shape: ', data._Xtrain_th.shape)
    print('Ytrain.shape: ', data._Ytrain_th.shape)
    # print('Beginning of Ytrain (one_hot encoding):\n', data._Ytrain_th[0:10])

    # TODO apprentissage
    for i in range(Nepoch):
        for j in range(int(N/Nbatch)):
            X = data._Xtrain_th[j * Nbatch : (j+1) * Nbatch]
            Y = data._Ytrain_th[j * Nbatch : (j+1) * Nbatch]

            Ytilde = model(X)
            L, accuracy = loss_accuracy(Ytilde, Y, loss)
            optim.zero_grad()
            L.backward()
            optim.step()

        if i % printInterval == 0:
            print('- Epoch: ', i)
            # Training set
            Ytilde = model(data._Xtrain_th)
            L, accuracy = loss_accuracy(Ytilde, data._Ytrain_th, loss)
            trainLosses.append(L.item())
            print(' Training set:\n', '   loss: ', L.item(), '; accuracy: ', accuracy, sep='')

            # Testing set
            Ytilde = model(data._Xtest_th)
            L, accuracy = loss_accuracy(Ytilde, data._Ytest_th, loss)
            print(' Test set:\n', '   loss: ', L.item(), '; accuracy: ', accuracy, sep='')
            testLosses.append(L.item())


    with torch.no_grad():
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
        plt.draw()
        plt.pause(1)
        input("Press enter to continue")
        plt.close()
        

        # Testing visually our trained model:
        nTestExamples = 5
        print('\nTesting visually our trained model with', nTestExamples, 'examples')
        for i  in range(nTestExamples):
            # Pick a random X from the test set
            index = random.randrange(data._Xtest_th.shape[0])
            
            # The first column is the label
            Y = data._Ytest_th[index]

            # The rest of columns are pixels
            X = data._Xtest_th[index]
            X = X.view(1, -1)
            # X.reshape((1, X.shape[0]))

            Ytilde = model(X)
            print('Ytilde.shape:', Ytilde.shape)
            # Ytilde.reshape((1, Ytilde.shape[0]))
            Yhat = torch.nn.Softmax(dim=1)(Ytilde)
            predicted = np.argmax(Yhat)

            # For the next steps:
            pixels = X
            label = np.argmax(Y)

            # Reshape the array into 28 x 28 array (2-dimensional array):
            pixels = pixels.reshape((28, 28))

            # Plot
            plt.close()
            plt.title('Label is {label} and predicted was {predicted}'.format(label=label, predicted=predicted))
            plt.imshow(pixels, cmap='gray')
            plt.draw()
            plt.pause(1)

            input("Press enter to continue")
            plt.close()

            


        







# if __name__ == '__main__':
#     # init
    
#     X, Y = mnist_data()
#     Y.resize((Y.shape[0], 1))

#     # Suffle our dataset
#     rdnState = np.random.get_state()
#     np.random.shuffle(X)
#     np.random.set_state(rdnState)
#     np.random.shuffle(Y)

#     # Seeing a bit of the dataset:
#     print('Glimpse of the dataset:')
#     print('X.shape: ', X.shape)
#     print('type(X[0][0]): ', type(X[0][0]))
    # print('Y.shape: ', Y.shape)
    # print('type(Y[0][0]): ', type(Y[0][0]))
    # print('Beginning of Y:\n', Y[0:10])
    
    # totalDatasetSize = X.shape[0]
    # trainningSize = int(totalDatasetSize * 0.8)
    # Xtrain = X[0:trainningSize]
    # Xtest = X[trainningSize+1:]

    # Ytrain = Y[0:trainningSize]
    # Ytest = X[trainningSize+1:]

    # # data = CirclesData()
    # # data.plot_data()
    # # N = data.Xtrain.shape[0]
    # N = Xtrain.shape[0]

    # Nepoch = 800 
    # Nbatch = 10
    # nx = Xtrain.shape[1]
    # nh = 10
    # ny = Ytrain.shape[1]
    # eta = 0.02
    # print('Data set size: ', N)  
    # model, loss, optim = init_model(nx, nh, ny, eta)
    # printInterval = 100
    # trainLosses = []
    # testLosses = []

    # # TODO apprentissage
    # for i in range(Nepoch):
    #     for j in range(int(N/Nbatch)):
    #         X = torch.from_numpy(Xtrain[j * Nbatch : (j+1) * Nbatch])
    #         Y = torch.from_numpy(Ytrain[j * Nbatch : (j+1) * Nbatch])

    #         # Yhat, outputs = forward(params, X)
    #         Ytilde = model(X)

    #         L, accuracy = loss_accuracy(Ytilde, Y, loss)

    #         # grads = backward(params, outputs, Y)
    #         optim.zero_grad()
    #         L.backward()
    #         optim.step()
    #         # model = sgd(model, eta)

    #     if i % printInterval == 0:
    #         print('- Epoch: ', i)
    #         # Training set
    #         # Yhat, _ = forward(params, torch.from_numpy(data._Xtrain))
    #         Ytilde = model(torch.from_numpy(Xtrain))
    #         L, accuracy = loss_accuracy(Ytilde, torch.from_numpy(Ytrain), loss)
    #         trainLosses.append(L.item())
    #         print(' Training set:\n', '   loss: ', L.item(), '; accuracy: ', accuracy, sep='')

    #         # Testing set
    #         # Yhat, _ = forward(params, torch.from_numpy(data._Xtest))
    #         Ytilde = model(torch.from_numpy(Xtest))
    #         L, accuracy = loss_accuracy(Ytilde, torch.from_numpy(Ytest), loss)
    #         print(' Test set:\n', '   loss: ', L.item(), '; accuracy: ', accuracy, sep='')
    #         testLosses.append(L.item())

    #         # Ygrid, _ = forward(params, data.Xgrid)
    #         # Ygridtilde = model(data.Xgrid)
    #         # Ygridhat = torch.nn.Softmax(dim=1)(Ygridtilde) # Softmax is not inside the "model" because "torch.nn.CrossEntropyLoss()" already has it, so you have to do this step to transform "Ygridtilde" in "Ygridhat" 
    #         # data.plot_data_with_grid(Ygridhat.detach())

    # x = np.arange(0, Nepoch, printInterval)
    # trainLosses = np.array(trainLosses)
    # testLosses = np.array(testLosses)
    # plt.close()
    # plt.clf()
    # plt.ylabel('Losses')
    # plt.xlabel('Epoch')
    # plt.title('Train loss and Test loss with time')
    # plt.grid(True)
    # plt.plot(x, trainLosses, 'b--', x, testLosses, 'r--')
    # plt.legend(['Train loss', 'Test loss'], loc=1)
    # plt.show()

    # # attendre un appui sur une touche pour garder les figures
    # # input("done")
