# import torch

from tme5 import CirclesData  # import de la classe
from circles import init_params, forward, loss_accuracy, backward, sgd

if __name__ == '__main__':
    # Chargement de la classe
    data = CirclesData()  # instancie la classe fournie

    # Acces aux donn ees
    Xtrain = data.Xtrain  # torch.Tensor contenant les entr ́ees du r ́eseau pour 􏱈→ l'apprentissage
    # print('Xtrain.shape: ', Xtrain.shape) # affiche la taille des donn ́ees : torch.Size([200, 2]) N = Xtrain.shape[0] # nombre d'exemples

    nx = Xtrain.shape[1]  # dimensionalite d'entree
    # donn ees disponibles : data.Xtrain, data.Ytrain, data.Xtest, data.Ytest, -> data.Xgrid
    # Fonctions d'affichage
    # data.plot_data() # affiche les points de train et test

    # Test of init_params
    # nh = 4
    # n = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(0.3))
    # wh = n.sample((nh,))
    # print('wh: ', wh.shape, '\n', wh)

    Ytrain = data.Ytrain
    # print(Xtrain[0])
    # print('Ytrain: ', Ytrain.shape)
    # print(Ytrain)
    nh = 6
    ny = 2
    params = init_params(nx, nh, ny)
    Yhat, outputs = forward(params, data.Xtrain)
    # print(Yhat)

    L = loss_accuracy(Yhat, Ytrain)
    print('loss_accuracy: ', L[0].item(), L[1])

    # Ygrid = forward(params, data.Xgrid) # calcul des predictions Y pour tous les points
    # -> de la grille (forward et params non fournis, `a coder) data.plot_data_with_grid(Ygrid) # affichage des points et de la fronti`ere de
    # -> d ́ecision gr^ace `a la grille

    # data.plot_loss(loss_train, loss_train, acc_train, acc_test) # affiche les courbes
    # -> delossetaccuracyentrainettest.Lesvaleursa`fournirsontdesscalaires,
    #-> elles sont stock ́ees pour vous, il suffit de passer les nouvelles valeurs a`
    # -> chaque ite ́ration

    # Backward section
    grads = backward(params, outputs, Ytrain)
    # print('nx: ', nx)
    # print('ny: ', ny)
    # print('nh: ', nh)
    # print('grads["Wy"].shape: ', grads["Wy"].shape) # must be: nh x ny
    # print('grads["by"].shape: ', grads["by"].shape) # must be: ny x 1 
    # print('grads["Wh"].shape: ', grads["Wh"].shape) # must be: nx x nh
    # print('grads["bh"].shape: ', grads["bh"].shape) # must be: nh x 1

    # SGD section
    # print('Wy:\n', params["Wy"])
    params = sgd(params, grads, eta = 1e-2)
    # print('new Wy:\n', params["Wy"])
    
