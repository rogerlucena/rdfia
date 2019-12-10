import argparse
import os
import time

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torchvision.models as models
from PIL import Image
import pickle
from torch.nn import functional as F

# import functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
# import import

torchvision.models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
os.environ["TORCH_HOME"] = "/tmp/torch"
PRINT_INTERVAL = 50
CUDA = False

def get_dataset(batch_size, path):
    # Cette fonction permet de recopier 3 fois une image qui
    # ne serait que sur 1 channel (donc image niveau de gris)
    # pour la "transformer" en image RGB. Utilisez la avec
    # transform.Lambda
    def duplicateChannel(img):
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img

    train_dataset = datasets.ImageFolder(path+'/train',
        transform=transforms.Compose([ # TODO Pré-traitement à faire
            duplicateChannel,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
    val_dataset = datasets.ImageFolder(path+'/test',
        transform=transforms.Compose([ # TODO Pré-traitement à faire
            duplicateChannel,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)

    return train_loader, val_loader


def extract_features(data, model):
    # TODO init features matrices
    X = torch.tensor()
    y = torch.tensor()
    for i, (input, target) in enumerate(data):
        if i % PRINT_INTERVAL == 0:
            print('Batch {0:03d}/{1:03d}'.format(i, len(data)))
        if CUDA:
            input = input.cuda()
        # TODO Feature extraction à faire
        feature = model.forward(input)
        X = torch.cat((X, feature), 0)
        y = torch.cat((y, target), 0)
        
    X = F.normalize(X, p=2, dim=1)
    return X, y

class VGG16relu7(torch.nn.Module): 
    def __init__(self):
        super(VGG16relu7, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        # recopier toute la partie convolutionnelle
        self.features = torch.nn.Sequential(*list(vgg16.features.children()))
        # garder une partie du classifieur, -2 pour s'arrêter à relu7 
        self.classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x) 
        return x


def main(params):
    # print('Instanciation de VGG16')
    # vgg16 = models.vgg16(pretrained=True)

    print('Instanciation de VGG16relu7')
    model = VGG16relu7()

    model.eval()
    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()

    '''
    imagenet_classes = pickle.load(open('imagenet_classes.pkl', 'rb')) # chargement des 􏰀→ classes

    # img = Image.open("cat.jpg")
    img = Image.open("everest.jpg") 
    img = img.resize((224, 224), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255
    img = img.transpose((2, 0, 1)) # permutating the axis

    # TODO preprocess image
    img = np.expand_dims(img, 0) # transformer en batch contenant une image x = torch.Tensor(img)
    img = torch.from_numpy(img)

    # TODO calcul forward
    y = model(img)
    ySoftmax = torch.nn.Softmax(dim=1)(y.detach())
    ySoftmax = ySoftmax.detach().numpy() # transformation en array numpy

    # TODO récupérer la classe prédite et son score de confiance
    classOutput = np.argmax(ySoftmax)
    print('Class:', imagenet_classes[classOutput], ', confidence:', ySoftmax[0][classOutput])
    '''
    
    # On récupère les données
    print('Récupération des données')
    train, test = get_dataset(params.batch_size, params.path)

    # Extraction des features
    print('Feature extraction')
    X_train, y_train = extract_features(train, model)
    X_test, y_test = extract_features(test, model)

    # TODO Apprentissage et évaluation des SVM à faire
    # print('Apprentissage des SVM')
    # accuracy = ...
    

'''
# "main" with only one image below:
def main(params):
    print('Instanciation de VGG16')
    vgg16 = models.vgg16(pretrained=True)

    # print('Instanciation de VGG16relu7')
    model = vgg16 # TODO À remplacer par un reseau tronché pour faire de la feature extraction

    model.eval()
    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()

    imagenet_classes = pickle.load(open('imagenet_classes.pkl', 'rb')) # chargement des 􏰀→ classes

    # img = Image.open("cat.jpg")
    img = Image.open("everest.jpg") 
    img = img.resize((224, 224), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255
    img = img.transpose((2, 0, 1)) # permutating the axis

    # TODO preprocess image
    img = np.expand_dims(img, 0) # transformer en batch contenant une image x = torch.Tensor(img)
    img = torch.from_numpy(img)

    # TODO calcul forward
    y = model(img)
    ySoftmax = torch.nn.Softmax(dim=1)(y.detach())
    ySoftmax = ySoftmax.detach().numpy() # transformation en array numpy

    # TODO récupérer la classe prédite et son score de confiance
    classOutput = np.argmax(ySoftmax)
    print('Class:', imagenet_classes[classOutput], ', confidence:', ySoftmax[0][classOutput])

'''


if __name__ == '__main__':

    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='15SceneData', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='activate GPU acceleration')

    args = parser.parse_args()
    if args.cuda:
        CUDA = True
        cudnn.benchmark = True

    main(args)

    # input("done")
