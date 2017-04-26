# from model import VGG16
import numpy as np
# from utils import Data
import cPickle as pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# root_path = '/home/zhangzimou/Desktop/Statistical_ML/ece6254_data'
# vgg16_weights_path = root_path + '/vgg16_weights.h5'
# our_weights_path = root_path + '/weights.h5'
# data = Data(root_path, load_size=1)
# model = VGG16(our_weights_path, vgg16_weights_path)
# model.fit(data, batch_size=8, nb_epoch=1)

# with open('feature.pkl', 'rb') as f:
#     X = np.array(pickle.load(f))

# with open('label.pkl', 'rb') as f:
#     y = np.array(pickle.load(f))

# idx = np.where(y <= 2)
# y0 = y[idx]
# X0 = X[idx]

# pca = PCA(n_components=50)
# X_pca = pca.fit_transform(X0)

# tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X_pca, y)


from mnist import MNIST
mndata = MNIST('./')
images, labels = mndata.load_training()
X = np.array(images)[:6000]
y = np.array(labels)[:6000]

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
print 'PCA done'
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_pca, y)
print 'TSNE done'


def plot_tsne():
    color_list = ['#f44242', '#f48341', '#f4e841', '#9df441', '#41f4a0',
                  '#41c1f4', '#4152f4', '#8541f4', '#f441f1', '#f44170']
    for i in range(10):
        idx = np.where(y == i)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], color=color_list[i])


plot_tsne()
