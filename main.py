from model import VGG16
from utils import Data
import h5py

root_path = '/home/zhangzimou/Desktop/Statistical_ML/ece6254_data'
vgg16_weights_path = root_path + '/vgg16_weights.h5'
our_weights_path = root_path + '/weights.h5'
data = Data(root_path, load_size=1)
# model = VGG16(our_weights_path, vgg16_weights_path)
# model.fit(data, batch_size=8, nb_epoch=1)
for X, y in data.load_train():
    print X.shape, y.shape

  
