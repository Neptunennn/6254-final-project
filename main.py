from model import VGG16
from utils import Data
import h5py
import pickle
import numpy as np


root_path = '/home/jhy/Desktop/ece6254/ece6254_data'
vgg16_weights_path = '/home/jhy/Desktop/ece6254/ece6254_data/vgg16_weights.h5'
our_weights_path = '/home/jhy/Desktop/ece6254/ece6254_data/weights.h5'

data = Data(root_path, load_size=1)
model = VGG16(our_weights_path, vgg16_weights_path)

model.test('SVM')





'''X,y = model.load_features(root_path)
X,y = np.array(X), np.array(y)
print X.shape, y.shape'''




#Y,X =  model.predict(data, batch_size=32, nb_epoch=1)
#output1 = open('feature.pkl','wb')
#output2 = open('label.pkl','wb')
#pickle.dump(X,output1,-1)
#pickle.dump(Y,output2,-1)
#output1.close()
#output2.close()





