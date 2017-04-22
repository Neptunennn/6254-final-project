from model import VGG16
from utils import Data
import h5py


root_path = '../../ece6254_data'
vgg16_weights_path = '../../ece6254_data/vgg16_weights.h5'
our_weights_path = '../../ece6254_data/weights.h5'
data = Data(root_path, load_size=5000)
model = VGG16(our_weights_path, vgg16_weights_path)
model.fit(data, batch_size=32, nb_epoch=1)

