from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import (SGD, RMSprop)
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import h5py
import os
from utils import (process_img, get_time)
import numpy as np
import cPickle as pickle


class VGG16(object):

    def __init__(self, our_weights_path, vgg16_weights_path):
        self.weights_path = our_weights_path
        self.vgg16_weights_path = vgg16_weights_path
        # if not os.path.isfile(our_weights_path):
        #     self.compile_vgg16()
        # else:
        #     self.model = load_model(our_weights_path)
        self.compile_vgg16()

    def fit(self, data, batch_size, nb_epoch):
        for X, y in data.load_train(nb_epoch=nb_epoch):
            self.model.fit(X, y, batch_size=batch_size, epochs=1,
                           verbose=1)
        self.model.save(self.weights_path)

    def compile_vgg16(self, trainable=False):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu', trainable=trainable))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu', trainable=trainable))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu', trainable=trainable))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu', trainable=trainable))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu', trainable=trainable))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu', trainable=trainable))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu', trainable=trainable))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', trainable=trainable))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', trainable=trainable))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', trainable=trainable))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', trainable=trainable))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', trainable=trainable))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', trainable=trainable))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu', trainable=trainable))
        model.add(Dropout(1))
        model.add(Dense(4096, activation='relu', trainable=trainable))
        # model.add(Dropout(1))

        f = h5py.File(self.vgg16_weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the
                # savefile
                break
            g = f['layer_{}'.format(k)]
            weights = []
            for p in range(g.attrs['nb_params']):
                data = g['param_{}'.format(p)]
                if data.ndim == 4:
                    weights.append(np.transpose(data, [2, 3, 1, 0]))
                else:
                    weights.append(data)

            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')

        model.add(Dense(10, activation='softmax'))

        # opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        opt = RMSprop(decay=1e-6)
        model.compile(optimizer=opt, loss='categorical_crossentropy')
        self.model = model

    
    
