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
from sklearn import svm
import cPickle as pickle


class VGG16(object):

    def __init__(self, our_weights_path, vgg16_weights_path):
        self.weights_path = our_weights_path
        self.vgg16_weights_path = vgg16_weights_path
        if not os.path.isfile(our_weights_path):
            self.compile_vgg16()
        else:
            self.model = load_model(our_weights_path)

    '''def fit(self, data, batch_size, nb_epoch):
        for X, y in data.load_train(nb_epoch=nb_epoch):
            self.model.fit(X, y, batch_size=batch_size, epochs=1,
                           verbose=2)
        self.model.save(self.weights_path)'''

    def load_features(self, path):
        with open(path + '/feature.pkl', 'rb') as f1:
            X = pickle.load(f1)
        with open(path + '/label.pkl', 'rb') as f2:
            y = pickle.load(f2)
        return X,y

    def predict(self, data, batch_size, nb_epoch):
        label = []
        feature = []
	i=0
        for X,y in data.load_train(nb_epoch=nb_epoch):
            #print y
            label.extend(y)
	    print i
            try:
                temp = self.model.predict(X, batch_size=batch_size,verbose=1)
            except:
                print X.shape
            #print temp[0]
            #print temp[0].shape
            feature.append(temp[0])
	    i+=1
            #feature.extend(self.model.predict(X, batch_size=batch_size,verbose=1))
        #self.model.save(self.weights_path)
        return label,feature

    def test(self, mode):
        if(mode == 'SVM'):
            raw_X, y = self.load_features('/home/jhy/Desktop/ece6254/ece6254_data/')
	    raw_X, y = np.array(raw_X), np.array(y)
            X= self.PCA_reduction(raw_X)
	    print X.shape
            C, gamma = self.SVM_fit(X,y)
            prediction = self.SVM_test(X, C, gamma)
            error = self.Count_error_rate(prediction, y)
            print error

    def SVM_fit(self, X, y):
        Cvec = np.logspace(-5,5,11)
        Gammaval = 1e-6
        Pe_train = []
        for Cval in Cvec:
            clf = svm.SVC(C=Cval,kernel='rbf',gamma=Gammaval)
            clf.fit(X,y)
            Pe_train.append(1.0-clf.score(X,y))

        print 'Optimal value of C based on train set: ' + str(Cvec[np.argmin(Pe_train)])
        return Cvec[np.argmin(Pe_train)], Gammaval

    def SVM_test(self, X, Cval, Gammaval):
        clf = svm.SVC(C=Cval,kernel='rbf',gamma=Gammaval)
        prediction = clf.predict(X)
        return prediction

    def Count_error_rate(self, prediction, y):
        i = 0
        count = 0
        for label in y:
            if(label == prediction[i]):
                count+=1
            i+=1

        return count/(i+1)

    def PCA_reduction(self, raw_X):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=200)
        X = pca.fit_transform(raw_X)
        return X


    def compile_vgg16(self):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        #model.add(Dropout(0.5))

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

        #model.add(Dense(10, activation='softmax'))

        # opt = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        opt = RMSprop(decay=1e-6)
        model.compile(optimizer=opt, loss='categorical_crossentropy')
        self.model = model
