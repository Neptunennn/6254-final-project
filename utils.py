import numpy as np
import cv2
import time
import os
from keras.utils import to_categorical
import cPickle as pickle


def process_img(path, size=(224, 224)):
    img = cv2.imread(path).astype(np.float64)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    return img


class Buffer(object):

    def __init__(self, root_path, files, labels=None):
        if labels:
            self.mode = 'train'
            self.files = files
            self.labels = labels
        else:
            self.mode = 'test'
            self.files = files

        self.root_path = root_path

    def __getitem__(self, key):
        path = self.root_path
        if isinstance(key, slice):
            pass
        else:
            path += '/c{}'.format(key)
            pass


class Data(object):

    def __init__(self, root_path, load_size=1000):
        self.root_path = root_path
        self.load_size = load_size

    def load_train(self, nb_epoch=1):
        train_dir = os.path.join(self.root_path, 'train')
        files = []
        labels = []
        for i in range(10):
            path = os.path.join(train_dir, 'c{}'.format(i))
            a = os.listdir(path)
            files.extend(a)
            labels.extend([i] * len(a))

        assert len(files) == len(labels)
        files, labels = np.array(files), np.array(labels)
        idx = np.random.permutation(files.shape[0])
        files, labels = files[idx], labels[idx]
        start, epoch = 0, 0
        while True:
            end = start + self.load_size
            if end > files.shape[0]:
                epoch += 1
                X = self.file2img(files[start:], labels[start:])
                y = to_categorical(labels[start:], num_classes=10)
                start = 0
            else:
                X = self.file2img(files[start:end], labels[start:end])
                y = to_categorical(labels[start:end], num_classes=10)
                start += self.load_size
            yield X, y
            if epoch >= nb_epoch:
                break

    def load_test(self, nb_epoch=1):
        test_dir = os.path.join(self.root_path, 'test')
        files = os.listdir(test_dir)
        files = np.array(files)
        start, epoch = 0, 0
        while True:
            end = start + self.load_size
            if end > files.shape[0]:
                epoch += 1
                X = self.file2img(files[start:])
                start = 0
            else:
                X = self.file2img(files[start:end])
                start += self.load_size
            yield X
            if epoch >= nb_epoch:
                break

    def file2img(self, files, labels=None):
        imgs = []
        if labels is not None:
            for i, f in enumerate(files):
                path = self.root_path + '/train/c{}/{}'.format(labels[i], f)
                imgs.append(process_img(path))
        else:
            for f in files:
                path = self.root_path + '/test/{}'.format(f)
                imgs.append(process_img(path))

        return np.array(imgs)

    def load_train2(self, use_cache=False):
        X, y = [], []
        print('Reading train images')
        cache_file = os.path.join(self.root_path, 'cache', 'train.p')
        if use_cache and os.path.isfile(cache_file):
            print('Reading from cache')
            with open(cache_file, 'rb') as f:
                X, y = pickle.load(f)

        else:
            for i in range(10):
                print('Loading folder c{}'.format(i))
                dir_path = os.path.join(
                    self.root_path, 'train', 'c{}'.format(i))
                files = os.listdir(dir_path)
                for f in files:
                    X.append(process_img(dir_path + '/' + f))
                    y.append(i)

        X, y = np.array(X), to_categorical(y, num_classes=10)
        idx = np.random.permutation(X.shape[0])
        X, y = X[idx], y[idx]
        self.train_X, self.train_y = X, y
        return X, y

    def load_test2(self):
        X = []
        print('Reading test images')
        dir_path = os.path.join(self.root_path, 'test')
        files = os.listdir(dir_path)
        for f in files:
            X.append(process_img(dir_path + '/' + f))

        X = np.array(X)
        self.test_X = X
        return X

    def cache_data(self, mode='train'):
        if mode == 'train':
            cache_file = os.path.join(self.root_path, 'cache', 'train.p')
            with open(cache_file, 'wb') as f:
                pickle.dump((self.train_X, self.train_y), f)

        elif mode == 'test':
            cache_file = os.path.join(self.root_path, 'cache', 'test.p')
            with open(cache_file, 'wb') as f:
                pickle.dump(self.test_X, f)


def get_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("Total time: {:.2e}s".format(end - start))
        return res

    return wrapper
