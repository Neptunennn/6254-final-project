from __future__ import division
import numpy as np
import cv2
import time
import os
from keras.utils import to_categorical
import cPickle as pickle
import sys


def process_img(path, size=(224, 224)):
    img = cv2.imread(path)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img


class Data(object):

    def __init__(self, root_path, load_size=1000):
        self.root_path = root_path
        self.load_size = load_size

    def load_train(self, nb_epoch=1, y_oneHot=True):
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
            if end >= files.shape[0]:
                epoch += 1
                X = self.file2img(files[start:], labels[start:])
                if y_oneHot:
                    y = to_categorical(labels[start:], num_classes=10)
                else:
                    y = labels[start:]
                start = 0
            else:
                X = self.file2img(files[start:end], labels[start:end])
                if y_oneHot:
                    y = to_categorical(labels[start:end], num_classes=10)
                else:
                    y = labels[start:end]
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


def serialize_data(root_path, mode='train'):
    if mode == 'train':
        file_dir = os.path.join(root_path, 'train')
        files, labels = [], []
        for i in range(10):
            path = os.path.join(file_dir, 'c{}'.format(i))
            tmp = os.listdir(path)
            files.extend(tmp)
            labels.extend([i] * len(tmp))

        assert len(files) == len(labels)
        files, labels = np.array(files), np.array(labels)
        idx = np.random.permutation(files.shape[0])
        files, labels = files[idx], labels[idx]
    else:
        file_dir = os.path.join(root_path, 'test')
        path = os.path.join(root_path, 'test')
        files = os.listdir(path)

    X = []
    y = labels
    N = len(files)
    if mode == 'train':
        for i, f in enumerate(files):
            file_path = root_path + '/train/c{}/{}'.format(labels[i], f)
            X.append(process_img(file_path))
            if i % 500 == 0:
                print("Complete {:.1%}".format(i / N))
        assert len(X) == len(y)
        X, y = np.array(X), np.array(y)
        print("Complete 100%, pickling...")
        with open(mode + '.pkl', 'wb') as f:
            pickle.dump((X, y), f, 2)
        print("Pickling complete!")
    else:
        for f in files:
            file_path = root_path + '/test/{}'.format(f)
            X.append(process_img(file_path))
            if i % 500 == 0:
                print("Complete {:.1%}".format(i / N))
        X = np.array(X)
        print("Complete 100%, pickling...")
        with open(mode + '.pkl', 'wb') as f:
            pickle.dump(X, f, 2)
        print("Pickling complete!")


def get_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("Total time: {:.2e}s".format(end - start))
        return res

    return wrapper
