import os
import gzip
import numpy as np
from chainer.datasets import tuple_dataset

def load_fmnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                           offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                           offset=16).reshape(len(labels), 784)

    return images, labels

def load_dataset():

    name = os.path.dirname(os.path.abspath(__name__))
    joined_path = os.path.join(name, './utils')
    data_path = os.path.normpath(joined_path)

    X_train, y_train = load_fmnist(str(data_path), kind='train')
    X_test, y_test = load_fmnist(str(data_path), kind='t10k')

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    train_data = np.array([X_train[i].reshape(1, 28, 28) for i in range(len(X_train))])
    test_data = np.array([X_test[i].reshape(1, 28, 28) for i in range(len(X_test))])

    y_train = y_train.astype('int8')
    y_test = y_test.astype('int8')

    train = tuple_dataset.TupleDataset(train_data, y_train)
    test = tuple_dataset.TupleDataset(test_data, y_test)

    return train, test
