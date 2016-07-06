import cPickle
import os
import numpy

NUM_CLASSES = 10
IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_CHANNELS = 3
NUM_TRAINING_BATCHES = 5
BATCH_SIZE = 10000

""" Reads the cifar batch into a struct """


def read_batch(f):
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d


""" Reads the given batches as integers for Lasagne compatibility"""


def read_batches(cifar_folder, batches, normalize):
    data = numpy.empty([BATCH_SIZE * len(batches), NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH], dtype=numpy.float32)
    labs = []
    for i, batch_num in enumerate(batches):
        batch_data = read_batch(os.path.join(cifar_folder, 'data_batch_' + str(batch_num)))
        data[(i * BATCH_SIZE):((i + 1) * BATCH_SIZE)] = \
            numpy.reshape(batch_data['data'], [BATCH_SIZE, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH])
        labs = labs + batch_data['labels']
    if normalize:
        data /= numpy.float32(256)
    return data, numpy.asarray(labs).astype(numpy.int32)


""" Reads the CIFAR data using the given batches for each set """


def read_cifar_images(cifar_folder, normalize, train_batches, val_batch, test_batch):
    x_train, y_train = read_batches(cifar_folder, train_batches, normalize)
    x_val, y_val = read_batches(cifar_folder, [val_batch], normalize)
    x_test, y_test = read_batches(cifar_folder, [test_batch], normalize)
    return x_train, y_train, x_val, y_val, x_test, y_test
