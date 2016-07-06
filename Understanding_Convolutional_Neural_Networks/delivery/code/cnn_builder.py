from __future__ import print_function

import lasagne

""" Builds a convolutional network with 2 conv layer with pooling and normalization, followed by
a hidden layer and a softmax layer given the input parameters """


def build_cnn(channels, image_height, image_width, num_classes, input_var=None, fsize1=5, fsize2=5, pool_s=2,
              kernels=[30, 30], hidden_size=256, dropout=0.5):

    print("Channels: " + str(channels))
    print("Image height: " + str(image_height))
    print("Image widht: " + str(image_width))
    print("Classes: " + str(num_classes))
    print("Mask 1: " + str(fsize1))
    print("Mask 2: " + str(fsize2))
    print("Pool area: " + str(pool_s))
    print("Kernel sizes: " + str(kernels))
    print("Hidden nuerons: " + str(hidden_size))
    print("Dropout: " + str(dropout))

    network = lasagne.layers.InputLayer(shape=(None, channels, image_height, image_width),
                                        input_var=input_var)
    # First convolutional layer
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=kernels[0], filter_size=(fsize1, fsize1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.01, mean=0))
    # Using default parameters
    network = lasagne.layers.LocalResponseNormalization2DLayer(
        network, alpha=0.0001, k=2, beta=0.75, n=5)
    # Max-pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(pool_s, pool_s))

    # Another convolution
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=kernels[1], filter_size=(fsize2, fsize2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.01, mean=0))
    # Using default parameters
    network = lasagne.layers.LocalResponseNormalization2DLayer(
        network, alpha=0.0001, k=2, beta=0.75, n=5)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(pool_s, pool_s))

    # A fully-connected layer
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout),
            num_units=hidden_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(std=0.01, mean=0))

    # Softmax layer
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropout),
            num_units=num_classes,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.Normal(std=0.01, mean=0))

    return network
