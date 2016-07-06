#!/usr/bin/env python

"""
This code has been adapted from the MNIST Example from LAsagne documentation
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
"""

from __future__ import print_function
import io_cnn as io
import numpy
import theano
import theano.tensor as T
import process_test_kaggle as pk
import cnn_builder as cnb
import cnn_utils as utils
import lasagne

# Constants

CIFAR_HEIGHT = 32
CIFAR_WIDTH = 32
CIFAR_CHANNELS = 3
CIFAR_CLASSES = 10


""" Given the path of the kaggle test folder and the path of the model to use, its configurations
(must be the same as the one stored in the path) and an output folder, generates the label outputs
for our model"""


def generate_kaggle_delivery(output, model_path, test_folder, input_params_path):

    # Read parameters
    parameters = io.read_parameters(input_params_path)
    num_epochs = parameters['epochs']
    batch_size = parameters['batch_size']

    print("Batch size: " + str(batch_size))
    print("Max epochs: " + str(num_epochs))
    print("Learning rate: " + str(parameters['learning_rate']))
    print("Momentum: " + str(parameters['momentum']))

    # Initialize metrics
    train_acc_list, train_loss_list, val_loss_list, val_acc_list = [], [], [], []

    # Load the dataset
    print("Loading data...")
    channels = CIFAR_CHANNELS
    width = CIFAR_WIDTH
    height = CIFAR_HEIGHT
    num_classes = CIFAR_CLASSES

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = cnb.build_cnn(channels, height, width, num_classes, input_var, parameters['fsize1'],
                            parameters['fsize2'], parameters['pool_s'], parameters['kernels'],
                            parameters['hidden_size'],
                            parameters['dropout'])

    # Load network
    with numpy.load(model_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # Define prediction function
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

    # Predict kaggle batches
    x_kaggle = pk.get_kaggle_test(test_folder, True)
    y_kaggle = numpy.zeros(x_kaggle.shape[0], dtype=int)
    results_kaggle = numpy.zeros(x_kaggle.shape[0], dtype=int)
    kaggle_batches = 0
    for batch in utils.iterate_minibatches(x_kaggle, y_kaggle, batch_size, shuffle=False):
        inputs, targets = batch
        print("Done")
        results_kaggle[kaggle_batches * batch_size: (kaggle_batches + 1) * batch_size] = predict_fn(inputs)
        kaggle_batches += 1

    io.save_results(output, network, train_acc_list, train_loss_list, val_loss_list,
                    val_acc_list, [], [], [], [], results_kaggle)

if __name__ == '__main__':
    generate_kaggle_delivery('.', 'model.npz', 'kaggle_test_folder', 'conf.properties')
