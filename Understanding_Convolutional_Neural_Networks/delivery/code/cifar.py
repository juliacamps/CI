#!/usr/bin/env python

"""
This code has been adapted from the MNIST Example from LAsagne documentation
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
"""

from __future__ import print_function
import os
import time
import io_cnn as io
import numpy
import theano
import theano.tensor as T
import download_and_extract as de
import cifar_input as cifar
import caltech_input as caltech
import cnn_builder as cnb
import cnn_utils as utils
import lasagne
from lasagne.regularization import regularize_layer_params, l2, l1

# Constants

CALTECH_HEIGHT = 80
CALTECH_WIDTH = 80
CALTECH_CHANNELS = 1
CALTECH_CLASSES = 102
CALTECH_PATH = '101_ObjectCategories'

CIFAR_HEIGHT = 32
CIFAR_WIDTH = 32
CIFAR_CHANNELS = 3
CIFAR_CLASSES = 10
CIFAR_PATH = 'cifar-10-batches-py'
CIFAR_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

ACC_THRESH = 1.00
acc_buffer = []


def main(output, input_params_path, dataset='CIFAR'):
    # Read parameters
    parameters = io.read_parameters(input_params_path)
    num_epochs = parameters['epochs']
    batch_size = parameters['batch_size']

    print("Batch size: " + str(batch_size))
    print("Max epochs: " + str(num_epochs))
    print("Learning rate: " + str(parameters['learning_rate']))
    print("Momentum: " + str(parameters['momentum']))

    # Initialize metrics
    init_time = time.time()
    train_acc_list, train_loss_list, val_loss_list, val_acc_list = [], [], [], []

    # Load the dataset
    print("Loading data...")
    if dataset == 'CIFAR':
        channels = CIFAR_CHANNELS
        width = CIFAR_WIDTH
        height = CIFAR_HEIGHT
        num_classes = CIFAR_CLASSES
        # Check if dataset exists. If not
        if not os.path.exists(CIFAR_PATH):
            de.download_and_extract(CIFAR_URL, '.')
        x_train, y_train, x_val, y_val, x_test, y_test = cifar.read_cifar_images(CIFAR_PATH, True, [1, 2, 3], 4, 5)
    else:
        channels = CALTECH_CHANNELS
        width = CALTECH_WIDTH
        height = CALTECH_HEIGHT
        num_classes = CALTECH_CLASSES
        x_train, y_train, x_val, y_val, x_test, y_test = \
            caltech.read_subset_caltech(CALTECH_PATH, height, width, 'log', 20, 5, 5, True)

    x_val = x_val[range(0, 2000)]
    y_val = y_val[range(0, 2000)]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = cnb.build_cnn(channels, height, width, num_classes, input_var, parameters['fsize1'],
                            parameters['fsize2'], parameters['pool_s'], parameters['kernels'],
                            parameters['hidden_size'],
                            parameters['dropout'])

    # Assign loss function: cross entropy
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

    # Add regularization if needed
    if parameters['l1'] and parameters['l2']:
        print("Selected L1 + L2")
        l1_penalty = regularize_layer_params(network, l1) * 1e-4
        l2_penalty = regularize_layer_params(network, l2) * 1e-4
        loss = loss.mean() + l1_penalty + l2_penalty
    elif parameters['l1']:
        print ("Selected L1")
        l1_penalty = regularize_layer_params(network, l1) * 1e-4
        loss = loss.mean() + l1_penalty
    elif parameters['l2']:
        print ("Selected L2")
        l2_penalty = regularize_layer_params(network, l2) * 1e-4
        loss = loss.mean() + l2_penalty
    else:
        print("No regularization")
        loss = loss.mean()

    # Update algorithm is SGD with momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.momentum(loss, params, parameters['learning_rate'], parameters['momentum'])
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    # Accuracy and loss for validation/test
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err, train_batches, train_acc = 0, 0, 0
        start_time = time.time()
        for batch in utils.iterate_minibatches(x_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            t_err, train_a = val_fn(inputs, targets)
            train_acc += train_a
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in utils.iterate_minibatches(x_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        # Save and print validation results at current epoch
        train_acc_res = train_acc / train_batches
        train_loss_res = train_err / train_batches
        val_loss_res = val_err / val_batches
        val_acc_res = val_acc / val_batches * 100
        print("  training accuracy:\t\t{:.6f}".format(train_acc_res))
        print("  training loss:\t\t{:.6f}".format(train_loss_res))
        print("  validation loss:\t\t{:.6f}".format(val_loss_res))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc_res))
        train_acc_list.append(train_acc_res)
        train_loss_list.append(train_loss_res)
        val_loss_list.append(val_loss_res)
        val_acc_list.append(val_acc_res)

        # Check if performance is decreasing or increasing too slowly
        acc_buffer.append(val_acc_res)
        if epoch >= 5:
            # Get different from performance in five epoches behind
            previous = acc_buffer.pop(0)
            diff = val_acc_res - previous
            print("Check of validation accuracy. Difference is {}:".format(diff))
            if diff < ACC_THRESH:
                print("Solution is decreasing, must stop")
                break

    # Define prediction function
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

    # After training, we compute and print the test error:
    results = numpy.zeros(x_test.shape[0], dtype=int)
    test_err, test_acc, test_batches = 0, 0, 0
    for batch in utils.iterate_minibatches(x_test, y_test, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        results[test_batches * batch_size: (test_batches + 1) * batch_size] = predict_fn(inputs)
        test_err += err
        test_acc += acc
        test_batches += 1

    # Show final results
    end_time = time.time()
    test_loss_res = test_err / test_batches
    test_acc_res = test_acc / test_batches * 100
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_loss_res))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc_res))
    print("Took {} seconds to finish".format(end_time - init_time))

    io.save_results(output, network, train_acc_list, train_loss_list, val_loss_list,
                    val_acc_list, test_loss_res, test_acc_res, results, y_test, None)

if __name__ == '__main__':
    main('.', 'conf.properties')
