import time
import os
import lasagne
import numpy as np

""" Saves the lists of the results of the training process into the desired output"""


def save_results(output, network, train_acc_l, train_loss_l, val_loss_l, val_acc_l,
                 test_loss_res, test_acc, results, truth, results_kaggle):
    # Create folder with timestamp
    d = os.path.join(output, str(time.time()).split('.')[0])
    os.makedirs(d)
    # Create files into folder
    save_list(train_acc_l, os.path.join(d, 'train_acc.txt'))
    save_list(train_loss_l, os.path.join(d, 'train_loss.txt'))
    save_list(val_loss_l, os.path.join(d, 'val_loss.txt'))
    save_list(val_acc_l, os.path.join(d, 'val_acc.txt'))
    test_results = [test_loss_res, test_acc]
    save_list(test_results, os.path.join(d, 'test_loss_and_acc.txt'))
    # Save predicted results
    np.savetxt(os.path.join(d, 'predicted.txt'), results, delimiter=',')
    np.savetxt(os.path.join(d, 'truth.txt'), truth, delimiter=',')
    if results_kaggle is not None:
        np.savetxt(os.path.join(d, 'predicted_kaggle.txt'), results_kaggle, delimiter=',')
    # Save network just in case
    np.savez(os.path.join(d, 'model.npz'), *lasagne.layers.get_all_param_values(network))


def save_list(l, path):
    with open(path, 'w') as out_file:
        out_file.write(str(l))


def read_parameters(path):

    params = dict()
    with open(path, 'r') as input_params_file:
        for line in input_params_file:
            if line.startswith("epochs="):
                params['epochs'] = int(line.replace("epochs=",""))
            elif line.startswith("batch_size="):
                params['batch_size'] = int(line.replace("batch_size=",""))
            elif line.startswith("fsize1="):
                params['fsize1'] = int(line.replace("fsize1=",""))
            elif line.startswith("fsize2="):
                params['fsize2'] = int(line.replace("fsize2=",""))
            elif line.startswith("pool_s="):
                params['pool_s'] = int(line.replace("pool_s=",""))
            elif line.startswith("kernels="):
                aux = line.replace("kernels=[","")
                aux = aux.replace("]","")
                aux = aux.split()
                if len(aux)<2:
                    aux = aux.split(",")
                else:
                    params['kernels'] = [0]*2
                    aux[0] = aux[0].replace(",","")
                    aux[1] = aux[1].replace(",","")
                params['kernels'][0] = int(aux[0])
                params['kernels'][1] = int(aux[1])
            elif line.startswith("hidden_size="):
                params['hidden_size'] = int(line.replace("hidden_size=",""))
            elif line.startswith("dropout="):
                params['dropout'] = float(line.replace("dropout=",""))
            elif line.startswith("learning_rate="):
                params['learning_rate'] = float(line.replace("learning_rate=",""))
            elif line.startswith("momentum="):
                params['momentum'] = float(line.replace("momentum=",""))
            elif line.startswith("l1="):
                params['l1'] = True if int(line.replace("l1=","")) == 1 else False
                print(params['l1'])
            elif line.startswith("l2="):
                params['l2'] =  True if int(line.replace("l2=","")) == 1 else False
                print(params['l2'])
            else:
                print "Unkown tag in line {}".format(line)

        input_params_file.close()
    return params