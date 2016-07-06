import os
import json
import numpy as np
import cv2
import math
import random

NUM_CATEGORIES = 102
NUM_CHANNELS = 1

"""Saves label assignation into a file """


def save_cat_mapping(map_list, out_file):
    with open(out_file, 'w') as f:
        json.dump(map_list, f)


""" Returns a list of paths to images, their labels and the map between
label identifiers and label names """


def get_caltech_files_and_categories(root_f):
    images = []
    categories = []
    cat_map = dict()
    cat_num = 0
    for cat in os.listdir(root_f):
        # Define mapping for current category
        cat_map[cat_num] = cat
        for filename in os.listdir(os.path.join(root_f, cat)):
            file_path = os.path.join(root_f, cat, filename)
            if os.path.isfile(file_path):
                images.append(file_path)
                categories.append(cat_num)
        # Update category label
        cat_num += 1
    return images, categories, cat_map

""" Reads grayscale content of the image and and resizes them into given shape """


def get_image_buffer(images, resize, height, width):
    im_size = height * width
    buf = np.empty([len(images), im_size])
    for ind, img in enumerate(images):
        # Read in grayscale for now
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if resize:
            img = cv2.resize(img, (height, width), cv2.INTER_LINEAR)
        buf[ind, ] = img.reshape((1, im_size))
    return buf
    
    
""" Reads grayscale content of the image and and resizes them into given shape """


def get_caltech_tensor(images, categories, train, val, test, height, width, normalize):
    label_list = np.asarray(categories)
    total = train + val + test
    x_train = np.empty([train * NUM_CATEGORIES, NUM_CHANNELS, height, width], dtype=np.float32)
    x_val = np.empty([val * NUM_CATEGORIES, NUM_CHANNELS, height, width], dtype=np.float32)
    x_test = np.empty([test * NUM_CATEGORIES, NUM_CHANNELS, height, width], dtype=np.float32)
    y_train = np.empty([train * NUM_CATEGORIES], dtype=np.int32)
    y_val = np.empty([val * NUM_CATEGORIES], dtype=np.int32)
    y_test = np.empty([test * NUM_CATEGORIES], dtype=np.int32)
    train_counter, eval_counter, test_counter = 0, 0, 0
    for i in range(0, NUM_CATEGORIES):
        # Get images from current category
        imgs_cat = np.where(label_list == i)[0]
        subset_all = random.sample(range(0, len(imgs_cat)), total)

        # Read train
        selected_train = imgs_cat[subset_all[0:train]]
        for p in selected_train:
            img = cv2.imread(images[p], cv2.IMREAD_GRAYSCALE)
            x_train[train_counter, :, :, :] = cv2.resize(img, (height, width), cv2.INTER_LINEAR)
            y_train[train_counter] = categories[p]
            train_counter += 1

        # Read eval
        selected_val = imgs_cat[subset_all[train:train + val]]
        for p in selected_val:
            img = cv2.imread(images[p], cv2.IMREAD_GRAYSCALE)
            x_val[eval_counter, :, :, :] = cv2.resize(img, (height, width), cv2.INTER_LINEAR)
            y_val[eval_counter] = categories[p]
            eval_counter += 1

        # Read test
        selected_test = imgs_cat[subset_all[train + val:]]
        for p in selected_test:
            img = cv2.imread(images[p], cv2.IMREAD_GRAYSCALE)
            x_test[test_counter, :, :, :] = cv2.resize(img, (height, width), cv2.INTER_LINEAR)
            y_test[test_counter] = categories[p]
            test_counter += 1

    if normalize:
        x_train /= np.float32(256)
        x_val /= np.float32(256)
        x_test /= np.float32(256)
    return x_train, y_train, x_val, y_val, x_test, y_test



""" Read Caltech data from the input folder and logs information in
 the desired output folder. Assumes data has fixed size"""


def read_caltech_images(root_folder, height, width, log_folder):
    # Read categories and files and save map between categories integers and labels
    i, c, mapped_cat = get_caltech_files_and_categories(root_folder)
    save_cat_mapping(mapped_cat, os.path.join(log_folder, 'map.json'))
    # Initialize result
    x_data = get_image_buffer(i, True, height, width)
    y_data = np.asarray(c)
    return x_data, y_data
    
""" Read subset of Caltech data"""


def read_subset_caltech(root_folder, height, width, log_folder, imgs_train, imgs_val, imgs_test, normalize):
    # Read categories and files and save map between categories integers and labels
    i, c, mapped_cat = get_caltech_files_and_categories(root_folder)
    save_cat_mapping(mapped_cat, os.path.join(log_folder, 'map.json'))
    return get_caltech_tensor(i, c, imgs_train, imgs_val, imgs_test, height, width, normalize)

# Split data into training and testing given the input ratio
# By http://stackoverflow.com/questions/3674409/numpy-how-to-split-partition-a-dataset-array-into-training-and-test-datasets
# Does not preserve the image ratio per category <- must be fixed


def split_data(x_data, y_data, train_ratio):
    # Compute permutations and split into training and testing indexes
    indices = np.random.permutation(x_data.shape[0])
    training_size = math.floor(x_data.shape[0] * train_ratio)
    training_idx, test_idx = indices[:training_size], indices[training_size:]
    # Initialize sets and divide data
    train = dict()
    test = dict()
    train['data'] = x_data[training_idx, :]
    test['data'] = x_data[test_idx, :]
    train['class'] = y_data[training_idx]
    test['class'] = y_data[test_idx]
    return train, test


""" Creates folder in case it does not exist"""


def create_folder(p):
    if not os.path.exists(p):
        os.makedirs(p)

""" Resizes caltech data and saves the new dataset into the given location"""


def resize_dataset(input_folder, output_folder, height, width):
    images, c, m = get_caltech_files_and_categories(input_folder)

    # Create output folder
    create_folder(output_folder)

    # Iterate through images, resize them and store them
    for i in range(len(images)):
        img = cv2.imread(images[i], cv2.IMREAD_COLOR)
        category = m[c[i]]
        # Create category folder if does not exist
        cat_folder = os.path.join(output_folder, category)
        create_folder(cat_folder)
        # Resize and store image
        res = cv2.resize(img, (height, width), cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(cat_folder, os.path.basename(images[i])), res)
