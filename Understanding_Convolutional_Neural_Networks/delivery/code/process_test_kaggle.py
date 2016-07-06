from __future__ import division
from glob import glob
import os
import cv2
import numpy

""" Returns the kaggle images sorted by file number"""


def get_kaggle_test(path, normalize):
    files_list = glob(os.path.join(path, '*.png'))
    file_nums = [int(os.path.basename(x).split('.')[0]) for x in files_list]
    sorted_files = [x for (y, x) in sorted(zip(file_nums, files_list))]
    imgs = [ cv2.imread(x, cv2.IMREAD_COLOR).transpose(2, 0, 1) for x in sorted_files]
    if normalize:
        imgs = [ x / 256.0 for x in imgs]
    return numpy.asarray(imgs)

if __name__ == '__main__':
    get_kaggle_test('/home/dani/Downloads/test')