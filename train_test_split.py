import os
import shutil

import numpy as np

from constants import *


# Creates two different set of folder with training set and testing set randomly.
def split():
    no_of_img = len(os.listdir(cropped_img_path))

    # selecting 10% of the data for test set
    no_of_testimages = int(no_of_img / 10)
    rand = np.random.choice(no_of_img, no_of_testimages)

    # making directories for training and test sets
    if not os.path.isdir(training_set) and not os.path.isdir(testing_set):
        os.makedirs(training_set)
        os.makedirs(testing_set)

    # moving images into their respective directories
    with os.scandir(cropped_img_path) as images:
        count = 0
        # here count is used as an index to move the random files
        for image in images:
            if count in rand:
                shutil.move(cropped_img_path + os.sep + image.name, testing_set)
            else:
                shutil.move(cropped_img_path + os.sep + image.name, training_set)
            count += 1
    # removing redundant cropped_img directory
    os.rmdir(cropped_img_path)
