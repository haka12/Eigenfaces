import os
import shutil

import numpy as np

from constants import *


# Creates two different set of folder with training set and testing set randomly.
def split():
    no_of_img = len(os.listdir(cropped_img_path))
    # 20 random data for test set.
    rand = np.random.choice(no_of_img, 20)
    if not os.path.isdir(training_set) and not os.path.isdir(testing_set):
        os.makedirs(training_set)
        os.makedirs(testing_set)
    with os.scandir(cropped_img_path) as images:
        count = 0
        for image in images:
            if count in rand:
                shutil.move(cropped_img_path + os.sep + image.name, testing_set)
            else:
                shutil.move(cropped_img_path + os.sep + image.name, training_set)
            count += 1
    os.rmdir(cropped_img_path)
