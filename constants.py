import os
img_size = 100*100
cropped_img_path = './cropped_images'
data_path = './data'
haar_path = './haarcascade_frontalface_default.xml'
training_set = './training_set'
testing_set = './testing_set'
no_of_eigfaces = 45
labels =[names for names in os.listdir(data_path) ]

