import cv2
import os
from constants import *
import math


class FaceDetect:

    def __init__(self):
        self.path = data_path
        self.imsize = int(math.sqrt(img_size))

    def face_return(self):
        if not os.path.isdir(cropped_img_path):
            os.makedirs(cropped_img_path)
        for subdirs, _, files in os.walk(self.path):
            # count is used to differentiate the names of same label
            count = 0
            for file in files:
                image_path = subdirs + os.sep + file
                if image_path.endswith(".jpg") or image_path.endswith("jpeg") or image_path.endswith("png"):
                    gray_face = self.haar_detection(image_path)
                    if gray_face is not None:
                        gray_face = cv2.resize(gray_face, (self.imsize, self.imsize))
                        # file name is taken from the folder name of the data subdirectories
                        file_name = subdirs[7:] + str(count) + '.jpeg'
                        cv2.imwrite(cropped_img_path + os.sep + file_name, gray_face)
                        count += 1

    @staticmethod
    def haar_detection(image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(haar_path)
        face = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        try:
            (x, y, w, h) = face[0]
        except IndexError:
            return None
        return gray[y:y + w, x:x + h]
