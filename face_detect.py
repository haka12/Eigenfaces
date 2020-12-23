import cv2
import os


class FaceDetect:

    def __init__(self, path):
        self.path = path
        self.imsize = 100

    def face_return(self):
        if not os.path.isdir('./cropped_image'):
            os.makedirs('./cropped_image')
        for subdirs, _, files in os.walk(self.path):
            for file in files:
                image_path = subdirs + os.sep + file
                if image_path.endswith(".jpg") or image_path.endswith("jpeg") or image_path.endswith("png"):
                    gray_face = self.haar_detection(image_path)
                    if gray_face is not None:
                        gray_face = cv2.resize(gray_face, (self.imsize, self.imsize))
                        cv2.imwrite('cropped_image' + os.sep + file, gray_face)

    def haar_detection(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        face = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        try:
            (x, y, w, h) = face[0]
        except IndexError:
            return None
        return gray[y:y + w, x:x + h]
