import math
import dlib
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

import cv2
import numpy as np
from itertools import chain

from deepface import DeepFace

import threading

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
unknown_dir = os.path.join(BASE_DIR, "unknown")
card_dir = os.path.join(BASE_DIR, "card")
face_dir = os.path.join(BASE_DIR, "face")

 
cam = cv2.VideoCapture(2)

cam.set(3,1280)
cnn_face_detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")

isTwoFace = False


HEIGHT = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))


cropedFrame = []
color = (255, 255, 255)

pt1_card = 0+10, 0+150
pt2_card = int(40 * WIDTH // 100)-10, HEIGHT-150

pt1_face = (int(40 * WIDTH // 100)+10, 0+10)
pt2_face = (WIDTH-10, HEIGHT-10)


while True:

    # Getting out image by webcam
    _, image = cam.read()

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    rects = cnn_face_detector(gray, 0)

    key = cv2.waitKey(1)

    cv2.rectangle(image, pt1_card, pt2_card, color, 4)
        # Mask for FACE
    cv2.rectangle(image, pt1_face, pt2_face, color, 4)

    if key == ord('e'):
        cv2.imwrite(os.path.join(card_dir, 'card.jpg'), image[0+150: HEIGHT-150, 0+10: int(40 * WIDTH // 100)-10])

    # For each detected face
    # for (i, rect) in enumerate(rects):
    #     # Finding points for rectangle to draw on face
    #     x1, y1, x2, y2, w, h = rect.rect.left(), rect.rect.top(), rect.rect.right() + \
    #         1, rect.rect.bottom() + 1, rect.rect.width(), rect.rect.height()
 
    #     cv2.rectangle(image, (x1-50, y1-50), (x2+50, y2+50), (0, 255, 0), 2)


    #     if i == 1:
    #         isTwoFace = True

    #         if key == ord('e'):
    #             for _ in range(1):
    #                 if ((y1 > 0 and x1 > 0) and (y2 < HEIGHT and x2 < int(40 * WIDTH // 100))):
    #                     cv2.imwrite(os.path.join(card_dir, 'card.jpg'), image[y1-48: y2+48, x1-48: x2+48])
    #                                 # FACE
    #                 elif ((y1 > 0 and x1 > 40 * WIDTH / 100) and (y2 < HEIGHT and x2 < WIDTH)):
    #                     cv2.imwrite(os.path.join(face_dir, 'face.jpg'), image[y1-48: y2+48, x1-48: x2+48])

        # if os.path.isfile("./face/face.jpg") and os.path.isfile("./card/card.jpg"):       
        #     print("Starting Compare Face...")
        #     print(compareFace("./face/face.jpg", "./card/card.jpg"))



    cv2.imshow('my webcam', image)

    if key == ord('q'):
        break
    

cv2.destroyAllWindows()