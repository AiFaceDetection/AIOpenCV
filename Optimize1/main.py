import math
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

isVerify = False
name = []

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.48):

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]



def deleteCompareFile():
    for file in os.listdir(card_dir):
        full_file_path = os.path.join(card_dir, file)
        os.remove(full_file_path)
    for file in os.listdir(face_dir):
        full_file_path = os.path.join(face_dir, file)
        os.remove(full_file_path)


def encodeImg(tID, word):
    try:
        img1 = cv2.imread("./face/0.jpg")
        img2 = cv2.imread("./card/0.jpg")

        reasult = DeepFace.verify(img1, img2)

        arrayReasult = list(reasult.values())
        MAX_THRESHOLD = 0.5
        # print(arrayReasult[0], arrayReasult[1])
        if (arrayReasult[1] > MAX_THRESHOLD):
            output = False
            isVerify = False
        else:
            output = True
            isVerify = True

        print(output, arrayReasult[1])

    except:
        print("Card is not clear")
        
def faceIdentification(HEIGHT, WIDTH, frame):
    print("hello")
    for image_file in os.listdir(unknown_dir):
        full_file_path = os.path.join(unknown_dir, image_file)

        print("Looking for faces in {}".format(image_file))

        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        for name, (top, right, bottom, left) in predictions:

            if (len(predictions) == 2): # Have 2 face in frame
                color = (0, 255, 0)
                for i in range(1):
                    if ((top > 0 and left > 0) and (bottom < HEIGHT and right < int(40 * WIDTH // 100))):
                        cv2.imwrite(os.path.join(card_dir, str(i) + '.jpg'), frame[0+20:HEIGHT-20, 0+20:int(40 * WIDTH // 100)-20])
                    # FACE
                    elif ((top > 0 and left > 40 * WIDTH / 100) and (bottom < HEIGHT and right < WIDTH)):
                        cv2.imwrite(os.path.join(face_dir, str(i) + '.jpg'), frame[0+20:HEIGHT-20, int(40 * WIDTH // 100)+20: int(40 * WIDTH // 100) + WIDTH - int(40 * WIDTH // 100)-20])

def start():
    cap = cv2.VideoCapture(0)
    cap.set(3,1920)
    # 852 480p
    # 1280 720p
    # 1920 full HD
    # 2560 2K

    
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
        for (x, y, w, h) in faces:
            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)


        cv2.imwrite(os.path.join(unknown_dir , 'unknown.jpg'), frame)
        # print(name)
        # cv2.putText(frame, name, (500, 500), cv2.FONT_HERSHEY_SIMPLEX , 1, color, stroke, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        t1 = threading.Thread(target=faceIdentification, args=(HEIGHT, WIDTH, frame ))
        t1.start()
        t1.join()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def main():
    start()

main()