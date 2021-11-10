"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.
When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.
Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under euclidean distance)
in its training set, and performing a majority vote (possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.
* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.
Usage:
1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.
2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.
3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.
NOTE: This example requires scikit-learn to be installed! You can install it with pip:
$ pip3 install scikit-learn
"""

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



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
unknown_dir = os.path.join(BASE_DIR, "unknown")

#"recognizers/face-trainner.yml"

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.48):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
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

def start():
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.

    cap = cv2.VideoCapture(0)

    cap.set(3,2560)
    # 1920 full HD
    # 2560 2K


    cropedFrame = []
    color = (255, 255, 255)

    while(True):
        ret, frame = cap.read()
        overlay = cv2.imread('maskOverlay.png', cv2.IMREAD_UNCHANGED)
        # cv2.imshow("overlay", overlay)
        # frame = cv2.addWeighted(frame,0.5,overlay,0.5,0)
        
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


        cv2.imwrite(os.path.join(unknown_dir , 'unknown.jpg'), frame)

    # STEP 2: Using the trained classifier, make predictions for unknown images
        for image_file in os.listdir(unknown_dir):
            full_file_path = os.path.join(unknown_dir, image_file)

            print("Looking for faces in {}".format(image_file))

            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = predict(full_file_path, model_path="trained_knn_model.clf")

            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))
                font = cv2.FONT_HERSHEY_SIMPLEX               
                stroke = 2


                cropedFrame.append(name)

                try:
                    if (cropedFrame[0] == cropedFrame[1]):
                        color = (0, 255, 0)
                    else:
                        color = (0,0,255)
                except:
                    pass

                cv2.putText(frame,'Press Q to Quit',(5,470), font,0.5,(255,255,255),1,cv2.LINE_AA)

            for name, (top, right, bottom, left) in predictions:
                cv2.rectangle(frame, (left, top), (right, bottom), color, 4)
                cv2.putText(frame, name, (left, top), font, 1, color, stroke, cv2.LINE_AA)

            color = (255, 255, 255)
            cropedFrame = []

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
    cv2.destroyAllWindows()