import cv2
import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import time
from math import sqrt
if len(sys.argv) != 5:
    print(
        "Call this program like this:\n"
        "   ./comparation.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat path2face1 path2face2 path2result\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()


def euclidean_dist(vector_x, vector_y):
    if len(vector_x) != len(vector_y):
        raise Exception('Vectors must be same dimensions')
    return sum((vector_x[dim] - vector_y[dim]) ** 2 for dim in range(len(vector_x)))

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
faces1_folder_path = sys.argv[3]
faces2_folder_path = sys.argv[4]

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
def compare():
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    img1 = cv2.imread(faces1_folder_path)
    print(img1.size, img1.shape)
    tick1 = time.time()
    shape1 = sp(img1, dlib.rectangle(0,0,img1.shape[0], img1.shape[1]))
    face_descriptor1 = facerec.compute_face_descriptor(img1, shape1)
    print("descriptor1 shape = {}, time = {}".format( face_descriptor1.shape, time.time() - tick1))

    img2 = cv2.imread(faces2_folder_path)
    print(img2.size, img2.shape)
    tick1 = time.time()
    shape2 = sp(img2,dlib.rectangle(0,0,img2.shape[0], img2.shape[1]))
    face_descriptor2 = facerec.compute_face_descriptor(img2, shape2)
    print("descriptor2 shape = {}, time = {}".format( face_descriptor2.shape, time.time() - tick1))

    tick1 = time.time()
    val = np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))
    #val = 1 - euclidean_dist(face_descriptor1, face_descriptor2)
    print("value = {}, time = {}".format(1 - val, time.time() - tick1))
    hmerge = np.hstack((img1, img2)) #水平拼接
    #vmerge = np.vstack((img1, img2)) #垂直拼接
    cv2.putText(hmerge, "confident = " + str(1-val), (10,10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("result", hmerge)
    #cv2.imshow("test2", vmerge)
    # cv2.imwrite(sys.argv[5] ,hmerge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 1-val

