import os
import dlib
import cv2

name = input("Enter your name: ")

path = "./images/" + name
num_of_images = 1

try:
    os.makedirs(path)
except:
    print('Directory Already Created')

count = 0

cam = cv2.VideoCapture(2)
cnn_face_detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")

while True:

    # Getting out image by webcam
    _, image = cam.read()

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    rects = cnn_face_detector(gray, 0)

    key = cv2.waitKey(1)

    # For each detected face
    for (i, rect) in enumerate(rects):
        # Finding points for rectangle to draw on face
        x1, y1, x2, y2, w, h = rect.rect.left(), rect.rect.top(), rect.rect.right() + \
            1, rect.rect.bottom() + 1, rect.rect.width(), rect.rect.height()

 
        cv2.rectangle(image, (x1-50, y1-50), (x2+50, y2+50), (0, 255, 0), 2)


        # show the face number
        cv2.putText(image, name, (x1 - 50, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 51, 255), 2)

        cv2.putText(image, str(str(num_of_images)+" images captured"), (x1, y1+h+70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 51, 255), 2)

        new_img = image[y1-48: y2+48, x1-48: x2+48]

    if count % 2 == 0:
        try :
            cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
            num_of_images += 1
        except :
            pass
    
    if key == ord("q") or key == 27 or num_of_images > 50:
        break
    cv2.imshow('my webcam', image)

    if cv2.waitKey(1) == ord('q'):
        break
    count+=1

cv2.destroyAllWindows()
