import sys
import dlib
import cv2
import time
from PIL import Image

detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(2)
img_counter = 0
cropedFrame = []
font = cv2.FONT_HERSHEY_SIMPLEX
dets = []
while True:
    ret_val, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    for det in dets:
        x = det.left()
        y = det.top()
        w = det.right()
        h = det.bottom()
        cv2.rectangle(img,(x-20, y-20), (w+20, h+20), (0,255,0), 2)
        cv2.putText(img, "Mansea", (x+20, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

    cv2.imshow('my webcam', img)
    k = cv2.waitKey(1)
    cv2.putText(img,'Press Q to Start',(5,470), font,0.5,(255,255,255),1,cv2.LINE_AA)
    timer = 39

    if k == ord('q'):
        while timer>=8:
            ret, img = cam.read()
            
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets = detector(rgb_image)
            for det in dets:
                x = det.left()
                y = det.top()
                w = det.right()
                h = det.bottom()
                cv2.rectangle(img,(x-20, y-20), (w+20, h+20), (0,255,0), 2)
                cv2.putText(img, "Mansea", (x+20, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                
            print(dets)
            if dets == []:
                break
            cv2.putText(img,str(timer//10),(250,250), font, 7,(255,255,255),4,cv2.LINE_AA)
            cv2.imshow('my webcam',img)
            cv2.waitKey(80)
            timer-=1
        try:
            roi = img[y-80: h+50, x-50: w+50][:,:,::-1]
            croped = Image.fromarray(roi)
            cropedFrame.append(croped)
        except:
            pass
        for i , img in enumerate(cropedFrame):
            img.save("test" + str(i) + ".jpg")

    elif k == 27:
        break

    

    
cv2.destroyAllWindows()