import cv2
import time
 
# Open the camera
cap = cv2.VideoCapture(0)
 
while True:
    # Read and display each frame
    ret, img = cap.read()
    cv2.imshow('a',img)
    k = cv2.waitKey(125)
    # Specify the countdown
    timer = 55
    # set the key for the countdown to begin
    if k == ord('q'):
        while timer>=10:
            ret, img = cap.read()

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,str(timer//10),(250,250), font, 7,(255,255,255),10,cv2.LINE_AA)
            cv2.imshow('a',img)
            cv2.waitKey(80)
            timer = timer-1
    # Press Esc to exit
    elif k == 27:
        break
cap.release()
cv2.destroyAllWindows()