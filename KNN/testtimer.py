
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
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