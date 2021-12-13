import cv2
import time
import numpy as np
import HandTrackingModule as Htm
import math
import webbrowser

##############################
wCam, hCam = 640, 480
##############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

previous_time = 0

detector = Htm.handDetector(detection_con=0.7)
check = False
while True:
    success, img = cap.read()
    detector.findHands(img)
    landmark_list = detector.findPosition(img, draw=False)
    if not len(landmark_list) == 0:
        # print(landmark_list[4], landmark_list[8])

        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        print(length)
        if length > 200:
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            check = True
        if check and length < 50:
            check = False
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            webbrowser.open("https://www.youtube.com/watch?v=rcHK5o4t9js")

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
