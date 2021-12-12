import cv2
import mediapipe as mp
import time
import HandTrackingModule as Htm

previous_time = 0
current_time = 0
cap = cv2.VideoCapture(0)
detector = Htm.handDetector()
while True:
    try:
        success, img = cap.read()
    except KeyboardInterrupt:
        pass

    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    if not len(lmlist) == 0:
        print(lmlist[0])
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)