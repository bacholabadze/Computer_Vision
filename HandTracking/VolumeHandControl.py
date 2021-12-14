import cv2
import time
import numpy as np
import HandTrackingModule as Htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##############################
wCam, hCam = 640, 480
##############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

previous_time = 0

detector = Htm.handDetector(detection_con=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()

# (-63.5, 0.0, 0.5)
volume_range = volume.GetVolumeRange()

min_vol = volume_range[0]
max_vol = volume_range[1]

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

        # Hand Size from wrist to thumb
        # x3, y3 = landmark_list[0][1], landmark_list[0][2]
        # cv2.line(img, (x1, y1), (x3, y3), (0, 255, 0), 2)
        # hand_length = math.hypot(x3 - x1, y3 - y1)
        # # Hand length 80 - 300
        # # print(hand_length)
        # hand_experiment = np.interp(hand_length, [100, 200], [80, 300])
        # # print(hand_experiment)
        # min_realtime_length = hand_experiment
        # # Hand Range 20 - 200

        length = math.hypot(x2 - x1, y2 - y1)
        print(length)

        vol = np.interp(length, [40, 200], [min_vol, max_vol])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 40:
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
