import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

previous_time = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.9)
while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for _id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            # print(_id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bounding_box = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bounding_box, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bounding_box[0], bounding_box[1]-20),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)

    current_time = time.time()
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time

    cv2.putText(img, f'FPS: {fps}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Image", img)

    cv2.waitKey(1)
