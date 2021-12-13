import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_con=0.75):
        self.minDetectionCon = min_detection_con

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_con)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bounding_boxes = []
        if self.results.detections:
            for _id, detection in enumerate(self.results.detections):
                self.mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bounding_box = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bounding_boxes.append([_id, bounding_box, detection.score])
                if draw:
                    img = self.fancyDraw(img, bounding_box)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bounding_box[0], bounding_box[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
        return img, bounding_boxes

    def fancyDraw(self, img, bbox, length=30, thick=5, rect_thick=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        color = (150, 0, 255)

        cv2.rectangle(img, bbox, color, rect_thick)
        # Top Left x,y
        cv2.line(img, (x, y), (x + length, y), color, thick)
        cv2.line(img, (x, y), (x, y + length), color, thick)

        # Top Right x1,y1
        cv2.line(img, (x1, y), (x1 - length, y), color, thick)
        cv2.line(img, (x1, y), (x1, y + length), color, thick)

        # Bottom left x1,y
        cv2.line(img, (x, y1), (x + length, y1), color, thick)
        cv2.line(img, (x, y1), (x, y1 - length), color, thick)

        # Bottom right x1,y1
        cv2.line(img, (x1, y1), (x1 - length, y1), color, thick)
        cv2.line(img, (x1, y1), (x1, y1 - length), color, thick)

        return img


def main():
    cap = cv2.VideoCapture(0)
    previous_time = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bounding_boxes = detector.findFaces(img)
        print(bounding_boxes)
        current_time = time.time()
        fps = int(1 / (current_time - previous_time))
        previous_time = current_time

        cv2.putText(img, f'FPS: {fps}', (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.imshow("Image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
