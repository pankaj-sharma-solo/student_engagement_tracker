import cv2
import numpy as np


class Extractor:
    def __init__(self, cascadeClassifierPath: str = None, videoPath: str = None):
        self.faceCascade = None
        self.videoCapture = None

    def __enter__(self):
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.videoCapture = cv2.VideoCapture(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.videoCapture.release()
        cv2.destroyAllWindows()

    def facialFeatureExtractor(self):
        while True:
            _, frame = self.videoCapture.read()
            self.detectFace(frame)
            self.detectEyes(frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def detectEyes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.eyeCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        left_right_eye = self.getLefAndRightEye(frame, eyes)
        for eye in left_right_eye:
            if eye is not None:
                eye = self.cutEyebrows(eye)
                keypoints = self.detectBlob(eye)
                eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # for (x, y, w, h) in eyes:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return eyes

    def detectFace(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        face_frame = None
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_frame = frame[y:y + h, x:x + w]
        return face_frame

    @staticmethod
    def detectBlob(frame):
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 1500
        detector = cv2.SimpleBlobDetector_create(detector_params)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray_frame, 42, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, None, iterations=2)
        img = cv2.dilate(img, None, iterations=4)
        img = cv2.medianBlur(img, 5)
        # cv2.imshow('Threshold', img)
        keypoints = detector.detect(img)
        return keypoints

    @staticmethod
    def getLefAndRightEye(frame, eyes):
        width = np.size(frame, 1)
        height = np.size(frame, 0)
        left_eye, right_eye = None, None
        for (x, y, w, h) in eyes:
            if x + w / 2 < width / 2:
                left_eye = frame[y:y + h, x:x + w]
            else:
                right_eye = frame[y:y + h, x:x + w]
        return left_eye, right_eye

    @staticmethod
    def cutEyebrows(eye):
        height, width = eye.shape[:2]
        eb_height = int(height / 4)
        eye = eye[eb_height:height, 0:width]
        return eye


# if __name__ == '__main__':
#     with Extractor() as fe:
#         fe.facialFeatureExtractor()
