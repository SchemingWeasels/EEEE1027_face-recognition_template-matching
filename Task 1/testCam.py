import cv2
import os
import numpy as np

import time
import csv

capture = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face-trainner_nohair_nosize.yml')

def tryFaces(frame, type, scale):
    frame_time_start = []
    frame_time_end = []
    frame_time = 0
    framCpy = np.copy(frame)
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
    frame_gray = cv2.cvtColor(framCpy, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.equalizeHist(frame_gray)

    # -- Detect faces
    frame_time_start.append(time.perf_counter())
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=scale)
    frame_time_end.append(time.perf_counter())
    best_dist = 1000
    best_match = (0,0,0,0)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        framCpy = cv2.ellipse(framCpy, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        roi = frame_gray[y:y + h, x:x + w]
        frame_time_start.append(time.perf_counter())
        _id, clvl = recognizer.predict(roi)
        frame_time_end.append(time.perf_counter())
        if clvl<best_dist and clvl>1:
            best_dist = clvl
            best_match = (x,y,w,h)
        #if clvl < 100:
        cv2.putText(framCpy, f"{clvl:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2,
                        cv2.LINE_AA)

    if best_match:
        x, y, w, h = best_match
        framCpy = cv2.rectangle(framCpy, (x,y), (x+w,y+h), (255,255,0), thickness=10)
        cv2.putText(framCpy, "Jon", (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5,
                                    cv2.LINE_AA)

    #Performace Calculation
    for start,end in zip(frame_time_start, frame_time_end):
        frame_time += end - start
    cv2.putText(framCpy, str(frame_time), (0, int(frame.shape[0] * 9 / 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2,
                cv2.LINE_AA)

    if(framCpy.shape[0] > 1000 or framCpy.shape[1] > 1000):
        scale_percent = 30  # percent of original size
        width = int(framCpy.shape[1] * scale_percent / 100)
        height = int(framCpy.shape[0] * scale_percent / 100)
        dim = (width, height)
        framCpy = cv2.resize(framCpy, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(str(type), framCpy)
    return framCpy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
challenge_dir = os.path.join(BASE_DIR, "challenges\Dataset")

for root, dirs, files in os.walk(challenge_dir):
    if "result" in root:
        continue
    for file in files:
        file = file.lower()
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            challenge = cv2.imread(os.path.join(root, file))
            output_dir = os.path.join(root, "result LBPH")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            cv2.imwrite(os.path.join(output_dir, file), tryFaces(challenge, file, 1.09))


cv2.waitKey()
cv2.destroyAllWindows()
