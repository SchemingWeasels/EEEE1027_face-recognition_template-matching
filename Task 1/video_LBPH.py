import os
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

BASE_DIR = os.path.dirname(__file__)
IMG_PATH = os.path.join(BASE_DIR, 'demo.jpg')

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face-trainner_nohair_nosize.yml')


#img = cv2.imread(IMG_PATH)
#cv2.imshow('test', img)

while 1:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray)

    best_dist = -1
    best_match = (-1, -1, -1, -1)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)

        roi = frame_gray[y:y + h, x:x + w] 
        _id, clvl = recognizer.predict(roi)
        if best_dist == -1 or (clvl < best_dist and clvl > 1):
            best_dist = clvl
            best_match = (x, y, w, h)

        cv2.putText(frame, f"{clvl:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2,
                    cv2.LINE_AA)

        if not best_match == (-1, -1, -1, -1):
            x, y, w, h = best_match
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), thickness=10)
            cv2.putText(frame, "Jon", (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5,
                        cv2.LINE_AA)

        cv2.imshow('capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
