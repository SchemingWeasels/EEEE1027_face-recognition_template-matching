import cv2
import os
import numpy as np
import cv2.cuda

dirName = 'no_specs'
COLOR = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(BASE_DIR, dirName)

# Create target Directory if don't exist
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

count = 0

capture = cv2.VideoCapture(0)

def save_faces(frame):
    global count
    cv2.imwrite(os.path.join(save_dir,str(count)+'.jpg'),frame)
    count += 1

def tryFaces(frame, type, scale):
    framCpy = np.copy(frame)
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

    frame_gray = cv2.cvtColor(framCpy, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=scale)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)

        if COLOR:
            roi = framCpy[y:y + h, x:x + w]
        else:
            roi = frame_gray[y:y + h, x:x + w]
        save_faces(roi)

        framCpy = cv2.ellipse(framCpy, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)

        cv2.imshow("testroi", roi)
    cv2.imshow(str(type), framCpy)


while (True):
    # Capture frame-by-frame
    ret, frame = capture.read()
    frame = cv2.flip(frame, 2)

    #    for numnumnum in range(1,3):
    #        tryFaces(frame, numnumnum, 1.09)
    tryFaces(frame, 2, 1.09)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey()
cv2.destroyAllWindows()
