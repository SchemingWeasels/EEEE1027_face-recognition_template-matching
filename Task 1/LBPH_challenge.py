import cv2
import os
import numpy as np
import cv2.cuda

capture = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face-trainner_nohair_nosize.yml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
challenge_dir = os.path.join(BASE_DIR, "challenges")

for root, dirs, files in os.walk(challenge_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(path)

            face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # frame_gray = cv2.equalizeHist(frame_gray)

            # -- Detect faces
            faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.09)
            best_dist = 1000
            best_match = (0, 0, 0, 0)
            for (x, y, w, h) in faces:
                center = (x + w // 2, y + h // 2)
                image = cv2.ellipse(image, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), thickness=5)
                roi = frame_gray[y:y + h, x:x + w]
                _id, clvl = recognizer.predict(roi)
                if clvl < best_dist and clvl > 1:
                    best_dist = clvl
                    best_match = (x, y, w, h)

                cv2.putText(image, f"{clvl:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                            cv2.LINE_AA)

            x, y, w, h = best_match
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), thickness=12)
            # cv2.putText(framCpy, "Jon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5,
            #                            cv2.LINE_AA)

            scale_percent = 30  # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow(file, image)
            output_dir = os.path.join(root, "LBPH")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            cv2.imwrite(os.path.join(output_dir, file), image)


    break #root only

cv2.waitKey()
cv2.destroyAllWindows()
