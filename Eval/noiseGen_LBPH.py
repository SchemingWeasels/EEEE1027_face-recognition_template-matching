import cv2
import os
import numpy as np
import cv2.cuda
import time
import csv

CSV_MODE = 'a'
noise_sd = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'noise')
ENCODING_DIR = os.path.join(BASE_DIR, '..', 'Task 1')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(ENCODING_DIR,'face-trainner_nohair_nosize.yml'))

def tryFaces(frame, sd, scale, file_name):
    framCpy = np.copy(frame)
    face_cascade = cv2.CascadeClassifier(os.path.join(ENCODING_DIR,'cascades/haarcascade_frontalface_alt2.xml'))
    frame_gray = cv2.cvtColor(framCpy, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.equalizeHist(frame_gray)

    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=scale)
    best_dist = 1000
    best_match = (-1,0,0,0)
    count = 0
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        framCpy = cv2.ellipse(framCpy, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 2)
        roi = frame_gray[y:y + h, x:x + w]
        _id, clvl = recognizer.predict(roi)
        if clvl > 0:
            count += 1
        if clvl<best_dist and clvl>1:

            best_dist = clvl
            best_match = (x,y,w,h)
        #if clvl < 100:
        cv2.putText(framCpy, f"{clvl:.2f}", (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
        #-- In each face, detect eyes
        #cv2.imshow("testroi", roi)

    writer.writerow([file_name, sd, len(faces), count])

    if not best_match == (-1,0,0,0):
        x, y, w, h = best_match
        framCpy = cv2.rectangle(framCpy, (x,y), (x+w,y+h), (255,255,0), thickness=10)
        cv2.putText(framCpy, "Jon", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)


    if(framCpy.shape[0] > 1000 or framCpy.shape[1] > 1000):
        scale_percent = 30  # percent of original size
        width = int(framCpy.shape[1] * scale_percent / 100)
        height = int(framCpy.shape[0] * scale_percent / 100)
        dim = (width, height)
        framCpy = cv2.resize(framCpy, dim, interpolation=cv2.INTER_AREA)

    return framCpy

def gen_noise(img, sd):
    # Generatr noise mat
    # Generate Gaussian noise
    gauss = np.random.normal(0, sd, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(img, gauss)
    return img_gauss


save_file = open(os.path.join(BASE_DIR, "noise_LBPH.csv"), CSV_MODE, newline='')
writer = csv.writer(save_file)

output_dir = os.path.join(BASE_DIR, "noise LBPH")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

if CSV_MODE == 'w':
    writer.writerow(["file", "sd", "detections", "matches"])

for root, dirs, files in os.walk(DATA_DIR):
    if "unused" in root:
        continue
    for file in files:
        file = file.lower()
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            file_name = os.path.splitext(file)[0]
            data_output_dir = os.path.join(output_dir, file_name)
            if not os.path.isdir(data_output_dir):
                os.mkdir(data_output_dir)
            img = cv2.imread(os.path.join(root, file))
            for sd in noise_sd:
                cv2.imwrite(os.path.join(data_output_dir, str(sd) + ".jpg"), tryFaces(gen_noise(img, sd), sd, 1.09, file_name))


print("Done")
cv2.destroyAllWindows()
