# import the necessary packages
import os
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))
import dlib
import numpy as np
import face_recognition
import pickle
import cv2
import os
import csv
import time

MODE = "hog"
CSV_MODE = 'a'
noise_sd = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'noise')
ENCODING_DIR = os.path.join(BASE_DIR, '..', 'Task 1')

save_file = open(os.path.join(BASE_DIR, "noise_dlib_" + MODE +".csv"), CSV_MODE, newline='')
writer = csv.writer(save_file)


# load the known faces and embeddings
print("[INFO] loading encodings...")
encoding_file = open(os.path.join(ENCODING_DIR,"encodings2.pickle"), "rb")
data = pickle.load(encoding_file)

knownEncodings = []
knownNames = []

if CSV_MODE == 'w':
    writer.writerow(["file", "sd", "detections", "matches"])

def recognize_faces(src, sd, thresh, file_name):
    img = np.copy(src)
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    scale_percent = 30 # percent of original size
    width = int(rgb.shape[1] * scale_percent / 100)
    height = int(rgb.shape[0] * scale_percent / 100)
    dim = (width, height)
    #rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model=MODE, number_of_times_to_upsample=2)

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    # loop over the encodings
    names = []
    match_percents = []
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            total_num = 0
            matched_num = 0
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            for (i, b) in enumerate(matches):
                total_num += 1
                if (b):
                    matched_num += 1
            match_percent = matched_num / total_num * 100
            match_percents.append(match_percent)

        # update the list of names
        else:
            match_percents.append(0)
        names.append(name)

    # finding JON via min threshold && max perc in IUT
    threshold = thresh
    max_percent = 0
    index_percent = -1
    count = 0
    for (i, (name, percent)) in enumerate(zip(names, match_percents)):
        if percent > threshold:
            count += 1
        if (percent > max_percent and percent > threshold):
            max_percent = percent
            index_percent = i
    if not index_percent == -1:
        names[index_percent] = "JON"

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for ((top, right, bottom, left), name, percent) in zip(boxes, names, match_percents):
        if (name == "JON"):
            color_font_name = (0, 255, 0)
            color_box = (0, 0, 255)
            size_font_name = 1
        else:
            color_font_name = (255, 0, 0)
            color_box = (0, 255, 0)
            size_font_name = 0.45
        # draw the predicted face name on the image
        cv2.rectangle(img, (left, top), (right, bottom), color_box, 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(img, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size_font_name, color_font_name, 2)
        cv2.putText(img, f"{percent:.2f}", (left, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 2)

    writer.writerow([file_name, sd, len(encodings), count])

    # cv2.imshow(file, img)
    return img

def gen_noise(img, sd):
    # Generatr noise mat
    # Generate Gaussian noise
    gauss = np.random.normal(0, sd, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(img, gauss)
    return img_gauss

output_dir = os.path.join(BASE_DIR, "noise " + MODE)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

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
                cv2.imwrite(os.path.join(data_output_dir, str(sd) + ".jpg"), recognize_faces(gen_noise(img, sd), sd, 98.8, file_name))


cv2.waitKey()
cv2.destroyAllWindows()