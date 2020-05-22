import os
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))
import dlib
import face_recognition
import pickle
import cv2
import os

MODE = "hog"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, 'demo.jpg')

#image = cv2.imread(IMG_PATH)

cap = cv2.VideoCapture(0)
encoding_file = open("encodings2.pickle", "rb")
data = pickle.load(encoding_file)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(MODE+'_demo.avi',fourcc, 10.0, (640, 480))

while 1:
    ret, image = cap.read()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    scale_percent = 80  # percent of original size
    width = int(rgb.shape[1] * scale_percent / 100)
    height = int(rgb.shape[0] * scale_percent / 100)
    dim = (width, height)
    #rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)

    boxes = face_recognition.face_locations(rgb, model=MODE, number_of_times_to_upsample=1)
    encodings = face_recognition.face_encodings(rgb, boxes)

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
            for (i,b) in enumerate(matches):
                total_num += 1
                if(b):
                    matched_num += 1
            match_percent = matched_num / total_num * 100
            match_percents.append(match_percent)

        # update the list of names
        else:
            match_percents.append(0)
        names.append(name)

    #finding JON via min threshold && max perc in IUT
    threshold = 98.8
    max_percent = 0
    index_percent = -1
    for (i ,(name, percent)) in enumerate(zip(names,match_percents)):
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

    for top, right, bottom, left in boxes:
        # draw the predicted face name on the image
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

    cv2.imshow('res', img)
    out.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.waitKey()
