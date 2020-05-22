# import the necessary packages
import os
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))
import dlib
import face_recognition
import pickle
import cv2
import os
import time

MODE = "cnn"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load the known faces and embeddings
print("[INFO] loading encodings...")
encoding_file = open("encodings2.pickle", "rb")
data = pickle.load(encoding_file)

knownEncodings = []
knownNames = []

cap = cv2.VideoCapture(0)

while(1):

    ret, frame = cap.read()
    frame = cv2.flip(frame, 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    #Image Sclaing
    scale_percent = 100  # percent of original size
    width = int(rgb.shape[1] * scale_percent / 100)
    height = int(rgb.shape[0] * scale_percent / 100)
    dim = (width, height)
    rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)


    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    #time_start = time.perf_counter()
    boxes = face_recognition.face_locations(rgb, model=MODE, number_of_times_to_upsample=1)
    #time_end = time.perf_counter()


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
            for (i,b) in enumerate(matches):
                total_num += 1
                if(b):
                    matched_num += 1
            threshold = 98.8
            match_percent = matched_num / total_num * 100
            match_percents.append(match_percent)
            if(match_percent > threshold):
                name = "JON"

        # update the list of names
        else:
            match_percents.append(0)
        names.append(name)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for ((top, right, bottom, left), name, percent) in zip(boxes, names, match_percents):
        if(name == "JON"):
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

    cv2.imshow("output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        #cv2.imwrite(__file__+" output.jpg",img)
        break


cv2.waitKey()
cv2.destroyAllWindows()