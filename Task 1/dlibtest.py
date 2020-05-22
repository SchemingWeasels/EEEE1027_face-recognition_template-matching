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

challenge_dir = ['challenges\Dataset\\boundry']

# load the known faces and embeddings
print("[INFO] loading encodings...")
encoding_file = open("encodings2.pickle", "rb")
data = pickle.load(encoding_file)

for dir in challenge_dir:
    image_dir = os.path.join(BASE_DIR, dir)
    for root, dirs, files in os.walk(image_dir):
        if "result" in root:
            continue
        for file in files:
            file = file.lower()
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                frame_time_start = []
                frame_time_end = []
                frame_time = 0
                path = os.path.join(root, file)
                # load the input image and convert it from BGR (OpenCV ordering)
                # to dlib ordering (RGB)
                image = cv2.imread(path)
                print("processing " + file)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


                scale_percent = 80  # percent of original size
                width = int(rgb.shape[1] * scale_percent / 100)
                height = int(rgb.shape[0] * scale_percent / 100)
                dim = (width, height)
                rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)


                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input image
                frame_time_start.append(time.perf_counter())
                boxes = face_recognition.face_locations(rgb, model=MODE, number_of_times_to_upsample=1)
                frame_time_end.append(time.perf_counter())

                # compute the facial embedding for the face
                encodings = face_recognition.face_encodings(rgb, boxes)
                # loop over the encodings
                names = []
                match_percents = []
                # loop over the facial embeddings
                for encoding in encodings:
                    # attempt to match each face in the input image to our known
                    # encodings
                    frame_time_start.append(time.perf_counter())
                    matches = face_recognition.compare_faces(data["encodings"], encoding)
                    frame_time_end.append(time.perf_counter())
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

                # Performace Calculation
                for start, end in zip(frame_time_start, frame_time_end):
                    frame_time += end - start
                cv2.putText(img, str(frame_time), (0, int(img.shape[0] * 9 / 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 0, 255), 2,
                            cv2.LINE_AA)

                #cv2.imshow(file, img)
                output_dir = os.path.join(root, "result " + MODE)
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                #cv2.imwrite(os.path.join(output_dir, file), img)

cv2.waitKey()
cv2.destroyAllWindows()