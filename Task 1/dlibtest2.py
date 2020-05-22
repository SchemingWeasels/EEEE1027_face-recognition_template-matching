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

#dirs = ["cropped", "spam_data"]#, "spam_data2"]
#dirs = ["testdetect"]
challenge_dir = ["challenges"]

# load the known faces and embeddings
print("[INFO] loading encodings...")
encoding_file = open("encodings.pickle", "rb")
data = pickle.load(encoding_file)

knownEncodings = []
knownNames = []

for dir in challenge_dir:
    image_dir = os.path.join(BASE_DIR, dir)
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                path = os.path.join(root, file)
                # load the input image and convert it from BGR (OpenCV ordering)
                # to dlib ordering (RGB)
                image = cv2.imread(path)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


                scale_percent = 32  # percent of original size
                width = int(rgb.shape[1] * scale_percent / 100)
                height = int(rgb.shape[0] * scale_percent / 100)
                dim = (width, height)
                rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)


                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input image
                #time_start = time.perf_counter()
                boxes = face_recognition.face_locations(rgb, model=MODE, number_of_times_to_upsample=1)
                #time_end = time.perf_counter()

                """
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                for (y1,x1,y2,x2) in boxes:
                    roi = image[y1:y2,x1:x2]
                    img = cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),thickness=2)
                time_taken = time_end - time_start

                cv2.putText(img, str(time_taken)+"s", (0,int(height*0.98)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), thickness=2)
                cv2.imshow(str(file), img)

                output_dir = os.path.join(root,MODE+" CUDA")
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                cv2.imwrite(os.path.join(output_dir,file), img)
                """

                # compute the facial embedding for the face
                encodings = face_recognition.face_encodings(rgb, boxes)
                # loop over the encodings
                names = []
                match_percents = []
                # loop over the facial embeddings
                for encoding in encodings:
                    # attempt to match each face in the input image to our known
                    # encodings
                    distance = face_recognition.api.face_distance(data["encodings"], encoding)
                    print(distance)
                    name = "Unknown"

                    """# check to see if we have found a match
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
                        threshold = 96.8
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

                cv2.imshow(file, img)
                output_dir = os.path.join(root, MODE + " filtered CUDA")
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                cv2.imwrite(os.path.join(output_dir, file), img)"""
        break  # only do top dir

cv2.waitKey()
cv2.destroyAllWindows()
