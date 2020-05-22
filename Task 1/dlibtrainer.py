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

dirs = ["cropped", "spam_data3"]#, "spam_data2"]
#dirs = ["testdetect"]

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")

knownEncodings = []
knownNames = []

for dir in dirs:
    image_dir = os.path.join(BASE_DIR, dir)
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                path = os.path.join(root, file)
                print("processing img " + file)
                name = "jon"
                # load the input image and convert it from BGR (OpenCV ordering)
                # to dlib ordering (RGB)
                image = cv2.imread(path)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                """
                scale_percent = 30  # percent of original size
                width = int(rgb.shape[1] * scale_percent / 100)
                height = int(rgb.shape[0] * scale_percent / 100)
                dim = (width, height)
                rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
                """

                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input image
                #time_start = time.perf_counter()
                boxes = face_recognition.face_locations(rgb, model=MODE, number_of_times_to_upsample=0)
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
                #encodings = face_recognition.face_encodings(rgb, boxes)
                encodings = face_recognition.face_encodings(rgb, boxes)
                # loop over the encodings
                for encoding in encodings:
                    # add each encoding + name to our set of known names and
                    # encodings
                    knownEncodings.append(encoding)
                    knownNames.append(name)
        #break  # only do top dir

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings2.pickle", "wb")
f.write(pickle.dumps(data))
f.close()

"""
cv2.waitKey()
cv2.destroyAllWindows()
"""