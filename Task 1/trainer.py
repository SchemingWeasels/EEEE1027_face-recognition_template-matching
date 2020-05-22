import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "cropped")

dirs = ["cropped", "spam_data3"]#, "spam_data2"]

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.setNeighbors(10)
#recognizer.setRadius(2)
#recognizer.setGridX(10)
#recognizer.setGridY(10)

count = 0
y_labels = []
x_train = []

for dir in dirs:
    image_dir = os.path.join(BASE_DIR, dir)
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                path = os.path.join(root, file)
                #label = os.path.basename(root).replace(" ", "-").lower()
                # print(label, path)
                #if not label in label_ids:
                #    label_ids[label] = current_id
                #    current_id += 1
                #id_ = label_ids[label]
                # print(label_ids)
                # x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
                pil_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # grayscale

                size = (550, 550)
                final_image = cv2.resize(pil_image,size)
                #cv2.imshow(file, final_image)
                #print(image_array)
                faces = face_cascade.detectMultiScale(pil_image, scaleFactor=1.09)

                for (x, y, w, h) in faces:
                    roi = pil_image[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(0) # some number
                else:
                    x_train.append(pil_image)
                    y_labels.append(0) # some number
                count += 1

# print(y_labels)
# print(x_train)

#with open("pickles/face-labels.pickle", 'wb') as f:
#    pickle.dump(label_ids, f)

output_file = os.path.join(BASE_DIR, "recognizers/face-trainner.yml")

recognizer.train(x_train, np.array(y_labels))
recognizer.save('face-trainner_nohair_nosize.yml')

print("trained "+str(count))

cv2.waitKey()
cv2.destroyAllWindows()