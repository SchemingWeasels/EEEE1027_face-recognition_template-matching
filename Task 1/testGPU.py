import cv2
import cv2gpu
import numpy as np

capture = cv2.VideoCapture(0)


def tryFaces(frame, type):
    cv2gpu.init_gpu_detector('cv2\data\haarcascades_cuda\haarcascade_frontalface_alt2.xml')
    framCpy = np.copy(frame)
    faces = cv2gpu.find_faces(image_file)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        framCpy = cv2.ellipse(framCpy, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        # -- In each face, detect eyes
    cv2.imshow(str(type), framCpy)
    framCpy = np.zeros((2,3))


# img = cv2.imread('IMG_7532.JPG')

# scale = 10
# img = cv2.resize(img, (int(img.shape[1] * scale / 100), int(img.shape[0] * scale / 100)))

# edges = cv2.Canny(img, 100, 200)

# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while (True):
    # Capture frame-by-frame
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    for numnumnum in range(4):
        tryFaces(frame, numnumnum)

    # frame = cv2.rectangle(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)), (int(frame.shape[1]*3/4), int(frame.shape[0]*3/4)), (0,0,0), 5)
    frame = cv2.circle(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), 50, (0, 0, 255), -1)

    cv2.putText(frame, 'BLABLABLA', (0, int(frame.shape[0] * 9 / 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2,
                cv2.LINE_AA)

    # Display the resulting frame

    cv2.imshow('Ori', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imshow("r", range)
cv2.waitKey()
cv2.destroyAllWindows()
