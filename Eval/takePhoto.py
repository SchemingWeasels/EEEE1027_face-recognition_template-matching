import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'pics')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

#Get next file number
file_num = 0
for root, dirs, files in os.walk(SAVE_DIR):
    for file in files:
        file = file.lower()
        file_name, file_ext = os.path.splitext(file)
        if file_name > file_num and file_ext == '.jpg':
            file_num = file_name

cap = cv2.VideoCapture(0)

while(1):
    ret, frame = cap.read()
    display = np.copy(frame)
    cv2.putText(display, "Q to quit ~ S to save", (0, int(display.shape[0] * 9 / 10)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Display", display)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        file_num += 1
        cv2.imwrite(os.path.join(SAVE_DIR, str(file_num) + ".jpg"), frame)

cap.release()
cv2.destroyAllWindows()