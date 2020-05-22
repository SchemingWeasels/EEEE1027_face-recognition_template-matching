import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)
SYMBOL_DIR = os.path.join(BASE_DIR, "Symbols3")
CHALLENGE_DIR = os.path.join(BASE_DIR, "Challenge")
SYMBOL_PREFX = "Slide"


img = cv2.imread(os.path.join(SYMBOL_DIR, "shapes.JPG"))

cv2.imshow("ori", img)

h, w = img.shape[:2]
crop_percent = 8
roi = img[int(h*crop_percent/100):int(h*(1-(crop_percent/100))), int(w*crop_percent/100):int(w*(1-(crop_percent/100)))]

cv2.imshow("crop", roi)

cv2.waitKey()
cv2.destroyAllWindows()