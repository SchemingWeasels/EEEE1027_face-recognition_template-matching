import cv2
import numpy as np
import os
import csv

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

img = cv2.imread("rotation_base.jpg")
cv2.imshow("ori", img)

cv2.imshow("rot", rotate_image(img,0))

for i in range(-46, 46):
  print(str(i))
cv2.waitKey()
cv2.destroyAllWindows()