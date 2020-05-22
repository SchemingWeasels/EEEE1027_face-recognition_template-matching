import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
SYMBOL_DIR = os.path.join(BASE_DIR, "Symbols3")
SYMBOL_DIR_EXTRA = os.path.join(BASE_DIR, "SymbolExtra")
CHALLENGE_DIR = os.path.join(BASE_DIR, "Challenge")
SYMBOL_PREFX = "Slide"

noise_max = 1000
title_window = "noise"

def on_trackbar(val):
    val = val/100
    gauss = np.random.normal(0, val, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(img, gauss)
    img_speck = img + img * gauss
    cv2.putText(img_gauss, str(val), (0,img_gauss.shape[0]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255),1)
    cv2.putText(img_speck, str(val), (0, img_gauss.shape[0]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 1)
    cv2.imshow(title_window, img_gauss)
    cv2.imshow("speck", img_speck)

trackbar_name = 'sd x %d' % (noise_max/100)

for root, dirs, files in os.walk(CHALLENGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            img = cv2.imread(os.path.join(root, file))
            cv2.imshow("ori", img)

            #Generatr noise mat
            # Generate Gaussian noise
            gauss = np.random.normal(0, 1, img.size)
            gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
            # Add the Gaussian noise to the image
            img_speck = img + img * gauss
            img_gauss = cv2.add(img, gauss)
            # Display the image
            cv2.imshow("noise", img_gauss)
            cv2.imshow("speck", img_speck)

on_trackbar(100)
cv2.createTrackbar(trackbar_name, title_window , 0, noise_max, on_trackbar)

cv2.waitKey()
cv2.destroyAllWindows()