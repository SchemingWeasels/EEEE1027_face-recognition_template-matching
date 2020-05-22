import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
SYMBOL_DIR = os.path.join(BASE_DIR, "SymbolsNew")
CHALLENGE_DIR = os.path.join(BASE_DIR, "Challenge")
SYMBOL_PREFX = "Slide"

img_t = cv2.imread(os.path.join(SYMBOL_DIR, "SlideB.PNG"))
img_c = cv2.imread(os.path.join(CHALLENGE_DIR, "1.jpeg"))

count = 0

def show_result():
    global count
    img_result = cv2.drawMatches(img_t, kp_t, img_c, kp_c, matches, None)

    cv2.putText(img_result, "Num of matches:" + str(len(matches)), (0, int(img_result.shape[0] * 0.98)),
                cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)

    cv2.imshow(str(count), img_result)
    count += 1

def show_result_k():
    global count
    img_result = cv2.drawMatchesKnn(img_t, kp_t, img_c, kp_c, good, None)

    cv2.putText(img_result, "Num of matches:" + str(len(good)), (0, int(img_result.shape[0] * 0.98)),
                cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)

    cv2.imshow(str(count), img_result)
    count += 1

sift = cv2.xfeatures2d_SIFT_create()
surf = cv2.xfeatures2d_SURF.create()
orb = cv2.ORB_create()

bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

kp_t, des_t = orb.detectAndCompute(img_t, None)
kp_c, des_c = orb.detectAndCompute(img_c, None)

img_t = cv2.drawKeypoints(img_t, kp_t, None)
img_c = cv2.drawKeypoints(img_c, kp_c, None)

matches = bf.match(des_t, des_c)
matches.sort(key=lambda x: x.distance)

show_result()

bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)

kp_t, des_t = sift.detectAndCompute(img_t, None)
kp_c, des_c = sift.detectAndCompute(img_c, None)

img_t = cv2.drawKeypoints(img_t, kp_t, None)
img_c = cv2.drawKeypoints(img_c, kp_c, None)

matches = bf.knnMatch(des_t, des_c, k=2)
#matches.sort(key=lambda x: x.distance)

# Apply ratio test
good = []
"""for m,n,o in matches:
    if m.distance < 0.65*n.distance:
        if m.distance < 0.65*o.distance:
            good.append([m])
"""
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
show_result_k()

bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

kp_t, des_t = surf.detectAndCompute(img_t, None)
kp_c, des_c = surf.detectAndCompute(img_c, None)

img_t = cv2.drawKeypoints(img_t, kp_t, None)
img_c = cv2.drawKeypoints(img_c, kp_c, None)

matches = bf.match(des_t, des_c)
matches.sort(key=lambda x: x.distance)

show_result()

"""for root, dirs, files in os.walk(SYMBOL_DIR):
    for file in files:
        cv2.imshow(file, cv2.imread(os.path.join(root, file)))
"""

cv2.waitKey(0)
cv2.destroyAllWindows()

# template = cv2.imread()
