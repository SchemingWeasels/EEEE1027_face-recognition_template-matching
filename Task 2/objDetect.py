import os
import cv2
import numpy as np
from statistics import mean

BASE_DIR = os.path.dirname(__file__)
SYMBOL_DIR = os.path.join(BASE_DIR, "SymbolsNew")
CHALLENGE_DIR = os.path.join(BASE_DIR, "Challenge")
SYMBOL_PREFX = "Slide"

def show_result(img_t, img_c, name, kp_t, kp_c, matches, good_matches):
    src_pts = np.float32([kp_t[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_c[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img_t.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, M)
    dst += (w, 0)  # adding offset

    ret = cv2.minAreaRect(dst_pts)
    box = cv2.boxPoints(ret)
    box = np.int0(box)
    box += (w, 0)

    #Avarge top 10
    top_10 = good_matches[:10]
    avg = mean(d.distance for d in top_10)

    img_result = cv2.drawMatches(img_t, kp_t, img_c, kp_c, good_matches, None)
    cv2.drawContours(img_result, [box], 0, (0, 255, 0), 3)
    cv2.putText(img_result, "matches:" + str(len(matches)) + " Min d:" + str(f"{matches[0].distance:.5f}" + " Ave " + str(avg)),
                (0, int(img_result.shape[0] * 0.98)),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    # Draw bounding box in Red
    img_result = cv2.polylines(img_result, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow(name, img_result)

def detect_img_orb(img_t, img_c, name, num):
    orb = cv2.ORB_create()#WTA_K=3)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    kp_t, des_t = orb.detectAndCompute(img_t, None)
    kp_c, des_c = orb.detectAndCompute(img_c, None)

    img_t = cv2.drawKeypoints(img_t, kp_t, None)
    img_c = cv2.drawKeypoints(img_c, kp_c, None)

    matches = bf.match(des_t, des_c)
    matches.sort(key=lambda x: x.distance)
    good_matches = matches[:num]

    show_result(img_t, img_c, "orb" + name, kp_t, kp_c, matches, good_matches)

def detect_img_surf(img_t, img_c, name, num):
    surf = cv2.xfeatures2d_SURF.create()

    bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    kp_t, des_t = surf.detectAndCompute(img_t, None)
    kp_c, des_c = surf.detectAndCompute(img_c, None)

    img_t = cv2.drawKeypoints(img_t, kp_t, None)
    img_c = cv2.drawKeypoints(img_c, kp_c, None)

    matches = bf.match(des_t, des_c)
    matches.sort(key=lambda x: x.distance)
    good_matches = matches[:num]

    show_result(img_t, img_c, "surf" + name, kp_t, kp_c, matches, good_matches)

def detect_img_sift(img_t, img_c, name, num):
    sift = cv2.SIFT_create()

    bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    kp_t, des_t = sift.detectAndCompute(img_t, None)
    kp_c, des_c = sift.detectAndCompute(img_c, None)

    img_t = cv2.drawKeypoints(img_t, kp_t, None)
    img_c = cv2.drawKeypoints(img_c, kp_c, None)

    matches = bf.match(des_t, des_c)
    matches.sort(key=lambda x: x.distance)

    good_matches = matches[:num]

    """# Apply ratio test
    good = []
    for m,n,o in matches:
        if m.distance < 0.65*n.distance:
            if m.distance < 0.65*o.distance:
                good.append([m])
                
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    """

    show_result(img_t, img_c, "sift" + name, kp_t, kp_c, matches, good_matches)

for root, dirs, files in os.walk(CHALLENGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            img_t = cv2.imread(os.path.join(SYMBOL_DIR, "SlideB.PNG"))
            img_c = cv2.imread(os.path.join(root, file))
            """detect_img_orb(img_t, img_c, file, 20)
            detect_img_sift(img_t, img_c, file, 40)
            detect_img_surf(img_t, img_c, file, 40)"""
            #cv2.imshow(file, cv2.imread(os.path.join(root, file)))


cap = cv2.VideoCapture(0)

while(1):

    ret, frame = cap.read()
    img_t = cv2.imread(os.path.join(SYMBOL_DIR, "SlideB.PNG"))
    img_t2 = cv2.imread(os.path.join(SYMBOL_DIR, "SlideY.PNG"))

    #detect_img_orb(img_t, frame, "cap", 20)
    detect_img_sift(img_t, frame, "cap", 40)
    detect_img_surf(img_t, frame, "cap", 40)

    #detect_img_orb(img_t2, frame, "cap2", 20)
    detect_img_sift(img_t2, frame, "cap2", 40)
    detect_img_surf(img_t2, frame, "cap2", 40)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

# template = cv2.imread()
