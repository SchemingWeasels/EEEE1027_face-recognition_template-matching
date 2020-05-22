import os
import cv2
import numpy as np
from statistics import mean

BASE_DIR = os.path.dirname(__file__)
SYMBOL_DIR = os.path.join(BASE_DIR, "Symbols3")
SYMBOL_DIR_EXTRA = os.path.join(BASE_DIR, "SymbolExtra")
CHALLENGE_DIR = os.path.join(BASE_DIR, "Challenge")
SYMBOL_PREFX = "Slide"

def show_result(img_t, img_c,name, algo, kp_t, kp_c, matches, good_matches, avg):
    src_pts = np.float32([kp_t[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_c[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    h, w = img_t.shape[:2]

    if len(good_matches) > 5:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)
        dst += (w, 0)  # adding offset

    ret = cv2.minAreaRect(dst_pts)
    box = cv2.boxPoints(ret)
    box = np.int0(box)
    box += (w, 0)

    if "orb" in algo:
        tresh = 80
    elif "sift" in algo:
        tresh = 150
    else:
        tresh = 0.08

    if(avg<tresh):
        img_result = cv2.drawMatches(img_t, kp_t, img_c, kp_c, good_matches, None)
        if "dist" in name:
            cv2.putText(img_result, "dist", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "button" in name:
            cv2.putText(img_result, "button", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "ball" in name:
            cv2.putText(img_result, "ball", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "shape" in name:
            cv2.putText(img_result, "shapes", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "stop" in name:
            cv2.putText(img_result, "stop", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "angle" in name:
            cv2.putText(img_result, "angle", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "red" in name:
            cv2.putText(img_result, "Roses are RED", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        elif "green" in name:
            cv2.putText(img_result, "Green Grass", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        elif "yellow" in name:
            cv2.putText(img_result, "Yellow Dandelion", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        elif "blue" in name:
            cv2.putText(img_result, "Blue Dabudee", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        #Draw minAreaRect
        cv2.drawContours(img_result, [box], 0, (0, 255, 0), 3)
        # Draw poly box in Red
        if len(good_matches) > 5:
            img_result = cv2.polylines(img_result, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(img_result,
                "matches:" + str(len(matches)) + " Min d:" + str(f"{matches[0].distance:.5f}" + " Ave " + str(avg)),
                (0, int(img_result.shape[0] * 0.98)),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    else:
        h1, w1 = img_t.shape[:2]
        h2, w2 = img_c.shape[:2]
        img_result = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        img_result[:h2, w1:w1 + w2] = img_c

    cv2.imshow(algo, img_result)

def detect_best_orb(templates, template_names, kp_t, des_t, img_c, name, top):
    orb = cv2.ORB_create()  # WTA_K=3)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    kp_c, des_c = orb.detectAndCompute(img_c, None)

    all_matches = []
    avg = []
    for des in des_t:
        matches = bf.match(des, des_c)
        matches.sort(key=lambda x: x.distance)
        # Avarge top 10
        top_10 = matches[:10]
        avg.append(mean(d.distance for d in top_10))
        all_matches.append(matches)

    # Sorting everything
    avg, templates, template_names, all_matches, kp_t, des_t = zip(*sorted(zip(avg, templates, template_names, all_matches, kp_t, des_t), key= lambda x:x[0]))

    img_t = cv2.drawKeypoints(templates[0], kp_t[0], None)
    img_c = cv2.drawKeypoints(img_c, kp_c, None)

    good_matches = all_matches[0][:top]

    show_result(img_t, img_c, template_names[0], "orb" + name, kp_t[0], kp_c, all_matches[0], good_matches, avg[0])

def detect_best_surf(templates, template_names, kp_t, des_t, img_c, name, top):
    surf = cv2.xfeatures2d_SURF.create()

    bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    kp_c, des_c = surf.detectAndCompute(img_c, None)

    all_matches = []
    avg = []
    for des in des_t:
        matches = bf.match(des, des_c)
        matches.sort(key=lambda x: x.distance)
        # Avarge top 10
        top_10 = matches[:10]
        avg.append(mean(d.distance for d in top_10))
        all_matches.append(matches)

    # Sorting everything
    avg, templates, template_names, all_matches, kp_t, des_t = zip(*sorted(zip(avg, templates, template_names, all_matches, kp_t, des_t), key= lambda x:x[0]))

    img_t = cv2.drawKeypoints(templates[0], kp_t[0], None)
    img_c = cv2.drawKeypoints(img_c, kp_c, None)

    good_matches = all_matches[0][:top]

    show_result(img_t, img_c, template_names[0], "surf" + name, kp_t[0], kp_c, all_matches[0], good_matches, avg[0])

def detect_best_sift(templates, template_names, kp_t, des_t, img_c, name, top):
    sift = cv2.SIFT_create()

    bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    kp_c, des_c = sift.detectAndCompute(img_c, None)

    all_matches = []
    avg = []
    for des in des_t:
        matches = bf.match(des, des_c)
        matches.sort(key=lambda x: x.distance)
        #Avarge top 10
        top_10 = matches[:8]
        avg.append(mean(d.distance for d in top_10))
        all_matches.append(matches)

    #Sorting everything
    avg, templates, template_names, all_matches, kp_t, des_t = zip(*sorted(zip(avg, templates, template_names, all_matches, kp_t, des_t), key= lambda x:x[0]))

    img_t = cv2.drawKeypoints(templates[0], kp_t[0], None)
    img_c = cv2.drawKeypoints(img_c, kp_c, None)

    good_matches = all_matches[0][:top]
    show_result(img_t, img_c, template_names[0], "sift" + name, kp_t[0], kp_c, all_matches[0], good_matches, avg[0])

def load_templates(SYMBOL_DIR):
    templates = []
    template_names = []
    for root, dirs, files in os.walk(SYMBOL_DIR):
        for file in files:
            file = file.lower()
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg") or file.endswith("PNG"):
                img = cv2.imread(os.path.join(root, file))
                scale_percent = 25  # percent of original size
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                img = cv2.resize(img, dim)
                templates.append(img)
                template_names.append(file)
        break
    return templates, template_names

def load_templates_extra(SYMBOL_DIR):
    templates = []
    template_names = []
    for root, dirs, files in os.walk(SYMBOL_DIR):
        for file in files:
            file = file.lower()
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg") or file.endswith("PNG"):
                img = cv2.imread(os.path.join(root, file))
                scale_percent = 58.8  # percent of original size
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                img = cv2.resize(img, dim)
                templates.append(img)
                template_names.append(file)
        break
    return templates, template_names

def gen_template_surf(templates):
    surf = cv2.xfeatures2d_SURF.create()
    kp = []
    des = []
    for img_t in templates:
        kp_t, des_t = surf.detectAndCompute(img_t, None)
        kp.append(kp_t)
        des.append(des_t)
    return kp, des

def gen_template_sift(templates, template_names):
    surf = cv2.SIFT_create()
    kp = []
    des = []
    for img_t, file in zip(templates, template_names):
        if "999" in file:
            h, w = img_t.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            mask.fill(255)
            mask[:h, 0:int(w/2)] = np.zeros((h,int(w/2)), np.uint8)

            kp_t, des_t = surf.detectAndCompute(img_t, mask)

        else:
            kp_t, des_t = surf.detectAndCompute(img_t, None)
        kp.append(kp_t)
        des.append(des_t)
    return kp, des

def gen_template_orb(templates):
    orb = cv2.ORB_create()
    kp = []
    des = []
    for img_t in templates:
        kp_t, des_t = orb.detectAndCompute(img_t, None)
        kp.append(kp_t)
        des.append(des_t)
    return kp, des

templates, template_names = load_templates(SYMBOL_DIR)
temp, temp_names = load_templates_extra(SYMBOL_DIR_EXTRA)
templates += temp
template_names += temp_names
kp_sift, des_sift = gen_template_sift(templates, template_names)
#kp_surf, des_surf = gen_template_surf(templates)
#kp_orb, des_orb = gen_template_orb(templates)

"""
for root, dirs, files in os.walk(CHALLENGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            img_c = cv2.imread(os.path.join(root, file))
            detect_img_orb(img_t, img_c, file, 20)
            detect_img_sift(img_t, img_c, file, 20)
            detect_img_surf(img_t, img_c, file, 20)

            detect_best_sift(templates, kp_sift, des_sift, img_c, file, 40)

# cv2.imshow(file, cv2.imread(os.path.join(root, file)))
"""

cap = cv2.VideoCapture(0)

while (1):

    ret, frame = cap.read()

    #detect_best_surf(templates, kp_surf, des_surf, frame, "cap", 40)
    frame = cv2.GaussianBlur(frame, (3,3), 1)
    detect_best_sift(templates, template_names, kp_sift, des_sift, frame, "cap", 40)
    #detect_best_orb(templates, kp_orb, des_orb, frame, "cap", 40)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

# template = cv2.imread()
