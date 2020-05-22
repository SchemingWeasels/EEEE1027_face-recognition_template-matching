import os
import cv2
import numpy as np
import csv
import time
import math
from statistics import mean

MODE = 'orb'
CSV_MODE = 'a'

BASE_DIR = os.path.dirname(__file__)
SYMBOL_DIR = os.path.join(BASE_DIR, '..', 'Task 2', 'Symbols3')
SYMBOL_DIR_EXTRA = os.path.join(BASE_DIR, '..', 'Task 2', "SymbolExtra")
CHALLENGE_DIR = os.path.join(BASE_DIR, "symbols sorted")
OUTPUT_DIR = os.path.join(BASE_DIR, 'rotation_' + MODE)

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

save_file = open(os.path.join(BASE_DIR, "rotation_" + MODE +".csv"), CSV_MODE, newline='')
writer = csv.writer(save_file)


if CSV_MODE == 'w':
    writer.writerow(["file", "detected", "ang"])


def count_shapes(image):
    # img = np.copy(image)
    img = cv2.imread(os.path.join(SYMBOL_DIR, "shapes.JPG"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    # Select inner 5%
    crop_percent = 8
    h, w = img.shape[:2]
    roi = img[int(h * crop_percent / 100):int(h * (1 - (crop_percent / 100))),
          int(w * crop_percent / 100):int(w * (1 - (crop_percent / 100)))]
    _, threshold = cv2.threshold(roi, 100, 240, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(roi, [approx], 0, (0), 5)
        if len(approx) > 2 and len(approx) < 5:
            shapes.append(len(approx))
        else:
            shapes.append(100)
    return shapes


def show_result(img_t, img_c, temp_name, file_name,  algo, kp_t, kp_c, matches, good_matches, avg, ang):
    detect = 'unknown'
    src_pts = np.float32([kp_t[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_c[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    h, w = img_t.shape[:2]


    if len(src_pts) > 7 and len(dst_pts) > 7 and len(src_pts) == len(dst_pts):
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)
        dst += (w, 0)  # adding offset

    if "orb" in algo:
        tresh = 40
    elif "sift" in algo:
        tresh = 150
    else:
        tresh = 0.09

    h1, w1 = img_t.shape[:2]
    h2, w2 = img_c.shape[:2]
    img_result = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    img_result[:h2, w1:w1 + w2] = img_c

    if (avg < tresh):
        img_result[:h1, :w1] = img_t
        detect = os.path.splitext(temp_name)[0]
        if "dist" in temp_name:
            cv2.putText(img_result, "dist", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "button" in temp_name:
            cv2.putText(img_result, "button", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "ball" in temp_name:
            cv2.putText(img_result, "ball", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "shape" in temp_name:
            cv2.putText(img_result, "shapes", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            shapes = count_shapes(img_t)
            cv2.putText(img_result, "Triangle:" + str(shapes.count(3)), (0, int(img_result.shape[0] * 0.50) + 20)
                        , cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img_result, "Square:" + str(shapes.count(4)), (0, int(img_result.shape[0] * 0.50) + 40)
                        , cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img_result, "Circle:" + str(shapes.count(100)), (0, int(img_result.shape[0] * 0.50) + 60)
                        , cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "stop" in temp_name:
            cv2.putText(img_result, "stop", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "angle" in temp_name:
            cv2.putText(img_result, "angle", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        elif "red" in temp_name:
            cv2.putText(img_result, "Roses are RED", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        elif "green" in temp_name:
            cv2.putText(img_result, "Green Grass", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        elif "yellow" in temp_name:
            cv2.putText(img_result, "Yellow Dandelion", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        elif "blue" in temp_name:
            cv2.putText(img_result, "Blue Dabudee", (0, int(img_result.shape[0] * 0.50)),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        # Draw poly box in Red
        if len(src_pts) > 7 and len(dst_pts) > 7 and len(src_pts) == len(dst_pts):
            img_result = cv2.polylines(img_result, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(img_result,
                    "matches:" + str(len(matches)) + " Min d:" + str(f"{matches[0].distance:.5f}" + " Ave " + str(avg)),
                    (0, int(img_result.shape[0] * 0.98)),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    output_dir = os.path.join(OUTPUT_DIR, detect)
    writer.writerow([file_name, detect, ang])
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    save_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    cv2.imwrite(os.path.join(save_dir, str(ang)+'.jpg'), img_result)


def detect_best_orb(templates, template_names, kp_t, des_t, img_c, name, top, ang):
    orb = cv2.ORB_create()  # WTA_K=3)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    kp_c, des_c = orb.detectAndCompute(img_c, None)

    all_matches = []
    avg = []
    for des in des_t:
        matches = bf.match(des, des_c)
        matches.sort(key=lambda x: x.distance)
        # Avarge top 10
        top_10 = matches[:8]
        avg.append(mean(d.distance for d in top_10))
        all_matches.append(matches)

    # Sorting everything
    avg, templates, template_names, all_matches, kp_t, des_t = zip(
        *sorted(zip(avg, templates, template_names, all_matches, kp_t, des_t), key=lambda x: x[0]))

    good_matches = all_matches[0][:top]
    show_result(templates[0], img_c, template_names[0],  name, 'orb', kp_t[0], kp_c, all_matches[0], good_matches,
                avg[0], ang)


def detect_best_surf(templates, template_names, kp_t, des_t, img_c, name, top, ang):
    surf = cv2.xfeatures2d_SURF.create()

    bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    kp_c, des_c = surf.detectAndCompute(img_c, None)

    all_matches = []
    avg = []
    for des in des_t:
        matches = bf.match(des, des_c)
        matches.sort(key=lambda x: x.distance)
        # Avarge top 10
        top_10 = matches[:8]
        avg.append(mean(d.distance for d in top_10))
        all_matches.append(matches)

    # Sorting everything
    avg, templates, template_names, all_matches, kp_t, des_t = zip(
        *sorted(zip(avg, templates, template_names, all_matches, kp_t, des_t), key=lambda x: x[0]))

    good_matches = all_matches[0][:top]
    show_result(templates[0], img_c, template_names[0],  name, 'surf', kp_t[0], kp_c, all_matches[0], good_matches,
                avg[0], ang)


def detect_best_sift(templates, template_names, kp_t, des_t, img_c, name, top, ang):
    sift = cv2.SIFT_create()

    bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)

    kp_c, des_c = sift.detectAndCompute(img_c, None)

    all_matches = []
    avg = []
    for des in des_t:
        matches = bf.match(des, des_c)
        matches.sort(key=lambda x: x.distance)
        # Avarge top 10
        top_10 = matches[:8]
        avg.append(mean(d.distance for d in top_10))
        all_matches.append(matches)

    # Sorting everything
    avg, templates, template_names, all_matches, kp_t, des_t = zip(
        *sorted(zip(avg, templates, template_names, all_matches, kp_t, des_t), key=lambda x: x[0]))

    good_matches = all_matches[0][:top]
    show_result(templates[0], img_c, template_names[0], name, 'sift', kp_t[0], kp_c, all_matches[0], good_matches,
                avg[0], ang)


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


def gen_template_surf(templates, template_names):
    surf = cv2.xfeatures2d_SURF.create()
    kp = []
    des = []
    for img_t, file in zip(templates, template_names):
        kp_t, des_t = surf.detectAndCompute(img_t, None)
        kp.append(kp_t)
        des.append(des_t)
    return kp, des


def gen_template_sift(templates, template_names):
    sift = cv2.SIFT_create()
    kp = []
    des = []
    for img_t, file in zip(templates, template_names):
        kp_t, des_t = sift.detectAndCompute(img_t, None)
        kp.append(kp_t)
        des.append(des_t)
    return kp, des


def gen_template_orb(templates, template_names):
    orb = cv2.ORB_create()
    kp = []
    des = []
    for img_t, file in zip(templates, template_names):
        kp_t, des_t = orb.detectAndCompute(img_t, None)
        kp.append(kp_t)
        des.append(des_t)
    return kp, des


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


templates, template_names = load_templates(SYMBOL_DIR)
temp, temp_names = load_templates_extra(SYMBOL_DIR_EXTRA)
templates += temp
template_names += temp_names
kp_t, des_t = [], []
if MODE == 'sift':
    kp_t, des_t = gen_template_sift(templates, template_names)
elif MODE == 'surf':
    kp_t, des_t = gen_template_surf(templates, template_names)
elif MODE == 'orb':
    kp_t, des_t = gen_template_orb(templates, template_names)


for root, dirs, files in os.walk(CHALLENGE_DIR):
    if "unused" in root:
        continue
    for file in files:
        file = file.lower()
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            print("matching " + file)
            img = cv2.imread(os.path.join(root, file))
            for ang in range(-91, 91):
                img_c = rotate_image(img, ang)
                if MODE == 'sift':
                    detect_best_sift(templates, template_names, kp_t, des_t, img_c, file, 40, ang)
                elif MODE == 'surf':
                    detect_best_surf(templates, template_names, kp_t, des_t, img_c, file, 40, ang)
                elif MODE == 'orb':
                    detect_best_orb(templates, template_names, kp_t, des_t, img_c, file, 40, ang)

# cv2.imshow(file, cv2.imread(os.path.join(root, file)))

print("Done")
cv2.destroyAllWindows()

# template = cv2.imread()
