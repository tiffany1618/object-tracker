import math

import cv2 as cv
import numpy as np

from util import params, util


def track_objects(video_path):
    # Read video
    cap = cv.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    ret, next_frame = cap.read()
    _, obj_box, obj_center = extract_object(prev_frame, next_frame)

    while True:
        ret, next_frame = cap.read()
        next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

        # Find average movement of background using obj_box
        prev_box, next_box = lucas_kanade(prev_gray, next_gray, obj_box)
        num_pts = 0
        delta = np.array([0.0, 0.0])
        for i, point in enumerate(prev_box):
            if math.dist(point, next_box[i]) > 3:
                num_pts += 1
                delta[0] += next_box[i][0] - point[0]
                delta[1] += next_box[i][1] - point[1]
        if num_pts < 0:
            delta[0] /= float(num_pts)
            delta[1] /= float(num_pts)

        # Calculate movement vector as difference between background and object movement
        prev_center, next_center = lucas_kanade(prev_gray, next_gray, obj_center)
        delta -= next_center[0] - prev_center[0]

        # Display
        img = np.copy(prev_frame)
        rect = np.array([pts[0] for pts in obj_box])
        cv.drawContours(img, [np.int0(rect)], 0, (0, 255, 0), 2)
        cv.circle(img, util.find_rect_center(rect).astype(np.int), 3, (0, 255, 0), -1)
        center = np.int0([pts[0] for pts in obj_center])
        delta = (-10 * delta).astype(np.int)
        cv.arrowedLine(img, (center[0][0], center[0][1]), (center[0][0] + delta[0], center[0][1] + delta[1]), (0, 0, 255), 2)
        cv.imshow("img", img)

        if cv.waitKey(1) == 13:
            break

        if math.dist(prev_center[0], next_center[0]) > 5:
            _, obj_box, obj_center = extract_object(prev_frame, next_frame)
        else:
            # Object has moved too much, so recalculate the object
            obj_box = np.array([[(next_center[0] - prev_center[0]) + pts[0]] for pts in obj_box])
            obj_center = next_center.reshape(-1, 1, 2)

        prev_gray = next_gray.copy()
        prev_frame = next_frame.copy()

    cap.release()
    cv.destroyAllWindows()


def find_contours(frame, contour_params):
    # Find edges
    blur = cv.GaussianBlur(frame, contour_params["gaussianKernelSize"],
                           contour_params["gaussianSigmaX"], contour_params["gaussianSigmaY"])
    edges = cv.Canny(blur, contour_params["cannyThreshMin"], contour_params["cannyThreshMax"])

    # Find contours
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Filter contours
    contours_filtered = []
    for contour in contours:
        if (arc_len := cv.arcLength(contour, True)) > contour_params["arcLengthMin"]:
            # approx = cv.approxPolyDP(contour, 0.1 * arc_len, True)
            contours_filtered.append(contour)

    return contours_filtered


def lucas_kanade(prev_frame, next_frame, prev_pts):
    next_pts, status, err = cv.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_pts, None, **params.LUCAS_KANADE)

    # Select good points
    prev_good = []
    next_good = []
    if next_pts is not None:
        prev_good = prev_pts[status == 1]
        next_good = next_pts[status == 1]

    return prev_good, next_good


# Extract the object with the greatest difference in movement between the object itself and its background
def extract_object(prev_frame, next_frame):
    obj_index = 0
    obj_box = []
    max_dist_difference = 0

    contours = find_contours(prev_frame, params.CONTOUR_PARAMS)
    for i, contour in enumerate(contours):
        # Find bounding rectangle
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.array([[list(pts)] for pts in box], dtype=np.float32)

        prev_box, next_box = lucas_kanade(prev_frame, next_frame, box)
        max_box_dist = 0
        for j, point in enumerate(prev_box):
            if (dist := math.dist(point, next_box[j])) > max_box_dist:
                max_box_dist = dist

        contour = np.array(contour, dtype=np.float32)
        prev_cnt, next_cnt = lucas_kanade(prev_frame, next_frame, contour)
        total_cnt_dist = 0
        for j, point in enumerate(prev_cnt):
            total_cnt_dist += math.dist(point, next_cnt[j])

        if (diff := abs(max_box_dist - (total_cnt_dist / len(prev_cnt)))) > max_dist_difference:
            # print(f"box: {max_box_dist}, cnt: {(total_cnt_dist / len(prev_cnt))}, diff: {diff}")
            max_dist_difference = diff
            obj_index = i
            obj_box = box

    obj_center = util.find_rect_center(np.array([pts[0] for pts in obj_box]))
    obj_center = np.array([[obj_center]], dtype=np.float32)

    return contours[obj_index], obj_box, obj_center
