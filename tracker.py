import cv2 as cv
import numpy as np

from util import params, display


def track_objects(video_path):
    # Read video
    cap = cv.VideoCapture(video_path)

    # Find corners on first frame
    ret, prev_frame = cap.read()
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    prev_pts = cv.goodFeaturesToTrack(prev_gray, mask=None, **params.SHI_TOMASI)

    # Frame count for testing
    frame_count = 0

    while frame_count < 2:
        ret, next_frame = cap.read()

        next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)

        mask, prev_pts = detect_objects(prev_gray, next_gray, prev_pts)

        # Display
        display.display(cv.add(prev_frame, mask))
        display.display(next_frame)

        prev_gray = next_gray.copy()
        prev_frame = next_frame.copy()
        frame_count = frame_count + 1


def detect_objects(prev_frame, next_frame, prev_pts):
    # Lucas Kanade optical flow
    next_pts, status, err = cv.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_pts, None, **params.LUCAS_KANADE)

    # Select good points
    prev_good = []
    next_good = []
    if next_pts is not None:
        prev_good = prev_pts[status == 1]
        next_good = next_pts[status == 1]

    # Directional masks
    mask_left = np.zeros_like(prev_frame)
    mask_right = np.zeros_like(prev_frame)
    mask_up = np.zeros_like(prev_frame)
    mask_down = np.zeros_like(prev_frame)

    directions = (next_good - prev_good).astype(np.int)
    prev_int = prev_good.astype(np.int)
    for i, (x, y) in enumerate(directions):
        if x < -1:
            cv.circle(mask_left, prev_int[i], 5, 255, -1)
        elif x > 1:
            cv.circle(mask_right, prev_int[i], 5, 255, -1)

        if y < -1:
            cv.circle(mask_up, prev_int[i], 5, 255, -1)
        elif y > 1:
            cv.circle(mask_down, prev_int[i], 5, 255, -1)

    prev_pts = next_good.reshape(-1, 1, 2)
    mask = cv.add(cv.add(cv.add(binary_to_rgb(mask_left, [0, 255, 0]), binary_to_rgb(mask_right, [255, 0, 0])),
                  binary_to_rgb(mask_up, [0, 0, 255])), binary_to_rgb(mask_down, [0, 255, 255]))

    return mask, prev_pts


# Convert binary mask to RGB mask with color color
def binary_to_rgb(mask, color):
    mask_rgb = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    h, w = mask.shape
    mask_color = np.full((h, w, 3), color, dtype=np.uint8)
    np.copyto(mask_rgb, mask_color, where=(mask_rgb != [0, 0, 0]))

    return mask_rgb


# Remove noise by twice opening after twice closing with 5x5 diamond shape structuring element
def remove_noise(mask):
    kernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.uint8)

    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    return mask
