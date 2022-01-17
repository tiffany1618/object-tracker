import cv2 as cv

LUCAS_KANADE = {
    "winSize": (15, 15),
    "maxLevel": 2,
    "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
}

CONTOUR_PARAMS = {
    "gaussianKernelSize": (11, 11),
    "gaussianSigmaX": 0,
    "gaussianSigmaY": 0,
    "cannyThreshMin": 100,
    "cannyThreshMax": 200,
    "arcLengthMin": 100,
}
