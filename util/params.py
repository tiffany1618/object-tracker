import cv2 as cv

SHI_TOMASI = {
    "maxCorners": 200,
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7,
}
LUCAS_KANADE = {
    "winSize": (15, 15),
    "maxLevel": 2,
    "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
}
