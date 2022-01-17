import math
import cv2 as cv
import numpy as np


# Displays mask overlayed on img
def display(img: np.array):
    cv.imshow("frame", img)

    # Display image until esc key is pressed
    while True:
        k = cv.waitKey(30) & 0xFF
        if k == 27:
            cv.destroyAllWindows()
            break


# Creates a grid of points across a 2D space of width and length with specified
# spacing (in pixels) between each point
def create_grid(height: int, width: int, spacing: int):
    x = np.linspace(0, width, math.floor(width / spacing))
    y = np.linspace(0, height, math.floor(height / spacing))

    return np.array([[[i, j]] for i in x for j in y], dtype=np.float32)


def find_rect_center(rect: np.array):
    x = (rect[0][0] + rect[1][0] + rect[2][0] + rect[3][0]) / 4.0
    y = (rect[0][1] + rect[1][1] + rect[2][1] + rect[3][1]) / 4.0

    return np.array([x, y])
