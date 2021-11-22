import cv2 as cv


# Displays mask overlayed on img
def display(img):
    cv.imshow("frame", img)

    # Display image until esc key is pressed
    while True:
        k = cv.waitKey(30) & 0xFF
        if k == 27:
            cv.destroyAllWindows()
            break
