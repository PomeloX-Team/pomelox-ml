import cv2 as cv
import numpy as np
from lib import *


def nothing(x):
    pass


def color_range():
    img = cv.imread(
        'C:/Users/skconan/Desktop/Workspace/pomeloX-ml/images_circle/A1_20171201.JPG')
    cv.namedWindow('image')
    cv.createTrackbar('minVal', 'image', 0, 500, nothing)
    cv.createTrackbar('maxVal', 'image', 0, 500, nothing)

    clahe = clahe_by_Lab(img)
    # blur = cv.bilateralFilter(clahe,7,75,75)
    gray = cv.cvtColor(clahe, cv.COLOR_BGR2GRAY)
    gray = equalization_gray(gray)
    cv2.imshow('img', img)
    while True:
        image = img.copy()
        min_val = cv.getTrackbarPos('minVal', 'image')
        max_val = cv.getTrackbarPos('maxVal', 'image')
        edges = cv.Canny(gray, min_val, max_val)
        image[edges > 127] = 255
        cv.imshow('image', image)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
    color_range()
