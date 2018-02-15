import cv2
import numpy as np
from lib import *
import constant as CONST
from matplotlib import pyplot as plt


def nothing(x):
    pass


def color_range(img):
    color = ['brown', 'red', 'orange']
    cv2.namedWindow('image')
    cv2.createTrackbar('Hmin', 'image', 0, 179, nothing)
    cv2.createTrackbar('Smin', 'image', 0, 255, nothing)
    cv2.createTrackbar('Vmin', 'image', 0, 255, nothing)
    cv2.createTrackbar('Hmax', 'image', 0, 179, nothing)
    cv2.createTrackbar('Smax', 'image', 0, 255, nothing)
    cv2.createTrackbar('Vmax', 'image', 0, 255, nothing)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    # hist = cv2.calcHist([h], [0], None, [256], [0, 256])
    # plt.plot(hist, color=color[0])
    # hist = cv2.calcHist([s], [0], None, [256], [0, 256])
    # plt.plot(hist, color=color[1])
    # hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    # plt.plot(hist, color=color[2])
    # plt.show()
    while True:
        if img is None:
            continue
        h, s, v = cv2.split(hsv)
        s = equalization_gray(s)
        v = equalization_gray(v)
        hsv = cv2.merge((h, s, v))
        h_min = cv2.getTrackbarPos('Hmin', 'image')
        s_min = cv2.getTrackbarPos('Smin', 'image')
        v_min = cv2.getTrackbarPos('Vmin', 'image')
        h_max = cv2.getTrackbarPos('Hmax', 'image')
        s_max = cv2.getTrackbarPos('Smax', 'image')
        v_max = cv2.getTrackbarPos('Vmax', 'image')
        lowerb = np.array([h_min, s_min, v_min], np.uint8)
        upperb = np.array([h_max, s_max, v_max], np.uint8)
        hsv_inrange = cv2.inRange(hsv, lowerb, upperb)
        res_bgr = cv2.bitwise_and(img, img, mask=hsv_inrange)
        cv2.imshow('image', hsv_inrange)
        cv2.imshow('image_bgr', res_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':

    img = cv2.imread(CONST.IMG_PATH + '/A1_20171225.JPG', 1)
    img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    color_range(img)
