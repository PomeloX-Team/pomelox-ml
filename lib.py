'''
    File name: lib.py
    Author: PomeloX
    Date created: 1/16/2018
    Date last modified: 2/17/2018
    Python Version: 3.6.1
'''

import cv2
import numpy as np
import operator


class Print:
    def __init__(self, debug_mode):
        self.debug_mode = debug_mode

    def print(self, string):
        if self.debug_mode:
            print(str(string))

    def imshow(self, winname, mat):
        if self.debug_mode:
            cv2.imshow(winname, mat)

    def change_mode(self, mode):
        self.debug_mode = mode

    def get_mode(self):
        return self.debug_mode


def gen_date():
    date = []
    for i in range(1, 31, 1):
        str_date = '201711' + str(i).zfill(2)
        date.append(str_date)

    for i in range(1, 26, 1):
        str_date = '201712' + str(i).zfill(2)
        date.append(str_date)
    return date


def get_mode(channel, min=0, max=255):
    # numpy return a contiguous flattened array.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             tened array.
    data = np.array(channel)
    data = data.ravel()
    data = list(data)
    count = {}

    for i in range(min, max + 1):
        count[i] = data.count(i)

    count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)

    max = count[0][1]
    print(count)
    mode = []
    for ct in count:
        if ct[1] < max:
            break
        mode.append(ct[0])
        print(mode, max)
    mode = np.array(mode)
    mode = mode.mean()
    return int(mode)


def get_kernel(shape='rect', ksize=(5, 5)):
    if shape == 'rect':
        return cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    elif shape == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    elif shape == 'plus':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
    elif shape == '\\':
        kernel = np.diag([1] * ksize[0])
        return np.uint8(kernel)
    elif shape == '/':
        kernel = np.fliplr(np.diag([1] * ksize[0]))
        return np.uint8(kernel)
    else:
        return None


def brightness(img_bgr, brightnessValue):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.uint16(v)
    v = np.clip(v + brightnessValue, 0, 255)
    v = np.uint8(v)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def brightness_gray(img_gray, brightnessValue):
    bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.uint16(v)
    v = np.clip(v + brightnessValue, 0, 255)
    v = np.uint8(v)
    hsv = cv2.merge((h, s, v))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def equalization_bgr(img_bgr):
    b, g, r = cv2.split(img_bgr)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    equBGR = cv2.merge((b, g, r))
    return equBGR


def equalization_hsv(img_hsv):
    h, s, v = cv2.split(img_hsv)
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)
    equHSV = cv2.merge((h, s, v))
    return equHSV


def equalization_gray(img_gray):
    equGRAY = cv2.equalizeHist(img_gray)

    return equGRAY


def clahe_gray(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    resGRAY = clahe.apply(img_gray)
    return resGRAY


def clahe_by_Lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    resBGR = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return resBGR


def clahe_by_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    v = clahe.apply(v)
    s = clahe.apply(s)
    hsv = cv2.merge((h, s, v))
    resBGR = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return resBGR
