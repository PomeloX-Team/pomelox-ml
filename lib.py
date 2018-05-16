'''
    File name: lib.py
    Author: PomeloX
    Date created: 2018/1/16
    Date last modified: 2018/03/12
    Python Version: 3.6.1
'''

import cv2 as cv
import numpy as np
import operator
import statistics


class Print:
    def __init__(self, debug_mode):
        self.debug_mode = debug_mode

    def print(self, string):
        if self.debug_mode:
            print(str(string))

    def imshow(self, winname, mat):
        if self.debug_mode:
            cv.imshow(winname, mat)

    def change_mode(self, mode):
        self.debug_mode = mode

    def get_mode(self):
        return self.debug_mode

    def imshow_float(self, winname, mat):
        if self.debug_mode:
            color_map = color_mapping(mat)
            cv.imshow(winname, color_map)


def color_mapping(mat):
    norm = None
    norm = cv.normalize(src=mat, dst=norm, alpha=0, beta=255,
                        norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC3)
    return cv.applyColorMap(norm, cv.COLORMAP_HSV)


def get_symbol_list():
    symbol_list = []
    print('=== Example A1,A2 ===\nThe number of symbol is 1\nThe symbol #1 is A\nThe ranges of symbol A is 1-2\n')
    print('\n=== Example A1,B1 ===\nThe number of symbol is 2\nThe symbol #1 is A\nThe ranges of symbol A is 1-1\nThe symbol #2 is B\nThe ranges of symbol B is 1-1\n')

    print('Put the number of symbol : ')
    n_symbol = input()

    for i in range(0, int(n_symbol), 1):
        print('Put the symbol #', i + 1, ': ')
        symbol = input()

        print('Put the ramges of symbol', symbol, ': ')
        symbol_range = input()

        range1, range2 = symbol_range.split('-')
        range1, range2 = int(range1), int(range2) + 1

        for j in range(range1, range2):
            symbol_list.append(symbol + str(j))

    print('\nResize the image(s) have a prefix :', symbol_list, '\n')

    return symbol_list


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
    # numpy return a contiguous flattened array.
    data = channel.ravel()
    data = np.array(data)

    if len(data.shape) > 1:
        data = data.ravel()
    try:
        mode = statistics.mode(data)
    except ValueError:
        mode = None
    return mode


def get_kernel(shape='[]', ksize=(5, 5)):
    if shape == '[]':
        return cv.getStructuringElement(cv.MORPH_RECT, ksize)
    elif shape == '0':
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize)
    elif shape == '+':
        return cv.getStructuringElement(cv.MORPH_CROSS, ksize)
    elif shape == '\\':
        kernel = np.diag([1] * ksize[0])
        return np.uint8(kernel)
    elif shape == '/':
        kernel = np.fliplr(np.diag([1] * ksize[0]))
        return np.uint8(kernel)
    else:
        return None


def brightness(img_bgr, brightnessValue):
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v = np.uint16(v)
    v = np.clip(v + brightnessValue, 0, 255)
    v = np.uint8(v)
    hsv = cv.merge((h, s, v))
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


def brightness_gray(img_gray, brightnessValue):
    bgr = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v = np.uint16(v)
    v = np.clip(v + brightnessValue, 0, 255)
    v = np.uint8(v)
    hsv = cv.merge((h, s, v))
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)


def equalization_bgr(img_bgr):
    b, g, r = cv.split(img_bgr)
    b = cv.equalizeHist(b)
    g = cv.equalizeHist(g)
    r = cv.equalizeHist(r)
    equ_bgr = cv.merge((b, g, r))
    return equ_bgr


def equalization_hsv(img_hsv):
    h, s, v = cv.split(img_hsv)
    s = cv.equalizeHist(s)
    v = cv.equalizeHist(v)
    equ_hsv = cv.merge((h, s, v))
    return equ_hsv


def equalization_gray(img_gray):
    equ_gray = cv.equalizeHist(img_gray)
    return equ_gray


def clahe_gray(img_gray):
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    resGRAY = clahe.apply(img_gray)
    return resGRAY


def clahe_by_Lab(img_bgr):
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv.merge((l, a, b))
    res_bgr = cv.cvtColor(lab, cv.COLOR_Lab2BGR)
    return res_bgr


def clahe_by_hsv(img_bgr):
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    v = clahe.apply(v)
    s = clahe.apply(s)
    hsv = cv.merge((h, s, v))
    res_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return res_bgr


def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv.LUT(image, table)


def adjust_gamma_Lab(image):
    Lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    L, a, b = cv.split(Lab)
    L_mean = np.mean(L)
    L_mean = L_mean / 50.0
    gamma = L_mean
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv.LUT(image, table)
