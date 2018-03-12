'''
    File name: lib.py
    Author: PomeloX
    Date created: 2018/1/16
    Date last modified: 2018/03/12
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
    # numpy return a contiguous flattened array.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             tened array.
    data = np.array(channel)
    data = data.ravel()
    data = list(data)

    count = {}

    for i in range(min, max + 1):
        count[i] = data.count(i)

    count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)

    max = count[0][1]
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
    equ_bgr = cv2.merge((b, g, r))
    return equ_bgr


def equalization_hsv(img_hsv):
    h, s, v = cv2.split(img_hsv)
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)
    equ_hsv = cv2.merge((h, s, v))
    return equ_hsv


def equalization_gray(img_gray):
    equ_gray = cv2.equalizeHist(img_gray)
    return equ_gray


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
    res_bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return res_bgr


def clahe_by_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    v = clahe.apply(v)
    s = clahe.apply(s)
    hsv = cv2.merge((h, s, v))
    res_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return res_bgr


def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def adjust_gamma_Lab(image):
    Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(Lab)
    L_mean = np.mean(L)
    L_mean = L_mean / 50.0
    gamma = L_mean
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
