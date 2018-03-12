'''
    File name: get_oil_gland.py
    Author: PomeloX
    Date created: 2018/2/13
    Date last modified: 2018/03/12
    Python Version: 3.6.1
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from lib import *
import constant as CONST
from operator import itemgetter


debug = False
p = Print(debug)


def get_oil_gland(img, img_resize):
    global p

    row, col, _ = img.shape
    res_cnt = img.copy()
    mask = np.uint8(np.zeros((row, col)))

    # White value of bgr
    lower_bound = np.array([230, 230, 230], dtype=np.uint8)
    upper_bound = np.array([255, 255, 255], dtype=np.uint8)
    res_inrange = cv2.inRange(img, lower_bound, upper_bound)

    _, cnts, _ = cv2.findContours(
        res_inrange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    circles = []

    res_x, res_y, res_r = 0, 0, 0
    r_min = 40

    for cnt in cnts:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        x = int(x)
        y = int(y)
        r = int(r)
        cv2.circle(img, (x, y), r, (0, 255, 0), 1)

        if r < r_min:
            continue

        if r > res_r:
            res_r = r
            res_x = x
            res_y = y

    if res_r == 0:
        return None

    w = int(0.7 * res_r)
    img_rect = img[res_y - w:res_y + w, res_x - w:res_x + w]
    img_rect = cv2.resize(img_rect, (250, 250))

    # Use Blur and CLAHE with BGR image
    blur = cv2.medianBlur(img_rect, 3)
    clahe = clahe_by_Lab(blur)
    clahe = adjust_gamma_Lab(clahe)

    gray = cv2.cvtColor(clahe, cv2.COLOR_BGR2GRAY)
    equ = equalization_gray(gray)
    equ = clahe_gray(equ)
    equ_mode = get_mode(equ)
    equ_mode = int(equ_mode * 0.3)

    _, th = cv2.threshold(equ, equ_mode, 255, cv2.THRESH_BINARY_INV)

    p.imshow('gray', gray)
    p.imshow('equ', equ)
    p.imshow('th', th)
    cv2.waitKey(-1)
    return th


def main():
    print('>' * 10, 'Get oil gland from the image(s)', '<' * 10, '\n')
    symbol_list = get_symbol_list()

    date = gen_date()
    overwrite = False

    for s in symbol_list:
        for j in date:
            img_name = s + '_' + j + '.JPG'
            print(img_name, 'is processing...\n')
            img = cv2.imread(CONST.IMG_CIRCLE_PATH + img_name, 1)
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            img_resize = cv2.imread(CONST.IMG_RESIZE_PATH + img_name, 1)
            img_resize = cv2.resize(img_resize, (0, 0), fx=0.5, fy=0.5)

            if img is None:
                print('Cannot read image: ', img_name)
                continue

            res = get_oil_gland(img, img_resize)

            if res is None:
                print('Cannot get oil gland from the image: ', img_name)
                continue

            if (not overwrite) and (not cv2.imread(CONST.IMG_OIL_GLAND_PATH + img_name, 1) is None):
                print('There is already a file with the same name\nPress s or S to Skip\nPress o or O to Overwrite\nPress a or A to Overwrite to all\nPress e or E to exit program\n')
                k = input()
                k = k.lower()
                if k == 's':
                    continue
                elif k == 'a':
                    overwrite = True
                elif k == 'e':
                    break

            cv2.imwrite(CONST.IMG_OIL_GLAND_PATH + img_name, res)


if __name__ == '__main__':
    # p.change_mode(True)
    main()
    pass
