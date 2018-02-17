'''
    File name: draw_circle.py
    Author: PomeloX
    Date created: 2/1/2018
    Date last modified: 2/17/2018
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
    lower_bound = np.array([240, 240, 240], dtype=np.uint8)
    upper_bound = np.array([255, 255, 255], dtype=np.uint8)
    res_inrange = cv2.inRange(img, lower_bound, upper_bound)

    _, cnts, _ = cv2.findContours(
        res_inrange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    circles = []

    res_x, res_y, res_r = 0, 0, 0

    for cnt in cnts:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        x = int(x)
        y = int(y)
        r = int(r)
        r_min = 100

        if r < r_min:
            continue

        if r > res_r:
            res_r = r
            res_x = x
            res_y = y

    if res_r == 0:
        return None

    res_r = int(res_r*0.95)
    cv2.circle(res_cnt, (res_x, res_y), res_r, (0, 0, 255), 2)

    mask = cv2.rectangle(mask, (0, 0), (col, row), (0), -1)
    mask = cv2.circle(mask, (res_x, res_y), res_r, (255), -1)

    # Use Blur and CLAHE with BGR image
    blur = cv2.medianBlur(img_resize, 3)
    clahe = clahe_by_Lab(blur)

    gray = cv2.cvtColor(clahe, cv2.COLOR_BGR2GRAY)
    # Invert B/W
    gray = 255 - gray

    # CLAHE Grayscale
    gray_clahe = clahe_gray(gray)

    # Clear noise by make gray to black
    gray_mean = np.array(gray_clahe).mean()
    gray_clahe[gray_clahe < gray_mean] = 0

    equ = equalization_gray(gray_clahe)

    _, result = cv2.threshold(equ, gray_mean * 1.2, 255, cv2.THRESH_BINARY)

    result = cv2.bitwise_and(result, result, mask=mask)
    result = result[res_y - res_r:res_y + res_r, res_x - res_r:res_x + res_r]
    result = cv2.resize(result, (CONST.RESULT_WIDTH, CONST.RESULT_HEIGHT))

    p.imshow('blur', blur)
    p.imshow('clahe', clahe)
    p.imshow('gray', gray)
    p.imshow('gray_clahe', gray_clahe)
    p.imshow('gray_equ', equ)
    p.imshow('res_inrange', res_inrange)
    p.imshow('result', result)

    k = cv2.waitKey(0)
    if k == ord('e'):
        exit(0)

    if not p.get_mode():
        return result


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
            img_resize = cv2.imread(CONST.IMG_RESIZE_PATH + img_name, 1)

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
