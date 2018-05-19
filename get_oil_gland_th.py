'''
    File name: get_oil_gland.py
    Author: PomeloX
    Date created: 2018/2/13
    Date last modified: 2018/03/12
    Python Version: 3.6.1
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from lib import *
import constant as CONST
from operator import itemgetter
import math

debug = False
p = Print(debug)


def mask_from_hsv(img):
    pass


def mask_only_circle(img):
    row, col, _ = img.shape
    res = np.zeros((row, col), dtype=np.uint8)
    clahe = clahe_by_Lab(img)
    blur = cv.medianBlur(clahe, 5)
    equ = equalization_bgr(blur)

    gray = cv.cvtColor(equ, cv.COLOR_BGR2GRAY)
    # gray = equalization_gray(gray)
    gray_mean = gray.mean()
    gray[gray>=gray_mean] = 170
    gray_mode = get_mode(gray)
    if gray_mode == None:
        gray_mode = 40
    gray_mode = max(40, int(gray_mode * 0.3))
    print(gray_mode)

    th_list = [b for b in range(0, 20)]

    for i in th_list:
        gray_mode -= i
        gray_mode = max(0, gray_mode)
        print('gray_mdoe', gray_mode)
        _, th = cv.threshold(gray, gray_mode, 255, cv.THRESH_BINARY_INV)
        dist_transform = cv.distanceTransform(th, cv.DIST_L2, 3)

        ret, th1 = cv2.threshold(
            dist_transform, 0.1 * dist_transform.max(), 255, 0)
 
        th1 = np.uint8(th1)
        _, cnts, _ = cv.findContours(
            th1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        for cnt in cnts:
            (x, y), (width, height), angle = cv2.minAreaRect(cnt)

            if width > 0 and height > 0 and not 0.8 - (i / 40.0) <= (width * 1.0) / height <= 1.2 + (i / 40.0):
                continue
            (x, y), r = cv.minEnclosingCircle(cnt)

            area = cv2.contourArea(cnt)
            area_cir = math.pi * r * r
            area_ratio = (area * 1.0) / area_cir
            if area_ratio < 0.7 - (i / 40.0):
                continue

            r = int((1+(i / 40.0))*r)

            cv.circle(res, (int(x), int(y)), int(r), (255), -1)
            cv.circle(dist_transform, (int(x), int(y)), int(r) + 2, (0), -1)
            cv.circle(th, (int(x), int(y)), int(r), (int(127)), 2)

        # kernel = get_kernel('/',(3,3))
        # dist_transform = cv.erode(dist_transform,kernel)
        # kernel = get_kernel('\\',(3,3))
        # dist_transform = cv.erode(dist_transform,kernel)
        p.imshow('th', th)
        p.imshow('th1', th1)
        p.imshow('dis', dist_transform)
        cv.waitKey(-1)

    p.imshow('equ', equ)
    p.imshow('th', th)
    # p.imshow('res_sum', res_sum)
    gray[res == 255] = 127
    p.imshow('gray', gray)
    p.imshow('res', res)
    p.imshow('clahe', clahe)
    # p.imshow('mask',cir_connected)
    k = cv.waitKey(-1)
    if k == ord('e'):
        exit(0)
    if not p.get_mode():
        return res


def main():
    print('>' * 10, 'Get oil gland from the image(s)', '<' * 10, '\n')
    symbol_list = get_symbol_list()

    date = gen_date()
    overwrite = False

    for s in symbol_list:
        for j in date:
            img_name = s + '_' + j + '.JPG'
            print(img_name, 'is processing...\n')
            img = cv.imread(CONST.IMG_RECT_PATH + img_name, 1)
            # print(img)
            if img is None:
                print('Cannot read image: ', img_name)
                continue

            # res = get_oil_gland(img,img_name)
            res = mask_only_circle(img)
            # res = mask_from_hsv(img)
            if res is None:
                print('Cannot get oil gland from the image: ', img_name)
                continue
            if p.get_mode():
                continue
            if (not overwrite) and (not cv.imread(CONST.IMG_OIL_GLAND_PATH + img_name, 1) is None):
                print('There is already a file with the same name\nPress s or S to Skip\nPress o or O to Overwrite\nPress a or A to Overwrite to all\nPress e or E to exit program\n')
                k = input()
                k = k.lower()
                if k == 's':
                    continue
                elif k == 'a':
                    overwrite = True
                elif k == 'e':
                    break

            cv.imwrite(CONST.IMG_OIL_GLAND_PATH + img_name, res)


if __name__ == '__main__':
    p.change_mode(True)
    main()
    pass
