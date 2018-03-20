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


debug = False
p = Print(debug)

def get_mask(gray):
    clahe = clahe_gray(gray)
    equ_mode = get_mode(clahe)

    if equ_mode == None:
        equ_mode = np.mean(clahe)  

    equ_mode = int(0.25*equ_mode)

    _, th = cv.threshold(clahe, equ_mode, 255, cv.THRESH_BINARY_INV)

    return th

def get_oil_gland(img):
    global p
    row, col, _ = img.shape
    res_cnt = img.copy()

    # blur = cv.medianBlur(img, 5)
    clahe = clahe_by_Lab(img)
    blur = cv.bilateralFilter(clahe,7,75,75)
    b,g,r = cv.split(blur)
   
    mask_g = get_mask(g)
    mask_r = get_mask(r)

    mask_xor = ~(mask_r ^ mask_g)
    mask_r_xor = mask_r & mask_xor
    mask_g_xor = mask_g & mask_xor

    mask = (mask_r_xor & mask_g_xor) 
    
    p.imshow('mask_xor', mask_xor)
    p.imshow('mask_g', mask_g)
    p.imshow('mask_r', mask_r)
    p.imshow('mask_r_xor', mask_r_xor)
    p.imshow('mask_g_xor', mask_g_xor)
    p.imshow('mask', mask)
    p.imshow('g', g)
    p.imshow('r', r)

    k = cv.waitKey(-1)
    if k == ord('e'):
        exit(0)
    if not p.get_mode():
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
            img = cv.imread(CONST.IMG_CIRCLE_PATH + img_name, 1)
            # print(img)
            if img is None:
                print('Cannot read image: ', img_name)
                continue

            res = get_oil_gland(img)

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
