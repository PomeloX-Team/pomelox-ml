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


def get_mask(gray):
    clahe = clahe_gray(gray)
    equ_mode = get_mode(clahe)

    if equ_mode == None:
        equ_mode = np.mean(clahe)

    equ_mode = int(0.25 * equ_mode)

    _, th = cv.threshold(clahe, equ_mode, 255, cv.THRESH_BINARY_INV)

    return th


def mask_only_circle(img):
    blur = cv.medianBlur(img, 5)
    clahe = clahe_by_Lab(blur)
    equ = equalization_bgr(clahe)
    gray = cv.cvtColor(equ, cv.COLOR_BGR2GRAY)

    gray_mode = get_mode(gray)
    if gray_mode == None:
        gray_mode = 40
    gray_mode = max(40, int(gray_mode * 0.3))
    print(gray_mode)
    _, th = cv.threshold(gray, gray_mode, 255, cv.THRESH_BINARY_INV)

    row, col = th.shape
    res = np.zeros((row, col), dtype=np.uint8)

    dist_transform = cv.distanceTransform(th, cv.DIST_L2, 5)
    print(dist_transform.max())
    
    dist_list = [0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1]
    dist_list.reverse()
    for i in dist_list:
        print('dist',i)
        ret, th1 = cv2.threshold(dist_transform, i * dist_transform.max(), 255, 0)
        plt.imshow(dist_transform) 
        plt.show()
        # k=cv.waitKey(-1)
        # plt.close()
        th1 = np.uint8(th1)
        _, cnts, _ = cv.findContours(th1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        for cnt in cnts:
            (x, y), (width, height), angle = cv2.minAreaRect(cnt)
            
            if width > 0 and height > 0 and not 0.4 <= (width * 1.0) / height <= 1.6:
                continue
            (x, y), r = cv.minEnclosingCircle(cnt)

            area = cv2.contourArea(cnt)
            area_cir = math.pi * r * r
            area_ratio = (area * 1.0) / area_cir
            if area_ratio < 0.45:
                continue
            
            if area_ratio < 0.5:
                r = int(r*0.8)


            cv.circle(res, (int(x), int(y)), int(r), (255), -1)
            cv.circle(dist_transform, (int(x), int(y)), int(r)+2, (0), -1)
            cv.circle(th, (int(x), int(y)), int(r), (int(255*i)), 1)

            # p.imshow('th', th)
            # p.imshow('th1', th1)
            # p.imshow('dis', dist_transform)
        
    p.imshow('gray', gray)
    p.imshow('equ', equ)
    p.imshow('th', th)
    # p.imshow('res_sum', res_sum)
    p.imshow('res', res)
    # p.imshow('th_cir', th_cir)
    # p.imshow('mask',cir_connected)
    k=cv.waitKey(-1)
    if k == ord('e'):
        exit(0)
    if not p.get_mode():
        return res

def main():
    print('>' * 10, 'Get oil gland from the image(s)', '<' * 10, '\n')
    symbol_list=get_symbol_list()

    date=gen_date()
    overwrite=False

    for s in symbol_list:
        for j in date:
            img_name=s + '_' + j + '.JPG'
            print(img_name, 'is processing...\n')
            img=cv.imread(CONST.IMG_CIRCLE_PATH + img_name, 1)
            # print(img)
            if img is None:
                print('Cannot read image: ', img_name)
                continue

            # res = get_oil_gland(img,img_name)
            res=mask_only_circle(img)
            if res is None:
                print('Cannot get oil gland from the image: ', img_name)
                continue
            if p.get_mode():
                continue
            if (not overwrite) and (not cv.imread(CONST.IMG_OIL_GLAND_PATH + img_name, 1) is None):
                print('There is already a file with the same name\nPress s or S to Skip\nPress o or O to Overwrite\nPress a or A to Overwrite to all\nPress e or E to exit program\n')
                k=input()
                k=k.lower()
                if k == 's':
                    continue
                elif k == 'a':
                    overwrite=True
                elif k == 'e':
                    break

            cv.imwrite(CONST.IMG_OIL_GLAND_PATH + img_name, res)


if __name__ == '__main__':
    p.change_mode(True)
    main()
    pass
