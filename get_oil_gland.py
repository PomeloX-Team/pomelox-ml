'''
    File name: get_oil_gland.py
    Author: PomeloX
    Date created: 2018/2/13
    Date last modified: 2018/03/12
    Python Version: 3.6.1
'''

import math
import cv2 as cv
import numpy as np
from lib import *
import constant as CONST
from operator import itemgetter
from matplotlib import pyplot as plt

debug = False
p = Print(debug)


def mask_only_circle(img):
    row, col, _ = img.shape
    res = np.zeros((row, col), dtype=np.uint8)
    blur = cv.medianBlur(img, 5)
    clahe = clahe_by_Lab(blur)

    # equ = equalization_bgr(clahe)
    gray = cv.cvtColor(clahe, cv.COLOR_BGR2GRAY)
    gray2color = color_mapping(gray)
    print(gray2color.shape)
    # color2hsv = cv.cvtColor(gray2color,cv.COLOR_BGR2HSV)
    # p.imshow_float('gray.copy',gray.copy())
    lower = np.array([100,0,0],dtype=np.uint8)
    upper = np.array([255,255,0],dtype=np.uint8)
    mask = cv.inRange(gray2color,lower,upper) 

    lower = np.array([0,0,0],dtype=np.uint8)
    upper = np.array([255,0,255],dtype=np.uint8)
    mask += cv.inRange(gray2color,lower,upper) 
    
    lower = np.array([0,0,250],dtype=np.uint8)
    upper = np.array([0,0,255],dtype=np.uint8)
    mask -= cv.inRange(gray2color,lower,upper)

    # p.imshow('color2hsv',color2hsv)
    a = gray2color & cv.cvtColor(mask,cv.COLOR_GRAY2BGR)
    p.imshow('gray+mask',a)

    _, th = cv.threshold(mask,127,255,cv.THRESH_BINARY_INV)
    mask = th.copy() 
    _, cnts, _ = cv.findContours(
            th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for cnt in cnts:
        rect = (x,y), (w, h), angle  = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(gray2color,[box],0,(255,255,255),2)
        area = cv.contourArea(cnt)
        if w <=0 or h <= 0 :
            continue
        if float(w)/h >= 1.3 or float(w)/h <= 0.7 or area/(w*h) <= 0.5:     
            x,y,w,h = cv.boundingRect(cnt)
            roi_mask = mask[y:y+h, x:x+w]
            roi_gray2color = gray2color[y:y+h, x:x+w]
            
            lower = np.array([0,255,0],dtype=np.uint8)
            upper = np.array([255,255,255],dtype=np.uint8)
            mask_tmp = cv.inRange(roi_gray2color,lower,upper)

            roi_mask -= mask_tmp

            mask_tmp = mask.copy()
            mask_tmp[y:y+h, x:x+w] = roi_mask
            

            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(mask_tmp,[box],0,(255),300)
            cv.drawContours(gray2color,[box],0,(0,0,0),2)
            mask = mask & mask_tmp



    p.imshow('gray2color',gray2color)
    p.imshow('mask',mask)
    # gray_mode = get_mode(gray)
    # if gray_mode == None:
    #     gray_mode = 40
    # gray_mode = max(40, int(gray_mode * 0.3))
    # print(gray_mode)
    # _, th = cv.threshold(gray, gray_mode, 255, cv.THRESH_BINARY_INV)


    # dist_transform = cv.distanceTransform(th, cv.DIST_L2, 3)
    # print(dist_transform.max())

    # dist_list = [float("%.3f" % (b*0.001)) for b in range(100,700,5)]
    # p.imshow_float('dist_transform',dist_transform)

    # for i in dist_list:
    #     print('dist', i)
    #     ret, th1 = cv.threshold(
    #         dist_transform, i * dist_transform.max(), 255, 0)
        
    #     th1 = np.uint8(th1)
    #     # p.imshow('th11',th1)
    #     # cv.waitKey(-1)
        
    #     _, cnts, _ = cv.findContours(
    #         th1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    #     for cnt in cnts:
    #         (x, y), (width, height), angle = cv.minAreaRect(cnt)

    #         if width > 0 and height > 0 and not 0.8-(i/2.0) <= (width * 1.0) / height <= 1.2+(i/2.0):
    #             continue
    #         (x, y), r = cv.minEnclosingCircle(cnt)

    #         area = cv.contourArea(cnt)
    #         area_cir = math.pi * r * r
    #         area_ratio = (area * 1.0) / area_cir
    #         if area_ratio < 0.7-(i/2.0):
    #             continue

    #         r = int((1+i)*r)

    #         cv.circle(res, (int(x), int(y)), int(r), (255), -1)
    #         cv.circle(dist_transform, (int(x), int(y)), int(r) + 2, (0), -1)
    #         cv.circle(th, (int(x), int(y)), int(r), (int(127)), 2)

       

    # p.imshow('gray', gray)
    # # p.imshow('equ', equ)
    # p.imshow('th', th)
    # p.imshow('res', res)
    p.imshow('img', img)

    k = cv.waitKey(-1)
    if k == ord('e'):
        exit(0)
    if not p.get_mode():
        return mask


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

            res = mask_only_circle(img)
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
    # p.change_mode(True)
    main()
    pass
