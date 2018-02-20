'''
    File name: features.py
    Author: PomeloX
    Date created: 1/22/2018
    Date last modified: 2/17/2018
    Python Version: 3.6.1
'''

import cv2
from lib import *
import numpy as np
import constant as CONST
import matplotlib.pyplot as plt
import math

p = Print(False)


def oil_gland_feature(img):
    global p

    res_cnt = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, cnts, _ = cv2.findContours(
        th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(res_cnt, cnts, -1, (0, 0, 255), 1)

    count = 0
    r_mean = []
    area_mean = []
    area_ratio = []
    radius = []
    for cnt in cnts:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cnt_area = cv2.contourArea(cnt)
        cir_area = math.pi*r*r
        area_mean.append(cnt_area)
        area_ratio.append(cnt_area / cir_area)
        if r > 40:
            continue
        
        if cnt_area / cir_area >= 0.6:
            radius.append(r)
        x = int(x)
        y = int(y)
        r = int(r)
        r_mean.append(r)
        count += 1
        cv2.circle(res_cnt, (x, y), r, (0, 255, 0), 1)
    
    # print(r_mean)
    radius = np.array(radius)
    radius = np.cov(radius)/25
    r_mean_sorted = sorted(r_mean)
    r_mean = np.array(r_mean).mean()
    area_ratio = np.array(area_ratio).mean()

    area_mean = np.array(area_mean).mean()
    p.print(area_ratio)
    min = r_mean_sorted[0]
    max = r_mean_sorted[-1]
    divide = (max-min)/3    
    ct = [0,0,0]
    for data in r_mean_sorted:
        if data <= min+divide:
            ct[0] += 1
        elif data <= max-divide:
            ct[1] += 1
        else:
            ct[2] += 1
    # res_cnt = cv2.addWeighted(res_cnt,0.5,res_cnt,0.5,0)
    # p.imshow('img_th', gray)
    p.imshow('res_cnt', res_cnt)

    k = cv2.waitKey(0) & 0xff
    if k == ord('e'):
        exit(0)

    if not p.get_mode():
        return [area_ratio,radius]
        # return [ct[1]]
    

def main():
    global p
    print('>' * 10, 'Get feature from the oil gland', '<' * 10, '\n')

    header = 'b_mean, g_mean, r_mean, gland_ratio, output\n'
    symbol_list = get_symbol_list()
    date = gen_date()
    overwrite = False

    for s in symbol_list:
        out = []
        output = 1
        gland_ratio = []
        for j in date:
            img_name = s + '_' + j + '.JPG'
            print(img_name, 'is processing...\n')
            img = cv2.imread(CONST.IMG_OIL_GLAND_PATH + img_name, 1)

            out.append(output)
            output += 1

            if img is None:
                print('Cannot read image: ', img_name)
                continue

            res = oil_gland_feature(img)

            if res is None:
                print('Cannot get oil gland from the image: ', img_name)
                continue

            gland_ratio.append(res[0:2])

        if not p.get_mode():
            # # gland_ratio[0] = gland_ratio[0] / np.linalg.norm(gland_ratio[0])
            # gland_ratio[1] = gland_ratio[1] / np.linalg.norm(gland_ratio[1])
            # gland_ratio = np.array(gland_ratio)
            # gland_ratio[1] = [1]*len(gland_ratio[1]) - gland_ratio[1]
            plt.plot(out, gland_ratio)        
            # plt.plot(out, gland_ratio)        
            # plt.legend(['1'], loc='upper left')
        plt.legend(['ratio_area','invert_radius_cov'], loc='upper left')

    if not p.get_mode():
        plt.show()


if __name__ == '__main__':
    # p.change_mode(True)
    main()
    pass
