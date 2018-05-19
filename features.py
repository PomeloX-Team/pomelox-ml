'''
    File name: features.py
    Author: PomeloX
    Date created: 1/22/2018
    Date last modified: 2/17/2018
    Python Version: 3.6.1
'''

import cv2 as cv
from lib import *
import numpy as np
import constant as CONST
import matplotlib.pyplot as plt
import math
from scipy import stats

p = Print(False)


def trimmed_mean(data, percent=0.1):
    data = np.array(data)
    data = sorted(data)
    trim_mean = stats.trim_mean(data, percent)
    print(trim_mean)
    trim_mean = math.floor(trim_mean)
    if trim_mean > 0:
        data = data[trim_mean:-trim_mean]
    print(data)
    return data

# radius, number of gland,


def radius_feature(img,img_name):
    global p
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    row, col = gray.shape
    area = float(row*col)
    res_cnt = img.copy()

    _, th = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    _, cnts, _ = cv.findContours(
        th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    res = []
    for cnt in cnts:
        (x, y), r = cv.minEnclosingCircle(cnt)
        res.append(int(round(r)))

    res = sorted(res)

    return np.mean(res)

def radius_interval_feature(img,img_name):
    global p
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    row, col = gray.shape
    area = float(row*col)
    res_cnt = img.copy()

    _, th = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    _, cnts, _ = cv.findContours(
        th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    res = []
    for cnt in cnts:
        (x, y), r = cv.minEnclosingCircle(cnt)
        res.append(int(round(r)))

    res = sorted(res)

    plt.title('radius '+img_name)
    plt.xlabel('radius (px)')
    plt.ylabel('frequency')
    # plt.plot(res)
    n, bins, patches = plt.hist(res,None,[0,20])
    print(n,bins)
    plt.savefig(CONST.GRAPH_PATH + img_name + '.png')
    plt.close()

    # return np.mean(res)
    return np.count_nonzero(n)

def gland_from_center(img):
    global p
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    row, col = gray.shape
    area = float(row*col)
    res_cnt = img.copy()

    _, th = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    _, cnts, _ = cv.findContours(
        th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    res = []
    for cnt in cnts:
        (x, y), r = cv.minEnclosingCircle(cnt)
        res.append(r)

    return np.mean(res)

def number_of_gland_feature(img):
    global p
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    row, col = gray.shape
    area = float(row*col)
    res_cnt = img.copy()

    _, th = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    _, cnts, _ = cv.findContours(
        th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    return len(cnts)


def oil_gland_feature(img, feature_name='radius', img_name=None):
    global p

    if feature_name == 'radius':
        return radius_feature(img, img_name)
    elif feature_name == 'number of gland':
        return number_of_gland_feature(img)
    elif feature_name == 'radius_interval_feature':
        return radius_interval_feature(img, img_name)
    else:
        return None

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # res_cnt = img.copy()

    # _,th = cv.threshold(gray,127,255,cv.THRESH_BINARY)
    # _, cnts, _ = cv.findContours(
    #     th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # cv.drawContours(res_cnt, cnts, -1, (0, 0, 255), 1)

    # area_ratio = []
    # radius = []
    # centroid = []
    # const_area_ratio = 1
    # res = []
    # for cnt in cnts:
    #     (x, y), r = cv.minEnclosingCircle(cnt)
    #     if r < 3:
    #         continue
    #     cnt_area = cv.contourArea(cnt)
    #     cir_area = math.pi * r * r

    #     if cnt_area / (cir_area * 1.0) < 0.4:
    #         continue
    #     # ratio = (cnt_area / cir_area) * const_area_ratio
    #     # area_ratio.append(ratio)
    #     x = int(x)
    #     y = int(y)
    #     r = int(r)
    #     res.append([r,x, y,cnt_area,cir_area])

    #     cv.circle(res_cnt, (x, y), r+2, (0, 255, 0), 1)

    # dis_list = []
    # r_list = []

    # for data in res:
    #     r,x, y,cnt_area,cir_area = data
    #     dis = math.sqrt((x-CONST.RECT_SIZE/2)**2+(y-CONST.RECT_SIZE/2)**2)
    #     dis_list.append(dis)
    #     r_list.append(r)
    # dis_list = np.array(dis_list).mean()
    # r_list = np.array(r_list).mean()
    # p.imshow('res_cnt', res_cnt)
    # p.imshow('th', th)
    # density = np.count_nonzero(th)/((CONST.RECT_SIZE*CONST.RECT_SIZE)*1.0)
    # k = cv.waitKey(-1) & 0xff
    # if k == ord('e'):
    #     exit(0)
    # ratio = np.mean(r_list) / np.mean(dis_list)
    # if not p.get_mode():
    #     return [density,ratio,r_list]


def main():
    global p
    print('>' * 10, 'Get feature from the oil gland', '<' * 10, '\n')

    symbol_list = get_symbol_list()
    date = gen_date()
    # overwrite = False
    # header = 'area_ratio, radius_covariance, output\n'

    feature_names = ['radius', 'number of gland','radius_interval_feature']

    for feature_name in feature_names:
        header = 'feature_name, output\n'
        text = header
        for s in symbol_list:
            out = []
            output = 1
            features = []
            for j in date:
                img_name = s + '_' + j + '.JPG'
                print(img_name, 'is processing...\n')
                img = cv.imread(CONST.IMG_OIL_GLAND_PATH + img_name, 1)

                if img is None:
                    print('Cannot read image: ', img_name)
                    continue

                res = oil_gland_feature(img, feature_name, s + '_' + j)

                if res is None:
                    print('Cannot get oil gland from the image: ', img_name)
                    continue

                text += str(res) + ', ' + str(output) + '\n'
                features.append(res)
                out.append(output)
                output += 1

            if not p.get_mode():
                plt.title(feature_name + '_' + s)
                plt.xlabel('time (days)')
                plt.ylabel(feature_name)
                plt.plot(out, features)
                plt.savefig(CONST.GRAPH_PATH + feature_name + '_' + s + '.png')
                plt.close()

            f = open(CONST.CSV_PATH + 'pomelo_features_' +
                     feature_name + '_' + s + '.csv', 'w')
            f.write(text)
            f.close()


if __name__ == '__main__':
    # p.change_mode(True)
    main()
    pass
