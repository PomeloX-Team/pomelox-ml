import cv2
import numpy as np
from matplotlib import pyplot as plt
from lib import *
import constant as CONST
from operator import itemgetter


debug = False
p = Print(debug)


def crop_cir(img,img_resize):
    global p

    row, col, _ = img.shape

    mask = np.uint8(np.zeros((row, col)))
    res_cnt = img.copy()

    # find circle from inrange
    lower_bound = np.array([240, 0, 240], dtype=np.uint8)
    upper_bound = np.array([255, 15, 255], dtype=np.uint8)
    res_inrange = cv2.inRange(img, lower_bound, upper_bound)
    kernel = get_kernel('rect', (3, 3))
    res_inrange = cv2.dilate(res_inrange, kernel, iterations=2)
    _, cnts, _ = cv2.findContours(
        res_inrange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    circles = []

    for cnt in cnts:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        x = int(x)
        y = int(y)
        r = int(r)
        r_min = 100

        if r < r_min:
            continue
        r = int(r * 0.95)

        cv2.drawContours(res_cnt, cnt, -1, (0, 0, 155), 2)
        cv2.circle(res_cnt, (x, y), r, (0, 0, 255), 2)

        circles.append([x, y, r])

    if len(circles) <= 0:
        return None

    circles = sorted(circles, key=itemgetter(2))

    res_x, res_y, res_r = circles[0]

    mask = cv2.rectangle(mask, (0, 0), (col, row), (0), -1)
    mask = cv2.circle(mask, (res_x, res_y), res_r, (255), -1)

    # blur = cv2.GaussianBlur(img_resize,(3,3),0)
    blur = cv2.medianBlur(img_resize,3)
    # clahe = clahe_by_hsv(blur)
    clahe = clahe_by_Lab(blur)

    gray = cv2.cvtColor(clahe, cv2.COLOR_BGR2GRAY)
    gray = 255-gray
    gray1 = clahe_gray(gray)
    gray_mean = np.array(gray1).mean()
    gray1[gray1<gray_mean] = 0
    equ = equalization_gray(gray1)
    # result = cv2.adaptiveThreshold(
    #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _,result = cv2.threshold(equ,gray_mean*1.2,255,cv2.THRESH_BINARY)
    # result = cv2.morphologyEx(result,cv2.MORPH_OPEN,get_kernel(3,3))

    
    result = cv2.bitwise_and(result, result, mask=mask)
    result = result[res_y - res_r:res_y + res_r, res_x - res_r:res_x + res_r]
    result = cv2.resize(result, (CONST.RESULT_WIDTH, CONST.RESULT_HEIGHT))

    p.imshow('gray', gray)    
    p.imshow('gray1', gray1)    
    p.imshow('grayequ', equ)    
    # p.imshow('blur1', blur1)    
    p.imshow('blur2', blur)    
    # p.imshow('blur3', blur3)    
    p.imshow('res_inrange', res_inrange)
    p.imshow('cnt', res_cnt)
    p.imshow('result', result)
    p.imshow('clahe', clahe)

    k = cv2.waitKey(0)
    if k == ord('e'):
        exit(0)

    if not p.get_mode():
        return result


def main():
    global p
    symbol = []
    print('Put number of symbol: ')
    n_symbol = input()
    for i in range(0, int(n_symbol), 1):
        print('Put symbol #', i, ' : ')
        s = input()
        print('Put order of symbol ex. 1-5: ')
        o = input()
        o1, o2 = o.split('-')
        o1 = int(o1)
        o2 = int(o2)
        for j in range(o1, o2 + 1, 1):
            symbol.append(s + str(j))
    print(symbol)
    error_list = []
    date = gen_date()
    print('Processing...')
    ct = 0
    k = 'y'
    for s in symbol:
        for j in date:
            img_name = s + '_' + j + '.JPG'

            if ct == 0 and not cv2.imread(CONST.IMG_SAVE_PATH + img_name, 1) is None:
                print(
                    'There is already file [press N or n] to exit or [any key] to continue.')
                k = input()
                ct += 1

            if k == 'N' or k == 'n':
                exit(0)

            print(img_name)
            img = cv2.imread(CONST.IMG_CIRCLE_PATH + img_name, 1)
            img_resize = cv2.imread(CONST.IMG_RESIZE_PATH + img_name, 1)
            res = crop_cir(img,img_resize)

            if res is None:
                error_list.append(img_name)
                continue
            cv2.imwrite(CONST.IMG_SAVE_PATH + img_name, res)
    p.print(error_list)


if __name__ == '__main__':
    # p.change_mode(True)
    main()
    pass
