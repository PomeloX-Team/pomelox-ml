import cv2
import numpy as np
from matplotlib import pyplot as plt
from lib import *
import constant as CONST


debug = False
p = Print(debug)


def crop_rect(img):
    global debug
    # imread return bgr color space
    radius = 0
    offset_radius = -5
    center = (0, 0)

    if debug:
        f = 0.075
    else:
        f = 0.075

    img = cv2.resize(img, (0, 0), fx=f, fy=f)

    # img = clahe(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    row, col, _ = img.shape
    crop_w = int(250 * f)
    crop_h = int(250 * f)

    img = img[crop_h:-crop_h, crop_w:-crop_w]
    result = img.copy()
    res_cnt = img.copy()
    r, c, ch = img.shape
    mask = np.uint8(np.zeros((r, c)))

    # convert bgr to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # get h value that appears most often
    h_mode = get_mode(h)
    p.print(h_mode)
    p.print(str(row) + " " + str(col))
    #### Find mask ####

    # find circle from inrange
    lower_bound = np.array([h_mode - 10, 0, 0], dtype=np.uint8)
    upper_bound = np.array([h_mode + 10, 255, 255], dtype=np.uint8)
    res_inrange = cv2.inRange(hsv, lower_bound, upper_bound)
    # Invert color black and white
    res_inrange_inv = 255 - res_inrange

    kernel = get_kernel('\\')
    dilate = cv2.dilate(res_inrange_inv, kernel, iterations=1)
    kernel = get_kernel('/')
    dilate = cv2.dilate(dilate, kernel, iterations=1)
    kernel = get_kernel('rect')
    dilate = cv2.dilate(dilate, kernel, iterations=5)   
    erode = cv2.erode(dilate, kernel, iterations=4)

    # find contour
    # Contour Retrieval Mode -> RETR_TREE
    # It retrieves all the contours and creates a full family hierarchy list
    # cv2.CHAIN_APPROX_NONE, all the boundary points are stored
    _, cnts, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if debug:
        cv2.drawContours(res_cnt, cnts, -1, (255, 155, 155), 2)

    r = 1000
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 70000 or area < 4000:
            continue

        cv2.drawContours(res_cnt, cnt, -1, (0, 0, 155), 2)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x = int(x)
        y = int(y)
        radius = int(radius)
        center = x, y
        width = int(radius // 3)
        x_left = x - width
        y_left = y - width
        x_right = x + width
        y_right = y + width

        radius_min = 52
        radius_max = 105
        radius = radius + offset_radius
        p.print(str(radius_min) + " " +
                    str(radius) + " " + str(radius_max))

        if radius < radius_min or radius > radius_max:
            continue

        if radius < r:
            cv2.rectangle(img, (x_left, y_left),
                          (x_right, y_right), (0, 0, 0), 2)
            mask = cv2.rectangle(mask, (0, 0), (col, row), (0, 0, 0), -1)
            mask = cv2.rectangle(mask, (x_left, y_left),
                                 (x_right, y_right), (255, 255, 255), -1)
            res_cnt = cv2.circle(res_cnt, center, radius, (255, 255, 255), 1)
            r = radius
            # cv2.imshow('mask',mask)
            # cv2.waitKey(0)
        if debug:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(res_cnt, " r:" + str(radius),
                        center, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    result = cv2.bitwise_and(result, result, mask=mask)

    ####################

    
    p.imshow('mask', mask)
    p.imshow('dilate', dilate)
    p.imshow('erode', erode)
    p.imshow('cnt', res_cnt)
    p.imshow('result', result)
    p.imshow('img', img)
    cv2.waitKey(0)
    
    x, y = center
    # width = radius + 2
    result = result[y - width:y + width, x - width:x + width]
    try:
        result = cv2.resize(
            result, (CONST.RESULT_WIDTH, CONST.RESULT_HEIGHT))
    except:
        result = None
    
    if not p.get_mode():
        return result


def main():
    symbol = []
    print('Put number of symbol: ')
    n_symbol = input()
    for i in range(0,int(n_symbol),1):
        print('Put symbol #',i,' : ')
        s = input()
        print('Put order of symbol ex. 1-5: ')
        o = input()
        o1,o2 = o.split('-')
        o1 = int(o1)
        o2 = int(o2)
        for j in range(o1,o2+1,1):
            symbol.append(s+str(j))
    print(symbol)
    error_list = []
    date = gen_date()
    print('Processing...')
    ct = 0
    k = 'y'
    for s in symbol:
        for j in date:
            img_name = s + '_' + j + '.JPG'
            img_name_save = s + j + '.JPG'
            if ct == 0 and not cv2.imread(CONST.IMG_SAVE_PATH + img_name, 1) is None:
                print('There is already file [press N or n] to exit or [any key] to continue.')
                k = input()
                ct += 1
            
            if k == 'N' or k == 'n':
                exit(0)
            print(img_name)
            img = cv2.imread(CONST.IMG_PATH + img_name, 1)
            res = crop_rect(img)
            if res is None:
                error_list.append(img_name)
                continue
            cv2.imwrite(CONST.IMG_SAVE_PATH + img_name, res)
    p.print(error_list)


if __name__ == '__main__':
    # p.change_mode(True)
    main()
    pass
