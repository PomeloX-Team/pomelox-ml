'''
    File name: crop_rect.py
    Author: PomeloX
    Date created: 2/1/2018
    Date last modified: 2/17/2018
    Python Version: 3.6.1
'''

import cv2 as cv
from lib import *
import numpy as np
import constant as CONST
from operator import itemgetter
import statistics

debug = False
p = Print(debug)

def get_mode(channel, min=0, max=255):
    # numpy return a contiguous flattened array.
    data = channel.ravel()
    data = np.array(data)

    if len(data.shape) > 1:
        data = data.ravel()
    try:
        mode = statistics.mode(data)
    except ValueError:
        mode = None
    return mode

def circle_matching(mask_cir, mask, x, y, r):
    found = 0
    for i in range(x - r - 5, x + r + 5):
        for j in range(y - r - 5, y + r + 5):
            if mask_cir[i, j] == mask[i, j] and mask_cir[i, j] == 255:
                found += 1
    return found


def crop_rect(image):
    global p
    result = image.copy()
    row, col, ch = image.shape

    # img_size must divions 600 
    img_size = 100
    ratio = int(CONST.RESULT_WIDTH/img_size)

    centroid = int(img_size / 2)
    r_min = centroid - 25
    r_max = centroid - 15
    centroid_range = range(centroid-10,centroid+10)

    blur = cv.medianBlur(image,5)
    clahe = clahe_by_Lab(blur)
    image = clahe

    img = cv.resize(image,(img_size,img_size))
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    row, col, ch = img.shape

    h, s, v = cv.split(hsv) 
    h_mode = get_mode(h)
    
    lower_bound = np.array([h_mode - 10, 0, 0], dtype=np.uint8)
    upper_bound = np.array([h_mode + 10, 255, 255], dtype=np.uint8)
    res_inrange = cv.inRange(hsv, lower_bound, upper_bound)
    res_inrange_inv = 255 - res_inrange

    kernel = get_kernel('plus',(5,5))
    res_inrange_inv = cv.dilate(res_inrange_inv, kernel, iterations=1)   
    
    result_matching = []
    for x in centroid_range:
        for y in centroid_range:
            for r in range(r_min,r_max+1):
                center = (x,y)
                mask_cir = np.uint8(np.zeros((row, col)))
                cv.circle(mask_cir, center, r, (255), 2)
                number_of_match = circle_matching(mask_cir,res_inrange_inv,x,y,r)
                result_matching.append([number_of_match,center,r])
        print(x)
                
    result_matching = sorted(result_matching, key=itemgetter(0),reverse=True)
    print(result_matching[:10])
    
    x_mean, y_mean, r_mean = [],[],[]
    
    for res in result_matching[:10]:
        _,center,r = res
        x,y = center
        x_mean.append(x)
        y_mean.append(y)
        r_mean.append(r)
    
    x = np.array(x_mean).mean() 
    y = np.array(y_mean).mean() 
    r = np.array(r_mean).mean() 

    res_x = int(x) * ratio 
    res_y = int(y) * ratio
    res_r = int(r) * ratio
    
    w = int(0.65 * res_r)
    img_rect = result[res_y - w:res_y + w, res_x - w:res_x + w]
    img_rect = cv.resize(img_rect, (CONST.RECT_SIZE, CONST.RECT_SIZE))
    
    p.imshow('res_inrange', res_inrange_inv)
    p.imshow('image', result)

    k = cv.waitKey(0)
    if k == ord('e'):
        exit(0)
       
    if not p.get_mode():
        return img_rect


def main():
    print('>' * 10, 'Crop rect on the image(s)', '<' * 10, '\n')
    symbol_list = get_symbol_list()

    date = gen_date()
    overwrite = False

    for s in symbol_list:
        for j in date:
            img_name = s + '_' + j + '.JPG'
            print(img_name, 'is processing...\n')
            img = cv.imread(CONST.IMG_RESIZE_PATH + img_name, 1)
   
            if img is None:
                print('Cannot read image: ', img_name)
                continue

            res = crop_rect(img)

            if res is None:
                print('Cannot draw circle on image: ', img_name)
                continue

            if (not overwrite) and (not cv.imread(CONST.IMG_CIRCLE_PATH + img_name, 1) is None):
                print('There is already a file with the same name\nPress s or S to Skip\nPress o or O to Overwrite\nPress a or A to Overwrite to all\nPress e or E to exit program\n')
                k = input()
                k = k.lower()
                if k == 's':
                    continue
                elif k == 'a':
                    overwrite = True
                elif k == 'e':
                    break

            cv.imwrite(CONST.IMG_CIRCLE_PATH + img_name, res)

if __name__ == '__main__':
    # p.change_mode(True)
    main()
    pass