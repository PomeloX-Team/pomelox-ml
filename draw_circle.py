'''
    File name: draw_circle.py
    Author: PomeloX
    Date created: 2/1/2018
    Date last modified: 2/17/2018
    Python Version: 3.6.1
'''

import cv2
from lib import *
import numpy as np
import constant as CONST
from operator import itemgetter

debug = False
p = Print(debug)

def circle_matching(mask_cir,mask,x,y,r):
    found = 0
    for i in range(x-r-5,x+r+5):
        for j in range(y-r-5,y+r+5):
            if mask_cir[i,j] == mask[i,j] and mask_cir[i,j] == 255:
                found += 1
    return found

def draw_circle(image):
    global p
    result = image.copy()
    row, col, ch = image.shape

    img_size = 100
    centroid = int(img_size / 2)
    r_min = 25
    r_max = 35
    centroid_range = range(centroid-5,centroid+5)

    blur = cv2.medianBlur(image,3)
    clahe = clahe_by_Lab(blur)
    image = clahe

    img = cv2.resize(image,(img_size,img_size))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    row, col, ch = img.shape

    h, s, v = cv2.split(hsv) 
    h_mode = get_mode(h)
    
    lower_bound = np.array([h_mode - 12, 0, 0], dtype=np.uint8)
    upper_bound = np.array([h_mode + 12, 255, 255], dtype=np.uint8)
    res_inrange = cv2.inRange(hsv, lower_bound, upper_bound)
    res_inrange_inv = 255 - res_inrange

    kernel = get_kernel('rect',(3,3))
    res_inrange_inv = cv2.dilate(res_inrange_inv, kernel, iterations=1)   
    
    result_matching = []
    for x in centroid_range:
        for y in centroid_range:
            for r in range(r_min,r_max+1):
                center = (x,y)
                mask_cir = np.uint8(np.zeros((row, col)))
                cv2.circle(mask_cir, center, r, (255), 2)
                number_of_match = circle_matching(mask_cir,res_inrange_inv,x,y,r)
                result_matching.append([number_of_match,center,r])
        print(x)
                
    result_matching = sorted(result_matching, key=itemgetter(0),reverse=True)
    print(result_matching[:5])
    
    x_mean, y_mean, r_mean = [],[],[]
    for res in result_matching[:5]:
        _,center,r = res
        x,y = center
        cv2.circle(result, (x*6,y*6), r*6, (255, 255, 255), 6)
        cv2.circle(res_inrange_inv, center, r, (155, 155, 155), 2)

    p.imshow('res_inrange', res_inrange_inv)
    p.imshow('image', result)

    k = cv2.waitKey(0)
    if k == ord('e'):
        exit(0)
       
    if not p.get_mode():
        return result


def main():
    print('>' * 10, 'Draw circle on the image(s)', '<' * 10, '\n')
    symbol_list = get_symbol_list()

    date = gen_date()
    overwrite = False

    for s in symbol_list:
        for j in date:
            img_name = s + '_' + j + '.JPG'
            print(img_name, 'is processing...\n')
            img = cv2.imread(CONST.IMG_RESIZE_PATH + img_name, 1)

            if img is None:
                print('Cannot read image: ', img_name)
                continue

            res = draw_circle(img)

            if res is None:
                print('Cannot draw circle on image: ', img_name)
                continue

            if (not overwrite) and (not cv2.imread(CONST.IMG_CIRCLE_PATH + img_name, 1) is None):
                print('There is already a file with the same name\nPress s or S to Skip\nPress o or O to Overwrite\nPress a or A to Overwrite to all\nPress e or E to exit program\n')
                k = input()
                k = k.lower()
                if k == 's':
                    continue
                elif k == 'a':
                    overwrite = True
                elif k == 'e':
                    break

            cv2.imwrite(CONST.IMG_CIRCLE_PATH + img_name, res)

if __name__ == '__main__':
    # p.change_mode(True)
    main()
    pass
