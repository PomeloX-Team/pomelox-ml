import cv2
import numpy as np
from matplotlib import pyplot as plt
from lib import *
import constant as CONST
from operator import itemgetter

debug = False
p = Print(debug)

def circle_matching(mask_cir,mask,x,y,r):
    global p
    found = 0
    for i in range(x-r-5,x+r+5):
        for j in range(y-r-5,y+r+5):
            if mask_cir[i,j] == mask[i,j] and mask_cir[i,j] == 255:
                found += 1
    return found

def draw_circle(image):
    global p
    image_default = image.copy()
    img_size = 300
    centroid = int(img_size / 2)
    r_min = 80
    r_max = 100
    centroid_range = range(centroid-15,centroid+15)

    blur = cv2.medianBlur(image,7)
    clahe = clahe_by_Lab(blur)
    image = clahe

    result = image.copy()
    row, col, ch = image.shape


    img = cv2.resize(image,(img_size,img_size))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    result_s = img.copy()
    row, col, ch = img.shape
    mask = np.uint8(np.zeros((row, col)))

    h, s, v = cv2.split(hsv) 

    
    h_mode = get_mode(h)
    print(h_mode)
    if h_mode is None:
        h_mode = 37
    print(h_mode)
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
                mask_cir = mask.copy()
                cv2.circle(mask_cir, center, r, (255), 4)
                number_of_match = circle_matching(mask_cir,res_inrange_inv,x,y,r)
                result_matching.append([number_of_match,center,r])
        print(x)
                
    result_matching = sorted(result_matching, key=itemgetter(0),reverse=True)
    print(result_matching[:1])
    x_mean, y_mean, r_mean = [],[],[]
    for res in result_matching[:1]:
        _,center,r = res
        x,y = center
        x_mean.append(x)
        y_mean.append(y)
        r_mean.append(r)
        cv2.circle(result_s, center, r, (255, 0, 255), 1)
        cv2.circle(image_default, (x*2,y*2), r*2, (255, 0, 255), 3)
        cv2.circle(res_inrange_inv, center, r, (155, 155, 155), 1)



    p.imshow('res_inrange', res_inrange_inv)
    p.imshow('result', result)
    p.imshow('mask', mask)
    p.imshow('image', image)
    # p.imshow('img_hsv', hsv)
    k = cv2.waitKey(0)
    if k == ord('e'):
        exit(0)
   
    
    if not p.get_mode():
        return image_default


def main():
    global p
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
            img_name = 'A1_20171112.JPG'
            if ct == 0 and not cv2.imread(CONST.IMG_CIRCLE_PATH + img_name, 1) is None:
                print('There is already file [press N or n] to exit or [any key] to continue.')
                k = input()
                ct += 1
            
            if k == 'N' or k == 'n':
                exit(0)
            print(img_name)
            img = cv2.imread(CONST.IMG_RESIZE_PATH + img_name, 1)
            res = draw_circle(img)

            # if not p.get_mode():
            #     continue

            if res is None:
                error_list.append(img_name)
                continue
            cv2.imwrite(CONST.IMG_CIRCLE_PATH + img_name, res)
    p.print(error_list)


if __name__ == '__main__':
    # p.change_mode(True)
    main()
    pass
