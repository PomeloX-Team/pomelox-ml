import cv2
import numpy as np
from matplotlib import pyplot as plt
from lib import *
import constant as CONST
from operator import itemgetter

debug = False
p = Print(debug)

def circle_matching(mask_cir,mask):
    global p
    r,c = mask_cir.shape
    found = 0

    for i in range(r):
        for j in range(c):
            if mask_cir[i,j] == mask[i,j] and mask_cir[i,j] == 255:
                found += 1
        # p.print(found)
    return found

def crop_rect(img):
    global p
    mul = 2
    img = cv2.resize(img,(100,100))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    result = img.copy()
    row, col, ch = img.shape
    mask = np.uint8(np.zeros((row, col)))

    h, s, v = cv2.split(hsv) 
   


    h_mode = get_mode(h)
    lower_bound = np.array([h_mode - 12, 0, 0], dtype=np.uint8)
    upper_bound = np.array([h_mode + 12, 255, 255], dtype=np.uint8)
    res_inrange = cv2.inRange(hsv, lower_bound, upper_bound)
    res_inrange_inv = 255 - res_inrange
    kernel = get_kernel('rect',(3,3))
    res_inrange_inv = cv2.dilate(res_inrange_inv, kernel, iterations=1)   
    # center 75,75 - 125,125
    # radius 50 - 85
    
    result_matching = []
   
    
    for x in range (45,55):
        for y in range(45,55):
            for r in range(28,37):
                center = (y,x)
                mask_cir = mask.copy()
                img_cp = res_inrange_inv.copy()
                cv2.circle(mask_cir, center, r, (255), 3)
                cv2.circle(img_cp, center, r, (150), 3)
                number_of_match = circle_matching(mask_cir,res_inrange_inv)
                result_matching.append([number_of_match,center,r])
                # print(result_matching[-1])
                # p.imshow('mask',img_cp)
                # cv2.waitKey(-1) 
                
        print(x)
    result_matching = sorted(result_matching, key=itemgetter(0),reverse=True)
    print(result_matching[:15])
    for res in result_matching[:15]:
        _,center,r = res
        cv2.circle(result, center, r, (255, 0, 255), 1)
        cv2.circle(res_inrange_inv, center, r, (155, 155, 155), 1)
    p.imshow('img',img)
    p.imshow('res',result)
    p.imshow('inrange',res_inrange_inv)
    cv2.waitKey(-1)


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

            if ct == 0 and not cv2.imread(CONST.IMG_SAVE_PATH + img_name, 1) is None:
                print('There is already file [press N or n] to exit or [any key] to continue.')
                k = input()
            ct += 1
            
            # if k == 'N' or k == 'n':
            #     exit(0)
            print(img_name)
            img = cv2.imread(CONST.IMG_RESIZE_PATH + img_name, 1)
            crop_rect(img)
            # cv2.imwrite(CONST.IMG_SAVE_PATH + img_name, res)


if __name__ == '__main__':
    p.change_mode(True)
    main()
    pass
