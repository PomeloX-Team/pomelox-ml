import cv2
from lib import *
import constant as CONST
import numpy as np
from matplotlib import pyplot as plt


def preprocessing_images(img):
    img = clahe(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_equ = equalization_gray(gray)
    hist = cv2.calcHist([gray_equ], [0], None,[256], [0, 256])
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    b,g,r = cv2.split(img)
    l = np.array([np.mean(h)-20,np.mean(s)-50,np.mean(v)-50],dtype=np.uint8)
    u = np.array([np.mean(h)+20,np.mean(s)+50,np.mean(v)+50],dtype=np.uint8)
    mask = cv2.inRange(hsv,l,u)
    # th = cv2.threshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)
    mean = np.mean(gray_equ)
    print(mean)
    print(np.mean(h),np.mean(s),np.mean(v))
    _, th = cv2.threshold(gray_equ,mean,255,cv2.THRESH_BINARY)
    cv2.imshow('gray', gray)
    cv2.imshow('th', th)
    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)
    cv2.imshow('mask', mask)
    cv2.imshow('img', img)
    cv2.imshow('equ', gray_equ)
    # cv2.imshow('img clahe', clahe_img)
    plt.plot(hist,color='blue')
    # plt.show()
    cv2.waitKey(0)


if __name__ == '__main__':
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
            img = cv2.imread(CONST.IMG_RESIZE_PATH + img_name, 1)

            preprocessing_images(img)
