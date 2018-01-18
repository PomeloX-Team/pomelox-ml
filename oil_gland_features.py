import numpy as np
from lib import *
import constant as CONST
from matplotlib import pyplot as plt

debug = False
p = Print(debug)

def pre_processing(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_mode = get_mode(v, 10, 254)
    v_mean = cv2.mean(v)[0]
    return v_mode, v_mean


def oil_gland_features(img):
    global p
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_equ = equalization_gray(gray)
    mode = get_mode(gray_equ, 30, 250)
    _, th = cv2.threshold(gray_equ, 70, 255, cv2.THRESH_BINARY)

    p.imshow('img', img)
    p.imshow('img_th', th)
    p.imshow('img_gray', gray)
    p.imshow('img_gray_equ', gray_equ)
    k = cv2.waitKey(0) & 0xff
    if k == ord('e'):
        exit(0)
    r, c = th.shape
    no_all = r * c
    no_white = np.count_nonzero(th)
    no_black_ratio = (no_all - no_white) / no_all
    return no_black_ratio


def main():
    global p
    # symbol = ['A', 'B', 'C', 'D', 'E', 'F']
    symbol = ['A']
    number = range(1, 2)
    error_list = []
    date = gen_date()
    no_black_ratio = []
    for prefix in symbol:
        for i in number:
            for j in date:
                img_name = prefix + str(i) + '_' + j + '.JPG'
                img = cv2.imread(CONST.IMG_SAVE_PATH + img_name, 1)
                if img is None:
                    continue
                res = oil_gland_features(img)
                no_black_ratio.append(res)
    plt.scatter(no_black_ratio,range(1,len(no_black_ratio)+1,1))
    plt.show()


if __name__ == '__main__':
    # p.change_mode(True)
    main()
    pass
