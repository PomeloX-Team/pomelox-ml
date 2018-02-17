import cv2
from lib import *
import numpy as np
import constant as CONST
import matplotlib.pyplot as plt

p = Print(False)


def oil_gland_feature(img):
    global p

    res_cnt = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, cnts, _ = cv2.findContours(
        th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(res_cnt, cnts, -1, (0, 0, 255), 1)

    ct = 0
    r_mean = []
    for cnt in cnts:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        x = int(x)
        y = int(y)
        r = int(r)
        r_mean.append(r)
        cv2.circle(res_cnt, (x, y), r, (0, 255, 0), -1)

    r_mean = np.array(r_mean).mean()

    p.imshow('img_th', gray)
    p.imshow('res_cnt', res_cnt)

    k = cv2.waitKey(0) & 0xff
    if k == ord('e'):
        exit(0)

    if not p.get_mode():
        return [r_mean]


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

            gland_ratio.append(res[0])

        if not p.get_mode():
            plt.plot(out, gland_ratio)
            plt.show()


if __name__ == '__main__':
    # p.change_mode(True)
    main()
    pass
