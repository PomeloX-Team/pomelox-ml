'''
    File name: resize.py
    Author: PomeloX
    Date created: 2/1/2018
    Date last modified: 2/17/2018
    Python Version: 3.6.1
'''

import cv2
from lib import *
import constant as CONST


def resize(img):
    row, col, _ = img.shape
    if col > 3900:
        img = img[:, 648:-648]
    img = cv2.resize(img, (CONST.RESULT_WIDTH, CONST.RESULT_HEIGHT))
    return img


if __name__ == '__main__':
    symbol_list = []
    print('>' * 10, 'Reize and crop image(s) to square image(s)', '<' * 10, '\n')
    symbol_list = get_symbol_list()

    date = gen_date()
    overwrite = False

    for s in symbol_list:
        for j in date:
            img_name = s + '_' + j + '.JPG'
            print(img_name, 'is processing...\n')
            img = cv2.imread(CONST.IMG_PATH + img_name, 1)

            if img is None:
                print('Cannot read image: ', img_name)
                continue

            res = resize(img)

            if res is None:
                print('Cannot resize image: ', img_name)
                continue

            if (not overwrite) and (not cv2.imread(CONST.IMG_RESIZE_PATH + img_name, 1) is None):
                print('There is already a file with the same name\nPress s or S to Skip\nPress o or O to Overwrite\nPress a or A to Overwrite to all\nPress e or E to exit program\n')
                k = input()
                k = k.lower()
                if k == 's':
                    continue
                elif k == 'a':
                    overwrite = True
                elif k == 'e':
                    break

            cv2.imwrite(CONST.IMG_RESIZE_PATH + img_name, res)
