import cv2
from lib import *
import constant as CONST

def resize(img):
    r,c,_ = img.shape
    if c > 3900:
        img = img[:,648:-648]
    print(img.shape)
    img = cv2.resize(img,(600,600))
    return img
    
if __name__=='__main__':
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
            img = cv2.imread(CONST.IMG_PATH + img_name, 1)

            res = resize(img)
            if ct == 0 and not cv2.imread(CONST.IMG_RESIZE_PATH + img_name, 1) is None:
                print('There is already file [press N or n] to exit or [any key] to continue.')
                k = input()
                ct += 1
            cv2.imwrite(CONST.IMG_RESIZE_PATH + img_name, res)