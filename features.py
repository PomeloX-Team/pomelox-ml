import cv2
from lib import *
import numpy as np
import constant as CONST

p = Print(False)
def oil_gland_feature(img):
    global p
    img_filter = clahe(img)
    gray = cv2.cvtColor(img_filter, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)

    p.imshow('img', img)
    p.imshow('img_filter', img_filter)
    p.imshow('img_th', th)
    p.imshow('img_gray', gray)
    k = cv2.waitKey(0) & 0xff
    if k == ord('e'):
        exit(0)
    r, c = th.shape
    no_all = r * c
    no_white = np.count_nonzero(th)
    no_black_ratio = (no_all - no_white) / no_all
    return [no_black_ratio]

def histogram_feature(img):
    global p
    img = cv2.resize(img, (0, 0), fx=0.075, fy=0.075)
    b, g, r = cv2.split(img)
    b = np.mean(b)
    g = np.mean(g)
    r = np.mean(r)
    return [b,g,r]


def main():
    global p
    # symbol = ['A', 'B', 'C', 'D', 'E', 'F']
    symbol = ['A']
    # symbol = ['B']
    # number = range(1,5)
    number = range(4,5)
    error_list = []
    header = 'b_mean, g_mean, r_mean, gland_ratio, output\n'
    date = gen_date()
    for prefix in symbol:
        text = header
        for i in number:
            gland_ratio = []        
            output = 1
            for j in date:
                img_name = prefix + str(i) + '_' + j + '.JPG'
                img = cv2.imread(CONST.IMG_SAVE_PATH + img_name, 1)
                if img is None:
                    continue
                res_gland = oil_gland_feature(img)
                res_hist = histogram_feature(img)
                text += str(res_hist[0])+', '+str(res_hist[1])+', '+str(res_hist[2])+', '+str(res_gland[0])+', '+str(output)+'\n'
                output += 1

                # gland_ratio.append(res_gland[0])                
                # res = str(res[0])+', '+str(res[1])+', '+str(res[2])
            # plt.scatter(no_black_ratio,range(1,len(no_black_ratio)+1,1))
            # plt.show()
        f=open('Pomelo_features_'+prefix,'w')
        f.write(text)
        f.close()

if __name__ == '__main__':
    # p.change_mode(True)
    main()
    pass
