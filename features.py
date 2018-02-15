import cv2
from lib import *
import numpy as np
import constant as CONST
import matplotlib.pyplot as plt
import math
p = Print(False)
def oil_gland_feature(img):
    global p

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    res_cnt1 = img.copy()
    res_cnt = img.copy()
    _, cnts, _ = cv2.findContours(
        th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(res_cnt1, cnts, -1, (0, 0, 155), 1)
    
    # circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=200,param2=50,minRadius=0,maxRadius=0)
    
    # circles = np.uint16(np.around(circles))
    # for i in circles[0,:]:
    #     # draw the outer circle
    #     cv2.circle(res_cnt,(i[0],i[1]),i[2],(0,255,0),2)
    #     # draw the center of the circle
    #     cv2.circle(res_cnt,(i[0],i[1]),2,(0,0,255),3)

    ct = 0
    for cnt in cnts:

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if not (2 < r < 15):
            continue
        x = int(x)
        y = int(y)
        r = int(r)
        
        ct += 1
        cv2.circle(res_cnt, (x, y), r, (0, 0, 255), 1)


    r, c = gray.shape
    no_all = r * c
    no_white = np.count_nonzero(gray)
    no_black_ratio = no_white / (math.pi*200*200)
    no_black_ratio = ct
    # no_black_ratio = len(cnts)
    p.imshow('img_th', gray)
    p.imshow('res_cnt', res_cnt)
    p.imshow('res_cnt111', res_cnt1)
    k = cv2.waitKey(0) & 0xff
    if k == ord('e'):
        exit(0)

    # if not p.get_mode():
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
    # number = range(1,2)
    # number = range(4,5)
    number = range(1,5)
    error_list = []
    header = 'b_mean, g_mean, r_mean, gland_ratio, output\n'
    date = gen_date()
    for prefix in symbol:
        text = header
        for i in number:
            gland_ratio = []        
            out = []
            output = 1
            for j in date:
                img_name = prefix + str(i) + '_' + j + '.JPG'
                img = cv2.imread(CONST.IMG_SAVE_PATH + img_name, 1)
                out.append(output)
                output += 1
                if img is None:
                    gland_ratio.append(gland_ratio[-1])                
                    continue
                res_gland = oil_gland_feature(img)
                gland_ratio.append(res_gland[0])                
                res_hist = histogram_feature(img)
                if p.get_mode():
                    continue
                
                # text += str(res_hist[0])+', '+str(res_hist[1])+', '+str(res_hist[2])+', '+str(res_gland[0])+', '+str(output)+'\n'
                
                # text += str(res_hist[1]-res_hist[0])+', '+str(res_hist[1]-res_hist[2])+', '+str(res_hist[2]-res_hist[0])+', '+str(res_gland[0])+', '+str(output)+'\n'
                # gland_ratio.append(res_gland[0])

                # res = str(res[0])+', '+str(res[1])+', '+str(res[2])
            # cv2.destroyAllWindows()
            # plt.scatter(gland_ratio,out)
            plt.plot(gland_ratio,out)
        plt.show()
        if not p.get_mode():
            continue
        # f=open('Pomelo_features_'+prefix+'_4_new','w')
        f=open('Pomelo_features_'+prefix+'_1_3_new','w')
        f.write(text)
        f.close()

if __name__ == '__main__':
    p.change_mode(True)
    main()
    pass
