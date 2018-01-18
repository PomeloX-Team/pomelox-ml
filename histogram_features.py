import numpy as np
from lib import *
import constant as CONST
from matplotlib import pyplot as plt

debug = False
p = Print(debug)



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
    number = range(1, 3)
    error_list = []
    date = gen_date()
    for prefix in symbol:
        for i in number:
            mean_bgr = []
            text = ''
            for j in date:
                img_name = prefix + str(i) + '_' + j + '.JPG'
                img = cv2.imread(CONST.IMG_SAVE_PATH + img_name, 1)
                if img is None:
                    continue
                res = histogram_feature(img)
                res = str(res[0])+','+str(res[1])+','+str(res[2])
                text += str(res)+'\n'
            f=open('histogram_'+prefix+str(i),'w')
            f.write(text)
            f.close()


if __name__ == '__main__':
    main()
    pass

