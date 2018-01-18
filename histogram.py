import cv2
import constant as CONST
from matplotlib import pyplot as plt


def histogram_represent():
    color = [['navy', 'green', 'lime'], ['brown', 'red', 'yellow']]
    imgs = []
    imgs.append(cv2.imread(CONST.IMG_SAVE_PATH + 'A1_20171101.JPG', 1))
    imgs.append(cv2.imread(CONST.IMG_SAVE_PATH + 'A1_20171101_b.JPG', 1))

    for (i, img, col) in zip(range(0, len(imgs), 1), imgs, color):
        img = cv2.resize(img, (0, 0), fx=0.075, fy=0.075)
        b, g, r = cv2.split(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img_' + str(i), gray)
        hist = cv2.calcHist([b], [0], None, [256], [0, 256])
        plt.plot(hist, color=col[0])
        hist = cv2.calcHist([g], [0], None, [256], [0, 256])
        plt.plot(hist, color=col[1])
        hist = cv2.calcHist([r], [0], None, [256], [0, 256])
        plt.plot(hist, color=col[2])
        plt.xlim([0, 256])
    plt.show()
    cv2.waitKey(1)
    pass


if __name__ == '__main__':
    histogram_represent()
