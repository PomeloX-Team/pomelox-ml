import cv2
import statistics
import numpy as np


def gen_date():
    date = []
    for i in range(1, 31, 1):
        str_date = '201711' + str(i).zfill(2)
        date.append(str_date)

    for i in range(1, 26, 1):
        str_date = '201712' + str(i).zfill(2)
        date.append(str_date)
    return date


class Print:
    def __init__(self, debug_mode):
        self.debug_mode = debug_mode

    def print(self, string):
        if self.debug_mode:
            print(str(string))

    def imshow(self, winname, mat):
        if self.debug_mode:
            cv2.imshow(winname, mat)

    def change_mode(self, mode):
        self.debug_mode = mode


def get_mode(channel, min=0, max=255):
    # numpy return a contiguous flattened array.
    data = channel.ravel()
    data = np.array(data)

    if len(data.shape) > 1:
        data = data.ravel()
    try:
        mode = statistics.mode(data)
    except ValueError:
        mode = None
    return mode


def get_kernel(shape='rect', ksize=(5, 5)):
    if shape == 'rect':
        return cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    elif shape == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
    elif shape == 'plus':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)
    elif shape == '\\':
        kernel = np.diag([1] * ksize[0])
        return np.uint8(kernel)
    elif shape == '/':
        kernel = np.fliplr(np.diag([1] * ksize[0]))
        return np.uint8(kernel)
    else:
        return None


def cut_contours(M, w, h, range_w, range_h):
    cx = None
    cy = None
    try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    except:
        print('err')
    if cx is None:
        return True
    if cx <= range_w or cy <= range_h or cx >= w - range_w or cy >= h - range_h:
        return True
    return False


def brightness(imgBGR, brightnessValue):
    hsv = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.uint16(v)
    v = np.clip(v + brightnessValue, 0, 255)
    v = np.uint8(v)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def brightness_gray(imgGRAY, brightnessValue):
    bgr = cv2.cvtColor(imgGRAY, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.uint16(v)
    v = np.clip(v + brightnessValue, 0, 255)
    v = np.uint8(v)
    hsv = cv2.merge((h, s, v))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def clip_v(imgBGR, min, max):
    hsv = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v, min, max)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def equalization_bgr(imgBGR):
    b, g, r = cv2.split(imgBGR)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    equBGR = cv2.merge((b, g, r))
    return equBGR


def equalization_hsv(imgHSV):
    h, s, v = cv2.split(imgHSV)
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)
    equHSV = cv2.merge((h, s, v))
    return equHSV


def equalization_gray(imgGRAY):
    equGRAY = cv2.equalizeHist(imgGRAY)

    return equGRAY


def clahe(imgBGR):
    lab = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    resBGR = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return resBGR


def stretching_hsv(hsv):
    h, s, v = cv2.split(hsv)

    if s.min() > 0:
        s *= int(round(255.0 / (s.max() - s.min())))
    if v.min() > 0:
        v *= int(round(255.0 / (v.max() - v.min())))
    hsv = cv2.merge((h, s, v))
    return hsv


def stretching_bgr(bgr):
    b, g, r = cv2.split(bgr)
    b -= b.min()
    b *= int(round(255.0 / (b.max() - b.min())))
    g -= g.min()
    g *= int(round(255.0 / (g.max() - g.min())))
    r -= r.min()
    r *= int(round(255.0 / (r.max() - r.min())))

    img = cv2.merge((b, g, r))
    return img


def stretching(img):

    b, g, r = cv2.split(img)
    b -= b.min()
    b *= int(round(255.0 / (b.max() - b.min())))
    g -= g.min()
    g *= int(round(255.0 / (g.max() - g.min())))
    r -= r.min()
    r *= int(round(255.0 / (r.max() - r.min())))

    img = cv2.merge((b, g, r))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)

    s -= s.min()
    s *= int(round(255.0 / (s.max() - s.min())))

    v -= v.min()
    v *= int(round(255.0 / (v.max() - v.min())))

    img = cv2.merge((h, s, v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img
