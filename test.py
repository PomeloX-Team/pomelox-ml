import cv2

def test():
    img = cv2.imread('./images/A1_20171101.JPG',1)
    img = img[:,648:-648]
    img = cv2.resize(img,(0,0),fx=0.1,fy=0.1)
    print(img.shape)
    cv2.imshow('img',img)
    img = cv2.imread('./images/A1_20171208.JPG',1)
    print(img.shape)
    cv2.waitKey(0)
    
if __name__=='__main__':
    test()