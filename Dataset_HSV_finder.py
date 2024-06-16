import cv2
import numpy as np

def nothing(*arg):
    pass

cv2.namedWindow( "result" ) 
cv2.namedWindow( "settings" )


cv2.createTrackbar('h1', 'settings', 0, 180, nothing)
cv2.createTrackbar('s1', 'settings', 0, 255, nothing)
cv2.createTrackbar('v1', 'settings', 0, 255, nothing)
cv2.createTrackbar('h2', 'settings', 180, 180, nothing)
cv2.createTrackbar('s2', 'settings', 255, 255, nothing)
cv2.createTrackbar('v2', 'settings', 255, 255, nothing)

while True:
    img = cv2.imread('000.jpg')
    h,w,_=img.shape
    img=cv2.resize(img,(w//5,h//5))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
 
    # считываем значения бегунков
    h1 = cv2.getTrackbarPos('h1', 'settings')
    s1 = cv2.getTrackbarPos('s1', 'settings')
    v1 = cv2.getTrackbarPos('v1', 'settings')
    h2 = cv2.getTrackbarPos('h2', 'settings')
    s2 = cv2.getTrackbarPos('s2', 'settings')
    v2 = cv2.getTrackbarPos('v2', 'settings')
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)
    img_bin = cv2.inRange(hsv, h_min, h_max)
    cv2.imshow('result', img_bin)
    cv2.imshow('original', img)
    ch = cv2.waitKey(5)
    if ch == 27:
        break
cv2.destroyAllWindows()
