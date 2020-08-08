import numpy as np
import cv2 as cv

def nothing(x):
    pass

cv.namedWindow('Trackbars')
cv.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# STARTING THE VIDEO CAPTURE#

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Cam\'s not working ')
    exit()

# for selecting skin value in roi:
hist_flag = 0
kernel = np.ones((3,3), dtype=np.uint8)

while True:

    ret, frame = cap.read()
    if not ret:
        print('Can\'t recieve frame, try again later maybe')
        break

    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    l_h = cv.getTrackbarPos("L - H", "Trackbars")
    l_s = cv.getTrackbarPos("L - S", "Trackbars")
    l_v = cv.getTrackbarPos("L - V", "Trackbars")
    u_h = cv.getTrackbarPos("U - H", "Trackbars")
    u_s = cv.getTrackbarPos("U - S", "Trackbars")
    u_v = cv.getTrackbarPos("U - V", "Trackbars")
    
    lower_col = np.array([l_h,l_s,l_v])
    upper_col = np.array([u_h,u_s,u_v])
    
    mask = cv.inRange(hsv, lower_col,  upper_col)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    masked_img = cv.bitwise_and(frame, frame, mask = mask)
    cv.imshow('Trackbars',masked_img)
    
    k = cv.waitKey(1) & 0xff
    if k == ord('q'):
        break

cv.destroyAllWindows()
