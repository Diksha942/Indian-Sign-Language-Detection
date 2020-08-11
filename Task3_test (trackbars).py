import numpy as np
import cv2 as cv
import math


def nothing(x):
    pass


def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist

def slope(x1,y1,top):
    theta = math.degrees(math.atan2(top[1]-y1, top[0]-x1))
    return theta


# creating track bar to find threshold for mask:
cv.namedWindow('Trackbars')
cv.resizeWindow('Trackbars', 256, 256)
cv.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L - S", "Trackbars", 10, 255, nothing)
cv.createTrackbar("L - V", "Trackbars", 70, 255, nothing)
cv.createTrackbar("U - H", "Trackbars", 180, 255, nothing)
cv.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# STARTING THE VIDEO CAPTURE#
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Cam\'s not working ')
    exit()

# for selecting skin value in roi:
flag = 0

dist = []
apex = []

# defining kernel:
kernel = np.ones((3, 3), dtype=np.uint8)
font = cv.FONT_HERSHEY_SIMPLEX
while True:

    ret, frame = cap.read()
    if not ret:
        print('Can\'t recieve frame, try again later maybe')
        break

    frame = cv.flip(frame, 1)

    # defining region of interest:
    roi = frame[100:300, 100:300]

    cv.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    # get current positions of Track-bars:
    l_h = cv.getTrackbarPos("L - H", "Trackbars")
    l_s = cv.getTrackbarPos("L - S", "Trackbars")
    l_v = cv.getTrackbarPos("L - V", "Trackbars")
    u_h = cv.getTrackbarPos("U - H", "Trackbars")
    u_s = cv.getTrackbarPos("U - S", "Trackbars")
    u_v = cv.getTrackbarPos("U - V", "Trackbars")

    lower_col = np.array([l_h, l_s, l_v])
    upper_col = np.array([u_h, u_s, u_v])

    mask_frame = cv.inRange(hsv_frame, lower_col, upper_col)
    mask_frame = cv.morphologyEx(mask_frame, cv.MORPH_CLOSE, kernel)
    masked_img = cv.bitwise_and(frame, frame, mask=mask_frame)
    cv.imshow('Trackbars', masked_img)

    mask_roi = cv.inRange(hsv_roi, lower_col, upper_col)
    mask_roi = cv.morphologyEx(mask_roi, cv.MORPH_CLOSE, kernel)
    mask_roi = cv.GaussianBlur(mask_roi, (5, 5), 100)
    masked_roi = cv.bitwise_and(roi, roi, mask=mask_roi)

    if cv.waitKey(10) == 13 or flag == 1:

        flag = 1
        # finding contours:
        contours, _ = cv.findContours(mask_roi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            if len(max(contours, key=cv.contourArea) > 100):

                # finding contour of max area(hand):
                cnt = max(contours, key=lambda y: cv.contourArea(y))
                rect = cv.minAreaRect(cnt)
                [(rx, ry), (w, h), _] = rect

                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(roi, [box], 0, (0, 0, 255), 1)
                cv.circle(roi, (int(rx), int(ry)), 2, [0, 0, 255], -1)

                # approx the contour a little:
                epsilon = 0.0005 * cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, epsilon, True)

                # make convex hull around hand
                hull = cv.convexHull(cnt)

                # define area of hull and area of hand
                areahull = cv.contourArea(hull)
                areacnt = cv.contourArea(cnt)

                # find the percentage of area not covered by hand in convex hull
                arearatio = ((areahull - areacnt) / areacnt) * 100

                # find the defects in convex hull with respect to hand
                hull = cv.convexHull(approx, returnPoints=False)
                defects = cv.convexityDefects(approx, hull)

                # l = no. of defects:
                l = 0

                try:
                    # code for finding no. of defects due to fingers
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(approx[s][0])
                        end = tuple(approx[e][0])
                        far = tuple(approx[f][0])
                        pt = (100, 180)

                        # finding length of all sides of triangle:
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                        # applying HERON's formula to find area of triangle:
                        s = (a + b + c) / 2
                        ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                        # distance between point and convex hull
                        d = (2 * ar) / a

                        # applying cosine rule here and finding angle(A):
                        angle = np.rad2deg(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))

                        # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                        if angle <= 90 and d > 20:
                            l += 1
                            cv.circle(roi, far, 3, [255, 0, 0], -1)

                        # draw lines around hand
                        cv.line(roi, start, end, [0, 255, 0], 2)
                except AttributeError:
                    pass

                l += 1
            else:
                pass

            theta = slope(rx, ry, far)

            if l == 1:
                if arearatio < 14.5:
                    if 4000 < areahull < 7200:
                        cv.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
                    else:
                        cv.putText(frame, '9', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
                else:
                    '''p = distance(apex[0][0], rx, apex[0][1], ry)
                    q = 100
                    r = distance(apex[0][0], rx + 100, apex[0][1], ry)'''

                    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
                    theta = slope(rx,ry,topmost)
                    if  0 < theta < 90:
                        cv.putText(frame, '6', (0, 50), font, 2, (0, 0, 255), 3,cv.LINE_AA)
                    # I tried, yeh wala, 1 ad 6 wala, acchese kaam nahi kr rha, please see if u can do this or not
                    else:
                        cv.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

                '''elif 14.5 <= arearatio < 16.5:
                    cv.putText(frame, '6', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
                elif 16.5 <= arearatio <= 19.5:
                    cv.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)'''

            elif l == 2:

                if theta < 0:
                    cv.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
                else:
                    cv.putText(frame, '7', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

            elif l == 3:

                if arearatio < 27 :
                    cv.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
                else:
                    cv.putText(frame, '8', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

            elif l == 4:
                cv.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

            elif l == 5:
                cv.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

            elif l == 6:
                cv.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

            else:
                cv.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
        else:
            cv.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow('Sign Language Detection', frame)
        cv.imshow('mask', mask_roi)

    k = cv.waitKey(1) & 0xff
    if k == ord('q'):
        break

    

cap.release()
cv.destroyAllWindows()
