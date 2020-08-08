import numpy as np
import cv2 as cv
from scipy.spatial import distance
import math


def nothing(x):
    pass


# creating track bar to find threshold for mask:
cv.namedWindow('Trackbars')
cv.resizeWindow('Trackbars', 256, 256)
cv.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L - S", "Trackbars", 20, 255, nothing)
cv.createTrackbar("L - V", "Trackbars", 70, 255, nothing)
cv.createTrackbar("U - H", "Trackbars", 20, 255, nothing)
cv.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# STARTING THE VIDEO CAPTURE#
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Cam\'s not working ')
    exit()

# for selecting skin value in roi:
hist_flag = 0

# defining kernel:
kernel = np.ones((3, 3), dtype=np.uint8)

while True:
    try:
        ret, frame = cap.read()

        frame = cv.flip(frame, 1)

        # defining region of interest:
        roi = frame[100:300, 100:300]

        cv.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        # get current positions of Track-bars:
        l_h = cv.getTrackbarPos("L - H", "Trackbars")
        l_s = cv.getTrackbarPos("L - S", "Trackbars")
        l_v = cv.getTrackbarPos("L - V", "Trackbars")
        u_h = cv.getTrackbarPos("U - H", "Trackbars")
        u_s = cv.getTrackbarPos("U - S", "Trackbars")
        u_v = cv.getTrackbarPos("U - V", "Trackbars")

        lower_col = np.array([l_h, l_s, l_v])
        upper_col = np.array([u_h, u_s, u_v])

        mask = cv.inRange(hsv, lower_col, upper_col)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # finding contours:
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # finding contour of max area(hand):
        #cnt = max(contours, key=lambda y: cv.contourArea(y))
        cnt = max(contours, key=cv.contourArea)
        M = cv.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

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
        farthestpt = []

        # code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])

            l += 1
            farthestpt = farthestpt + [far]
        #taking topmost pt in contour
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        center = (cx, cy)

        #dist between topmost and center
        dist = distance(topmost, center)

        # print corresponding gestures which are in their ranges
        font = cv.FONT_HERSHEY_SIMPLEX

        if l == 0:
            if areacnt < 2000:
                cv.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
            elif arearatio<16:
                if dist < 10:
                    cv.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
                else:
                    cv.putText(frame, '9', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
            else:
                if dist > 30:
                    cv.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
                else:
                    cv.putText(frame, '6', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

        elif l == 1:
            if arearatio < 13.5 and math.degrees(math.atan2(farthestpt[0][1]-cy, farthestpt[0][0]-cx)) < 90:
                cv.putText(frame, '7', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
            else:
                cv.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

        elif l == 2:

            if arearatio < 27 and math.degrees(math.atan2(farthestpt[0][1]-cy, farthestpt[0][0]-cx)) < 90:
                cv.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
            else:
                cv.putText(frame, '8', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

        elif l == 3:
            cv.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

        elif l == 4:
            cv.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

        elif l == 5:
            cv.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

        else:
            cv.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

        # Displaying the windows:
        cv.imshow('mask', mask)
        cv.imshow('frame', frame)

        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
    except:
        pass
cap.release()
cv.destroyAllWindows()
