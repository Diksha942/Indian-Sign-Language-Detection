import numpy as np
import cv2 as cv
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

    ret, frame = cap.read()
    if not ret:
        print('Can\'t recieve frame, try again later maybe')
        break

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
    cnt = max(contours, key=lambda y: cv.contourArea(y))
    #cnt = max(contours, key=cv.contourArea)

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
        if angle <= 90 and d > 30:
            l += 1
            cv.circle(roi, far, 3, [255, 0, 0], -1)

        # draw lines around hand
        cv.line(roi, start, end, [0, 255, 0], 2)

    l += 1

    # print corresponding gestures which are in their ranges
    font = cv.FONT_HERSHEY_SIMPLEX

    if l == 1:
        if areacnt < 2000:
            cv.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
        else:
            if arearatio < 14.5:
                cv.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
            elif 14.5 <= arearatio < 16.5:
                cv.putText(frame, '6', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
            elif 16.5 <= arearatio <= 19.5:
                cv.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
            else:
                cv.putText(frame, '9', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

    elif l == 2:
        if arearatio < 13.5:
            cv.putText(frame, '7', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
        else:
            cv.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

    elif l == 3:

        if arearatio < 27:
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

    # Displaying the windows:
    cv.imshow('mask', mask)
    cv.imshow('frame', frame)

    k = cv.waitKey(1) & 0xff
    if k == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
