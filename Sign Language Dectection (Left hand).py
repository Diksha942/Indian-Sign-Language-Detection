import numpy as np
import cv2 as cv
import math


def nothing(x):
    pass

def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


# creating track bar to find threshold for mask:
cv.namedWindow('Trackbars')
cv.resizeWindow('Trackbars', 256, 256)
cv.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("L - S", "Trackbars", 31, 255, nothing)
cv.createTrackbar("L - V", "Trackbars", 131, 255, nothing)
cv.createTrackbar("U - H", "Trackbars", 67, 179, nothing)
cv.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

#we'll require these ahead
flag = 0   # for the sign lanugage window to appear after pressing enter

defect_coord= []   # used in line 180 to store co-ordinates of defects
far1 = [0,0]   #required in line 170 forstoring changed co-ordinates of defects without changing the original

kernel = np.ones((3, 3), dtype=np.uint8) 
font = cv.FONT_HERSHEY_SIMPLEX

# STARTING THE VIDEO CAPTURE #
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print('Cam\'s not working ')
    exit()

while True:

    ret, frame = cap.read()
    if not ret:
        print('Can\'t recieve frame, try again later maybe')
        break

    frame = cv.flip(frame, 1)

    #getting the area we wanna work on
    roi = frame[100:440, 100:400]

    #converting them into hsv images for us to work
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    #flag changes to one when we press enter, and then we want this wiindow to close
    if flag == 0:

        #get current positions of Track-bars:
        l_h = cv.getTrackbarPos("L - H", "Trackbars")
        l_s = cv.getTrackbarPos("L - S", "Trackbars")
        l_v = cv.getTrackbarPos("L - V", "Trackbars")
        u_h = cv.getTrackbarPos("U - H", "Trackbars")
        u_s = cv.getTrackbarPos("U - S", "Trackbars")
        u_v = cv.getTrackbarPos("U - V", "Trackbars")

        lower_col = np.array([l_h, l_s, l_v])
        upper_col = np.array([u_h, u_s, u_v])

        #adjusting the HSV values acc to lighting conditions and skin colour
        mask_frame = cv.inRange(hsv_frame, lower_col, upper_col)
        mask_frame = cv.morphologyEx(mask_frame, cv.MORPH_CLOSE, kernel)
        mask_frame = cv.GaussianBlur(mask_frame, (5, 5), 100)
        masked_img = cv.bitwise_and(frame, frame, mask=mask_frame)
       
        cv.imshow('Trackbars', masked_img)
        
    else:
        pass

    #creating mask of our hand to get contours
    mask_roi = cv.inRange(hsv_roi, lower_col, upper_col)
    mask_roi = cv.morphologyEx(mask_roi, cv.MORPH_CLOSE, kernel)
      
    mask_roi = cv.GaussianBlur(mask_roi, (5, 5), 100)
    masked_roi = cv.bitwise_and(roi, roi, mask=mask_roi)
    
    #when pressed enter, it'll show the blurred mask of roi, with roi and the panel where it displays no.
    if cv.waitKey(10) == 13 or flag == 1:

        flag = 1

        #finding contours:
        contours, _ = cv.findContours(mask_roi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        #ensuring that contours are detected
        if len(contours) > 1:
            if len(max(contours, key=cv.contourArea) > 100):

                # finding contour of max area(hand)
                cnt = max(contours, key=lambda y: cv.contourArea(y))

                rect = cv.minAreaRect(cnt)
                [(rx, ry), (_,_), _] = rect   #we'll need these co-ordinates to differentiate between 1 and 6

                # approx the contour a bit
                epsilon = 0.0005 * cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, epsilon, True)

                # make convex hull around hand
                hull = cv.convexHull(cnt)

                # we'll need the area and area ratio to tell the numbers apart
                areahull = cv.contourArea(hull)
                areacnt = cv.contourArea(cnt)

                arearatio = ((areahull - areacnt) / areacnt) * 100

                # find the defects in convex hull with respect to hand
                hull = cv.convexHull(approx, returnPoints=False)
                defects = cv.convexityDefects(approx, hull)
                
                l = 0 # the no. of defects

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

                        # ignore angles > 80 and ignore points very close to convex hull(they generally come due to noise)
                        if angle <= 80 and d > 20:
                            l += 1
                            cv.circle(roi, far, 3, [255,153,51], -1)
                            defect_coord.append(far)

                        cv.line(roi, start, end, [51,153,255], 2)

                except AttributeError:
                    pass
            else:
                pass

            #define the panel where no. will be displayed
            cv.rectangle(frame,(100,400), (400,440),(96,96,96),-1)
            cv.rectangle(frame,(100,100),(400,440),(255,255,255),2)

            #setting conditions for numbers to display, according to hand gestures
            if l == 0:
                if arearatio < 14.5:

                    #the gesture 9 has a largere area than 0
                    if 10000< areahull < 18000: 
                        cv.putText(frame, '0', (105,435), font, 1, (0, 255, 128), 3, cv.LINE_AA)
                    else:
                        cv.putText(frame, '9', (105,435), font, 1, (0, 255, 128), 3, cv.LINE_AA)
                else:

                    #getting the co-ordinates of topmost point i.e out fingertip
                    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
                    theta = getAngle((rx+100,ry),(rx,ry),topmost) #remeber the minAreaRect? The angle between finger tip, and the horizontal of center of
                                                                #rect co-ordinate will help us distinguish 1 and 6
                    if  260 < theta:
                        cv.putText(frame, '1', (105,435), font, 1, (0, 255, 128), 3,cv.LINE_AA)
                    else:
                        cv.putText(frame, '6', (105,435), font, 1, (0, 255, 128), 3, cv.LINE_AA)

            elif l == 1:
                
                theta = getAngle((400,400),(250,400),defect_coord[-1]) #getting the angle of defects from the horizontal

                if theta>245:
                    cv.putText(frame, '2', (105,435), font, 1, (0, 255, 128), 3, cv.LINE_AA)
                else:
                    cv.putText(frame, '7', (105,435), font, 1, (0, 255, 128), 3, cv.LINE_AA)

                if len(defect_coord)>5:     #we are kinda appending every co-ordinate of defect from each frame to
                    del defect_coord[0:3]   #the array. Deleting the previous unwanted elements would save memory
                    
            elif l == 2:
                
                theta1 = getAngle((400,400),(250,400),defect_coord[-1])
                theta2 = getAngle((400,400),(250,400),defect_coord[-2])
                #print(theta1, theta2)
            
                if theta1>235 and theta2>240:
                    cv.putText(frame, '3', (105,435), font,1, (0, 255, 128), 3, cv.LINE_AA)
                else:
                    cv.putText(frame, '8', (105,435), font, 1, (0, 255, 128), 3, cv.LINE_AA)

                if len(defect_coord)>5:
                    del defect_coord[0:3]

            elif l == 3:
                cv.putText(frame, '4', (105,435), font, 1, (0, 255, 128), 3, cv.LINE_AA)

            elif l == 4:
                cv.putText(frame, '5', (105,435), font, 1, (0, 255, 128), 3, cv.LINE_AA)
                
            else:
                cv.putText(frame, 'Reposition', (105,435), font, 1, (0, 255, 128), 3, cv.LINE_AA)
        else:
            cv.rectangle(frame,(100,400), (400,440),(96,96,96),-1)
            cv.putText(frame, 'Put hand in the box', (105,435), font, 0.7, (0, 255, 128), 3, cv.LINE_AA) #if no contour is detected

        cv.destroyWindow('Trackbars')
        cv.imshow('Sign Language Detection', roi)
        cv.imshow('mask', mask_roi)
    

    k = cv.waitKey(1) & 0xff
    if k == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
