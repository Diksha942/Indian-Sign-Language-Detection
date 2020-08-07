import numpy as np
import cv2 as cv
import itertools

def drawRect(frame):

    global rect_coord

    h,w,_ = frame.shape
    coord1 = list(itertools.product([((w/2)-34),((w/2)-8),((w/2)+18)],[((h/2)-34),((h/2)-8),((w/2)+18)]))  #gives us all the upper left corners
    coord2 = list(itertools.product([((w/2)-18),((w/2)+8),((w/2)+34)],[((h/2)-18),((h/2)+8),((w/2)+34)]))   #gives us all the bottom right corners
    
    rect_coord = [i+j for i,j in zip(coord1,coord2)]  #combining all the coordinates, to avoid nesting of for loop

    for  i in rect_coord:
        cv.rectangle(frame,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(0,255,0),1)   #drawing the rectangles, where we need to place our hand. Dimensions are 16x16, with 10 pixels space

    return(frame)

def getROI(frame):   #For getting the region of intrest

    roi = np.zeros((2304,2304,3), dtype = frame.dtype) 

    j=0
    for i in rect_coord:
        roi[j:j+16,j:j+16] = frame[i[0]:i[2],i[1]:[3]]
        j+=16
    
        
def detectHSV(roi,frame):

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))

    hist = cv.calcHist([roi],[0,1],None,[180,256],[0,180,0,256])   #creating the histogram of Region of intrest
    cv.normalize(hist, hist,0,255,cv.NORM_MINMAX)                   #normalising the histogram
    dst = cv.calcBackProject([frame], [0, 1], hist, [0, 180, 0, 256], 1)      #back projecting the normalised histogram on our frame
    
    dst=cv.morphologyEx(dst,cv.MORPH_CLOSE,kernel)

    ret, thresh = cv.threshold(dst,127, 255, cv.THRESH_BINARY)  
    thresh1 = cv.merge((thresh, thresh, thresh))            #creating a 3 channel threshold mask, to then bitwise_and with the 3 channel frame

    masked = cv.bitwise_and(frame,thresh1)

    return(masked)

#STARTING THE VIDEO CAPTURE#

cap = cv.VideoCapture(0) 
if not cap.isOpened():
    print('Cam\'s not working ')
    exit()

while True:

    ret, frame = cap.read()
    if not ret:
        print('Can\'t recieve frame, try again later maybe')
        break

    frame = cv.flip(frame,1)
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    





















    

















    
