import cv2
import pandas as pd
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import screeninfo
import serial



arduino = serial.Serial('com7', 9600) # set arduino serial


def safe_div(x,y): # no div by zero
    if y==0: return 0
    return x/y


def nothing(x): # for trackbar
    pass


def rescale_frame(frame, percent=100):  # resize window fx
    width = int(frame.shape[1] * percent/ 70)
    height = int(frame.shape[0] * percent/ 70)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def midpoint(ptA, ptB): 
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


#set a monitoring zone (ROI)
def CheckEntranceLineCrossing(x, CoorXEntranceLine, CoorXExitLine):
    if (tl[0] < CoorXEntranceLine) and (tr[0] < CoorXEntranceLine) and (bl[0] < CoorXEntranceLine) and (br[0] < CoorXEntranceLine):
        if (tl[0] > CoorXExitLine) and (tr[0] > CoorXExitLine) and (bl[0] > CoorXExitLine) and (br[0] > CoorXExitLine):
            return True
        else:
            return False


def contorni_e_aree(): # useful to prevent empty areas array when system starts
    
    ret, frame=videocapture.read() 
    frame_resize = rescale_frame(frame)
    if not ret:
        print("cannot capture the frame")
        exit()
   
    thresh= cv2.getTrackbarPos("threshold", windowName) 
    ret,thresh1 = cv2.threshold(frame_resize,thresh,255,cv2.THRESH_BINARY) #set threshold
    
    kern=cv2.getTrackbarPos("kernel", windowName) 
    kernel = np.ones((kern,kern),np.uint8) #set kernel
    
    itera=cv2.getTrackbarPos("iterations", windowName) 
    dilation =   cv2.dilate(thresh1, kernel, iterations=itera)
    erosion = cv2.erode(dilation,kernel,iterations = itera) # set iterations

    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel) 
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  
    closing = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY) # contours analysis on closing
 
    
    contours,hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # find contours with simple approximation cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE

    closing = cv2.cvtColor(closing,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(closing, contours, -1, (128,255,0), 1)
   

    
    areas = [] #list to hold all areas contours related

    for contour in contours:
      ar = cv2.contourArea(contour)
      areas.append(ar)
          return areas, contours



# --- BEGIN --- !!!

videocapture=cv2.VideoCapture(0)
if not videocapture.isOpened():
    print("can't open camera")
    exit()
    
windowName="Realtime analysis"
cv2.namedWindow(windowName)

# Sliders to adjust image
cv2.createTrackbar("threshold", windowName, 75, 255, nothing)
cv2.createTrackbar("kernel", windowName, 5, 30, nothing)
cv2.createTrackbar("iterations", windowName, 1, 10, nothing)


conta_cicli = 0 # number of cycles

showLive=True  # main 
while(showLive):
    
    ret, frame=videocapture.read()
    frame_resize = rescale_frame(frame)
    if not ret:
        print("cannot capture the frame")
        exit()
                               
    thresh= cv2.getTrackbarPos("threshold", windowName) 
    ret,thresh1 = cv2.threshold(frame_resize,thresh,255,cv2.THRESH_BINARY) #get threshold value from trackbar
                                
    kern=cv2.getTrackbarPos("kernel", windowName) 
    kernel = np.ones((kern,kern),np.uint8) # get kernel value from trackbar
                                
    itera=cv2.getTrackbarPos("iterations", windowName) 
    dilation = cv2.dilate(thresh1, kernel, iterations=itera)
    erosion = cv2.erode(dilation,kernel,iterations = itera) # refines all edges in the binary image

    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel) 
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  
    closing = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY) 
                                
    contours,hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # find contours with simple approximation cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE

    closing = cv2.cvtColor(closing,cv2.COLOR_GRAY2RGB)
                                
    areas = [] #list to hold all areas contours related

    for contour in contours:
        ar = cv2.contourArea(contour)
        areas.append(ar)

    while len(areas)==0: # no empty areas list
        areas, contours = contorni_e_aree() 
       
   
    max_area = max(areas)
    max_area_index = areas.index(max_area)   
    cnt = contours[max_area_index-1] # largest area contour viewed
                                                           
   

    #plot reference lines (entrance and exit lines)
    height, width =frame_resize.shape[:2]
    OffsetRefLines=350
    CoorXEntranceLine = int((width / 2) + OffsetRefLines)
    CoorXExitLine = int((width / 2) - OffsetRefLines)
    cv2.line(frame_resize, (CoorXEntranceLine,0), (CoorXEntranceLine,height), (125, 0, 255), 1)
    cv2.line(frame_resize, (CoorXExitLine,0), (CoorXExitLine,height), (125, 0, 255), 1)

    cv2.line(closing, (CoorXEntranceLine,0), (CoorXEntranceLine,height), (125, 0, 255), 1)
    cv2.line(closing, (CoorXExitLine,0), (CoorXExitLine,height), (125, 0, 255), 1)


    # compute the rotated bounding box of the contour
    orig = frame_resize.copy()
    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    
                           
    # loop over the original points and draw them
    for (x, y) in box:

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
                                 
    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-right and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # compute the center of the contour
    M = cv2.moments(cnt)
    cX = int(safe_div(M["m10"],M["m00"]))
    cY = int(safe_div(M["m01"],M["m00"]))

    passed = CheckEntranceLineCrossing(cX,CoorXEntranceLine,CoorXExitLine) #ROI monitoring
    
    if passed==True: 
        # related to the "loop over the original points and draw them" part
        for (x,y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # related to the "compute the rotated bounding box of the contour" part
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 5)
        cv2.drawContours(closing, [cnt], 0, (128,255,0), 10)
                                 
        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
                                   
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        # compute the size of the object
        pixelsPerMetric = 1
        dimA = (dA / (pixelsPerMetric*10)) + ((dA / (pixelsPerMetric*10))/100)*10 #set this formula prop to your camera distance
        dimB = (dB / (pixelsPerMetric*10)) + ((dB / (pixelsPerMetric*10))/100)*10
                          
        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}mm".format(dimB), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(orig, "{:.1f}mm".format(dimA), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # draw lines between the midpoints
        if (dimA > 5 and dimB > 5) and (dimA < 95 and dimB <95) :
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
            cv2.drawContours(orig, [cnt], 0, (0,0,255), 1)
                           
        # draw the contour and center of the shape on the image
        if (dimA > 10 and dimB > 10) and (dimA < 95 and dimB <95) :
            cv2.circle(orig, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(orig, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        
       
        # Shape Recognition (frame by frame)
        differenzaAB = dimA-dimB
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

        if (dimA > 10 and dimB > 10) and (dimA<70 and dimB<70): # set this prop to your object min and max size
            
            if len(approx) == 3:
                        tipoforma="Triangolo"
                        cv2.putText(orig, tipoforma, (int(tltrX)-20, int(tltrY)-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (125, 0, 255), 2)
                        
            elif len(approx) == 4:
                if differenzaAB > 5 or differenzaAB < -5:
                        tipoforma="Rettangolo"
                        cv2.putText(orig, tipoforma, (int(tltrX)-20, int(tltrY)-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (125, 0, 255), 2)
                else:
                        tipoforma="Quadrato"
                        cv2.putText(orig, tipoforma, (int(tltrX)-20, int(tltrY)-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (125, 0, 255), 2)
                        
            elif len(approx) > 6:
                        tipoforma="Cerchio"
                        cv2.putText(orig, tipoforma, (int(tltrX)-20, int(tltrY)-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (125, 0, 255), 2)
        
     
        if conta_cicli==10: # for non continous serial communication
            if (dimA>5 and dimB>10) and (dimA<70 and dimB<70):
                arduino.write(str.encode(" x: "+str(round(dimB,1))+" "+"y: "+str(round(dimA,1))+" [mm] "+"?"+tipoforma))
            conta_cicli=0
        else:
            conta_cicli = conta_cicli + 1
     

    orig_stampa = orig.copy()
    orig_stampa = rescale_frame(orig_stampa,45)
    cv2.imshow(windowName, orig_stampa)

    
    closing_stampa = closing.copy()
    closing_stampa = rescale_frame(closing_stampa,45)
    cv2.imshow('Mask', closing_stampa)
   
                                    
    if cv2.waitKey(30)>=0:
        showLive=False
    
        
videocapture.release()
cv2.destroyAllWindows()
