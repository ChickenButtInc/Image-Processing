# Contour reading Test

import numpy as np
import cv2

#Color limits: adjust values for color limits
orangeLower = (170,114,240)          
orangeUpper = (180,182,255)

cap = cv2.VideoCapture(0)

while(True):
    ref, frame = cap.read()

    #image adjustment
    blur = cv2.GaussianBlur(frame, (5,5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) # hsv = hue, saturation, value

    mask = cv2.inRange(hsv, orangeLower, orangeUpper)   #only processes within range
    mask = cv2.erode(mask, None, iterations=2)          #erodes image to rid spots
    mask = cv2.dilate(mask, None, iterations=2)         #rid of specs
    
    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1] # (__, cnts, __)
    #cnts = cnts[1]


    if len(cnts) > 0:   #only if 1 contour is found
        
        for c in cnts:
            # Compute Moments
            M = cv2.moments(c)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            center = (cX, cY)
            cv2.circle(frame, center, 5, (0,255,0), -1)# (img, center, radius, filled)

    cv2.imshow('raw', frame)
    cv2.imshow('mask', mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
