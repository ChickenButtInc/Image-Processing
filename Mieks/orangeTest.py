# Determine Orange limit for tracking

import numpy as np
import cv2

#Color limits: adjust values for color limits Hue,sat, val
orangeLower = (150,100,0)          
orangeUpper = (250,200,255)


cap = cv2.VideoCapture(0)

while(True):
    ref, frame = cap.read()

    blur = cv2.GaussianBlur(frame, (11,11), 0)
    #thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]   # retuen, thresh sets threshold for what is ____
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) # hsv = hue, saturation, value

    mask = cv2.inRange(hsv, orangeLower, orangeUpper)   #only processes within range
    mask = cv2.erode(mask, None, iterations=2)          #erodes image to rid spots
    mask = cv2.dilate(mask, None, iterations=2)         #rid of specs

    cv2.imshow('raw', frame)
    cv2.imshow('mask',mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

