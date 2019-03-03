import cv2  

import numpy as np 
  
 
cap = cv2.imread('a.png',0)
retval, threshold = cv2.threshold(cap, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('threshold',threshold)  

edges = cv2.Canny(threshold,100,200) 
 
cv2.imshow('Edges',edges)

converted = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(converted, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

img_dilation = cv2.dilate(cap, kernel, iterations=1)

cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()  
