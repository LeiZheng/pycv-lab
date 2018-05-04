import cv2
import numpy as np

img = cv2.imread('frame1250.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray.jpg',gray)
edges = cv2.Canny(gray,20,150,apertureSize = 3)
cv2.imshow('edges.jpg',edges)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
cv2.Hough
cv2.imshow('houghlines3.jpg',img)
cv2.waitKey()
cv2.destroyAllWindows()

