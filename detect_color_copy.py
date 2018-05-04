import argparse
import cv2
from matplotlib import pyplot as plt
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
import numpy as np

refPt = []
cropping = False

def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1

def detectHockeyRinkEdgeByColor(image, pointColor, showDebugImage = False):
    hsvColor = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # draw a rectangle around the region of interest
    # cv2.addText(image, 'test', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255),2,cv2.LINE_AA)
    #      cv2.imshow("image", image)
    lower_blue = np.array([pointColor[0] - 5, 100, 100])
    upper_blue = np.array([pointColor[0] + 5, 250, 250])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsvColor, lower_blue, upper_blue)
    output = cv2.bitwise_and(image, image, mask=mask)
    if(showDebugImage):
        cv2.imshow("output", output)
    outputGray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    if(showDebugImage):
        cv2.imshow("outputgray", outputGray)
    _ret, edges = cv2.threshold(outputGray, 150, 255, cv2.THRESH_BINARY)
    if(showDebugImage):
        cv2.imshow("edges", edges)
    edges = cv2.Canny(edges, 150, 255)

    shapex, shapey = edges.shape
    xdata = []
    ydata = []

    for i in range(shapex):
        for j in range(shapey):
            if edges[i][j] >= 225:
                xdata.append(j)
                ydata.append(i)
    # Reshape into [[x1, y1],...]
    # Translate points back to original positions.

    z = np.polyfit(xdata, ydata, 5)
    f = np.poly1d(z)

    m = np.arange(0, edges.shape[1], 1)

    cloneImage = image.copy()
    pts = np.array([[t, f(t)] for t in m], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(cloneImage, [pts], False, (255, 0, 0))
    return cloneImage

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        i = 1

#construct the argument parser and parse the arguments

# load the image, clone it, and setup the mouse callback function
#image = cv2.imread(filename)
#clone = image.copy()
cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_and_crop)
videoname = 'C:/vcs/dataset/nhl/video/Bruins-Leafs Game 2 Highlights 4_14_18.mp4'
# keep looping until the 'q' key is pressed
cap = cv2.VideoCapture(videoname)
frameRate = cap.get(5)

while cap.isOpened():
    # display the image and wait for a keypress
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    if (not ret):
        break
    if (frameId % np.math.floor(frameRate) == 0):
        shape = frame.shape
        frame = cv2.resize(frame, (int(shape[1]/2), int(shape[0]/2)))
        try:
            #cv2.imshow('frame', frame)
            cv2.imshow('frameEdge', detectHockeyRinkEdgeByColor(image=frame, pointColor= [22,125,186]))
        except:
            print('oops...')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
