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

def detectHockeyRinkEdgeByColor(image, pointColor, showDebugImage=False):
    hsvColor = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # draw a rectangle around the region of interest
    # cv2.addText(image, 'test', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255),2,cv2.LINE_AA)
    #      cv2.imshow("image", image)
    print(pointColor)
    lower_blue = np.array([pointColor[0] -7, max(0, pointColor[1] - 100), max(0, pointColor[2] - 100)])
    upper_blue = np.array([pointColor[0] + 7, min(255, pointColor[1] + 100), min(255, pointColor[2] + 100)])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsvColor, lower_blue, upper_blue)
    output = cv2.bitwise_and(image, image, mask=mask)
    if (showDebugImage):
        cv2.imshow("output", output)
    outputGray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    if (showDebugImage):
        cv2.imshow("outputgray", outputGray)
    _ret, edges = cv2.threshold(outputGray, 150, 255, cv2.THRESH_BINARY)
    edges = cv2.GaussianBlur(edges, (15, 15), 0)
    if (showDebugImage):
        cv2.imshow("edges", edges)
    canny = cv2.Canny(edges, 50, 255)
    if (showDebugImage):
        cv2.imshow('canny', canny)

    # im2, contours, hierarchy = cv2.findContours(edges,  cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cnt = contours[0]
    # max_area = cv2.contourArea(cnt)
    #
    # for cont in contours:
    #     if cv2.contourArea(cont) > max_area:
    #         cnt = cont
    #         max_area = cv2.contourArea(cont)
    #
    # epsilon = 0.1 * cv2.arcLength(cnt, True)
    # approx = cv2.approxPeolyDP(cnt, epsilon, True)
    # cv2.drawContours(image, approx, -1, (0, 255, 0), 3)
    # cv2.imshow('contours',image)
    edges = canny

    def hough_lines(image):
        """
        `image` should be the output of a Canny transform.

        Returns hough lines (not the image with lines)
        """
        return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)

    list_of_lines = hough_lines(edges)

    def calc_lenght_slope(lines):
        line_weights = []
        line_slopes = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                line_weights.append(length)
                line_slopes.append(slope)
        return np.asarray(line_weights), np.asarray(line_slopes)


    weights, slopes = calc_lenght_slope(list_of_lines)
    print(np.size(slopes))
    print(slopes.shape)
    ind = np.argsort(slopes)
    print(slopes[ind])
    def merge_lines(lines, line_weigths, line_slopes, ind):
        argg_inds = []
        argg_item = []

        for v in ind:
            print(line_slopes[v])
            if 0 == np.size(argg_item):
                argg_item.append(v)
            else:
                last_item_slope = line_slopes[argg_item[-1]]

                if abs(line_slopes[v] - last_item_slope) > 0.1:
                    argg_inds.append(argg_item)
                    argg_item = []
                else:
                    argg_item.append(v)

        if argg_item:
            argg_inds.append(argg_item)

        procesed_merged_lines = []

        for line_inds in argg_inds:
            process_weights = line_weigths[line_inds]
            process_slopes = line_slopes[line_inds]
            process_lines = lines[line_inds]

            procesed_merged_lines.append(np.average(process_lines, axis=0))

        return procesed_merged_lines

    merged_lines = merge_lines(list_of_lines, weights, slopes, ind)

    for line in merged_lines:
        for x1, y1, x2, y2 in line.astype(int):
            cv2.line(image, (x1, y1), (x2, y2), [255, 0, 0], 10)

    cv2.imshow("line", image)

    return image


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
    if (frameId):
        shape = frame.shape
        frame = cv2.resize(frame, (int(shape[1]/2), int(shape[0]/2)))

        #cv2.imshow('frame', frame)
        try:
            cv2.imshow('frameEdge', detectHockeyRinkEdgeByColor(image=frame, pointColor= [22,125,186]))
        except :
            print('oops')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
