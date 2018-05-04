import cv2
import math
# cv2.namedWindow("image")\
# cv2.setMouseCallback("image", click_and_crop)
import os

videoname = 'C:/vcs/dataset/nhl/video/Bruins-Leafs Game 2 Highlights 4_14_18.mp4'
# keep looping until the 'q' key is pressed
cap = cv2.VideoCapture(videoname)
image_output_dir = 'C:/vcs/dataset/nhl/image/Bruins-Leafs Game 2 Highlights 4_14_18'
frameRate = cap.get(5)
if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)
while cap.isOpened():
    # display the image and wait for a keypress
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    if( not ret):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = image_output_dir + '/' + 'frame_'+str(frameId) + '.jpg'
        cv2.imwrite(filename, frame)

cap.release()
cv2.destroyAllWindows()