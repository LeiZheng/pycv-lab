import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

img = cv2.imread('frame1250.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[209,194],[561,325],[567,211]])
pts2 = np.float32([[205,220],[645,220],[(205 + 645) / 2,110]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()