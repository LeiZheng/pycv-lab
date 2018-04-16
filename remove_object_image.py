
from skimage.io import imread, imsave
from inpainter import Inpainter

image = imread('resources/image6.jpg')
mask = imread('resources/mask6.jpg', as_grey=True)
output_image = Inpainter(
        image,
        mask,
        patch_size=16,
        plot_progress=True
    ).inpaint()

imsave('output1.jpg', output_image, quality=100)