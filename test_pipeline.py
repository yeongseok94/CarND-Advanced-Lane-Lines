import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from scipy.optimize import curve_fit

from lanedetection_depends import LaneDetection
ld = [LaneDetection(),LaneDetection(),LaneDetection(),
      LaneDetection(),LaneDetection(),LaneDetection(),LaneDetection()]

def poly_ord2(y, a2, a1, a0):
    return a2*y**2 + a1*y + a0

images_format = './test_images/test'
output_format = './output_images/test'

for idx in range(1, 7):
    img = mpimg.imread(images_format + str(idx) + '.jpg')
    out_img = ld[idx].process_image(img)

    mpimg.imsave(output_format + str(idx) + '_final.png', out_img)