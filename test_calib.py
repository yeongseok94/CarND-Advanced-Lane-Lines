import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from lanedetection_depends import CameraCalibration
cc = CameraCalibration()

output_format = './output_images/undist'

for idx in range(1, 21):
    img = cv2.imread(cc.images_format + str(idx) + '.jpg')
    undistimg = cc.undistort_image(img)
    cv2.imwrite(output_format + str(idx) + '.jpg', undistimg)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(cv2.cvtColor(undistimg, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.subplots_adjust(left=0.02, right=0.98, top=1, bottom=0)
    plt.savefig(output_format + str(idx) + '.png')