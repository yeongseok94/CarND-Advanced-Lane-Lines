import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from lanedetection_depends import LaneDetection
ld = LaneDetection()

images_format = './test_images/straight_lines'
output_format = './output_images/straight_warped'

for idx in range(1, 3):
    img = cv2.imread(images_format + str(idx) + '.jpg')
    ld.obtain_perspective(img)
    undistimg = ld.cc.undistort_image(img)
    warpedimg = ld.warp_perspective(undistimg)
    
    pts = np.array(ld.src, np.int32)
    undistimg = cv2.polylines(undistimg, [pts], True, (0,0,255), 2)
    pts = np.array(ld.dst, np.int32)
    warpedimg = cv2.polylines(warpedimg, [pts], True, (0,0,255), 2)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(undistimg, cv2.COLOR_BGR2RGB))
    ax1.set_title('Undistorted Image', fontsize=30)
    ax2.imshow(cv2.cvtColor(warpedimg, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted Warped Image', fontsize=30)
    plt.subplots_adjust(left=0.02, right=0.98, top=1, bottom=0)
    plt.savefig(output_format + str(idx) + '.png')
    
print(ld.src)
print(ld.dst)