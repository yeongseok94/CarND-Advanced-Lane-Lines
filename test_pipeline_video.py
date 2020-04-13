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
videolist = os.listdir("test_videos/")

for videoname in videolist:
    ld = LaneDetection()
    outputdir = "output_videos/" + videoname
    clip = VideoFileClip("test_videos/" + videoname)
    clip_processed = clip.fl_image(ld.process_image)
    %time clip_processed.write_videofile(outputdir, audio=False)