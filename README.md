## Advanced Lane Finding Project

Also uploaded on GitHub repository: https://github.com/yeongseok94/CarND-Advanced-Lane-Lines.git

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undist2.png "Undistorted"
[image2]: ./test_images/test2.jpg "Road"
[image2.1]: ./output_images/test2_undist.jpg "Road Transformed"
[image3]: ./output_images/test2_binary.jpg "Binary Example"
[image4]: ./output_images/straight_warped2.png "Warp Example"
[image5]: ./output_images/test2_laneline.png "Fit Visual"
[image6]: ./output_images/test2_final.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

Code Descriptions:

* `lanedetection_depends.py`: Contains all the functions and pipelines for this project.
* `test_calib.py`: Test code for camera calibration.
* `test_perspective.py`: Test code for perspective transform.
* `test_pipeline.py`: Test code for executing overall pipeline onto test images.
* `test_pipeline_video.py`: Main execution file. Code for executing overall pipeline onto test videos.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in class `CameraCalibration()` in `lanedetection_depends.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each chessboard detection. The chessboard is originally 9x6, but for the cases where detection has failed, I modified the size to the smaller size so that I can grab more `objpoints` and `imgpoints`.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the methods already shown in previous section, I get the distortion-corrected images. This is the example of `./test_images/test1.jpg`.

* Before distortion correction:
![alt text][image2]

* After distortion correction:
![alt text][image2.1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The procedure is shown in `LaneDetection.threshold_image()`.

First, I obtained HLS color space using `cv2.cvtColor()` function. Then, I applied Sobel X kernel with size 7 by 7 onto L channel and thresholded from 20 to 100 by `cv2.Sobel` function.
Also, I thresholded onto S channel from 120 to 255.
Finally, I stacked all thresholded binary image and obtained final binary image. This is example of `./test_images/test1.jpg` undistorted and thresholded.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `LaneDetection.obtain_perspective()` and `LaneDetection.warp_perspective()`. This uses function `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()`. The source point `src` and the destination point `dst` are obtained like:

```python
img_size = (img.shape[1], img.shape[0])
self.src = np.float32(
    [[(img_size[0] / 2) - 53, img_size[1] / 2 + 95],
    [((img_size[0] / 6) - 7), img_size[1]],
    [(img_size[0] * 5 / 6) + 57, img_size[1]],
    [(img_size[0] / 2 + 57), img_size[1] / 2 + 95]])
self.dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For first frame, I identified lane line pixels by histogram peaks and windows method which is introduced in the lecture. From the second frame, I identified lane line pixels around previous frame's polynominal fit of lane lines. Then, I obtained each line's 2nd order polynominal fit using `scipy.optimize.curve_fit()`. If obtained lane line polynominal fit is not parallel enough, then I obtained refitted the lane line with the less lane line pixels to be approximately parallel to other lane line.

This procedure is included in `LaneDetection.search_laneline_window()` and `LaneDetection.search_laneline_nearpoly()`. Example result looks like:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `Line.update()` so that I can update lane line class variables each step of video frames.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `LaneDetection.process_image()`. This function includes whole process within one frame. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My lane detection pipeline works well for when there are not much lightness and color change in horizontal direction within lane line area.
However, there can be serious lightness change between daytime and nighttime, or there can be steep color or lightness change other than lane line due to defects on road.

To make robust to defects on road, we can add additional algorithms within lane line seach window so that we can distinguish pixels which is out of mainstream line.
Or, we can make modification on thresholds with respect to lightness so that the pipeline itself adjust to the lightness of lane line area.