import numpy as np
import cv2
from scipy.optimize import curve_fit

def poly_ord2(y, a2, a1, a0):
    return a2*y**2 + a1*y + a0

class CameraCalibration:
    def __init__(self):
        self.objpoints = [] # 3d points in real world space
        self.imgpoints = [] # 2d points in image plane.
        
        self.ret = False
        self.mtx = []
        self.dist = []
        self.rvecs = []
        self.tvecs = []
        
        self.camera_preset() # Works only for this project
        
    def find_matchpoints(self, img, nx, ny):
        '''
        Collect sets of objpoints & imgpoints for finding distortion coefficients of the camera
        Input image should be chessboard image, read by cv2.imread
        '''
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.imshape = gray.shape[::-1]
        
        self.ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        if self.ret == True:
            self.objpoints.append(objp)
            self.imgpoints.append(corners)
        else:
            pass
            
    def calibrate_camera(self):
        '''
        Calibrate camera with stacked objpoints & imgpoints
        '''
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.imshape, None, None)
        
    def undistort_image(self, img):
        '''
        Provide undistorted image with found camera coefficients
        Input image should be read by cv2.imread
        '''
        undistimg = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undistimg
    
    def camera_preset(self):
        '''
        Preset function for this project.
        '''
        self.images_format = './camera_cal/calibration'
        
        for idx in range(1, 21):
            img = cv2.imread(self.images_format + str(idx) + '.jpg')
            if idx == 1:
                self.find_matchpoints(img, 9, 5)
            elif idx == 4:
                self.find_matchpoints(img, 7, 4)
            elif idx == 5:
                self.find_matchpoints(img, 7, 5)
            else:
                self.find_matchpoints(img, 9, 6)
        self.calibrate_camera()
        
class Line():
    def __init__(self):
        '''
        Lane line class variables
        '''
        # frame number
        self.framenum = 0
        # iterations for averaging
        self.maxiter = 5
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([0,0,0]) 
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # distance in meters of vehicle center from the line
        self.line_base_pos = None 
        # x values for detected line pixels
        self.allx = None  
        # y values for detected line pixels
        self.ally = None
        
    def update(self, fit, xpts, ypts, x_eval, y_eval):
        '''
        Update lane line class variables each step
        '''
        self.framenum += 1
        self.allx = xpts
        self.ally = ypts
        if self.framenum < self.maxiter:
            self.best_fit = (self.best_fit*(self.framenum-1) + fit)/self.framenum
        else:
            self.best_fit = (self.best_fit*(self.maxiter-1) + fit)/self.maxiter
            
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.radius_of_curvature = ((1 + (2*self.best_fit[0]*y_eval*ym_per_pix + 
                                          self.best_fit[1])**2)**1.5)/np.absolute(2*self.best_fit[0])
        self.line_base_pos = xm_per_pix * np.absolute(x_eval - 
                                                      poly_ord2(y_eval,self.best_fit[0],
                                                                self.best_fit[1],self.best_fit[2]))
        

class LaneDetection():
    def __init__(self):
        # Camera Calibration Preset
        self.cc = CameraCalibration()
        
        # Lane Line class
        self.leftline = Line()
        self.rightline = Line()
        
        # Perspective Transform Preset
        self.src = []
        self.dst = []
        self.M = []
        self.Minv = []
    
    def obtain_perspective(self, img):
        '''
        Preset function for this project.
        '''
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
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
    
    def warp_perspective(self, img):
        '''
        Obtain warped image with given perspective transform.
        '''
        warpedimg = cv2.warpPerspective(img, self.M, (img.shape[1],img.shape[0]))
        return warpedimg
    
    def unwarp_perspective(self, img):
        '''
        Obtain unwarped image with given inverse perspective transform.
        '''
        unwarpedimg = cv2.warpPerspective(img, self.Minv, (img.shape[1],img.shape[0]))
        return unwarpedimg
    
    def threshold_image(self, undistimg, sx_thresh=(20,100), sx_kernel=7, s_thresh=(120,255)):
        '''
        Thresholds on Sobel x & S kernel within HLS color space
        '''
        # Median filter image
        medfiltimg = cv2.medianBlur(undistimg, 5)
        
        # Convert to HLS color space and separate the V channel
        # hls = cv2.cvtColor(medfiltimg, cv2.COLOR_BGR2HLS)
        hls = cv2.cvtColor(undistimg, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        
        # Sobel x onto l_channel & Threshold x gradient
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sx_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold S channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        # Stack each channel
        binaryimg = (sxbinary | s_binary) * 255
        return binaryimg
    
    def search_laneline_window(self, img, binaryimg, nwindows=9, margin=100, minpix=200):
        '''
        Search lane lines via histogram peaks and window averaging
        '''
        ## Pipeline ##
        # Perspective Transform
        self.obtain_perspective(binaryimg)
        binary_warped = self.warp_perspective(binaryimg)
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Create an output image to draw on and visualize the result
        out_img = np.zeros_like(img)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
    
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each using `curve_fit`
        left_fit, _ = curve_fit(poly_ord2, lefty, leftx)
        right_fit, _ = curve_fit(poly_ord2, righty, rightx)
        
        # Select 'reliable lane line' and
        # re-fit 'unreliable lane line' to be parallel if its 2nd order term is out of 10% bound
        if left_lane_inds.shape[0] > right_lane_inds.shape[0]:
            a2bound1 = 0.9*left_fit[0]
            a2bound2 = 1.1*left_fit[0]
            a2lowerbound = a2bound1 if a2bound1<a2bound2 else a2bound2
            a2upperbound = a2bound2 if a2bound2>=a2bound1 else a2bound1
            a1bound1 = 0.9*left_fit[1]
            a1bound2 = 1.1*left_fit[1]
            a1lowerbound = a1bound1 if a1bound1<a1bound2 else a1bound2
            a1upperbound = a1bound2 if a1bound2>=a1bound1 else a1bound1
            
            
            if ((right_fit[0] > a2lowerbound) & (right_fit[0] < a2upperbound) &
                (right_fit[1] > a1lowerbound) & (right_fit[1] < a1upperbound)):
                pass
            else:
                right_fit, _ = curve_fit(poly_ord2, righty, rightx, 
                                         bounds=([a2lowerbound,a1lowerbound,-np.inf],
                                                 [a2upperbound,a1upperbound,np.inf]))
        else:
            a2bound1 = 0.9*right_fit[0]
            a2bound2 = 1.1*right_fit[0]
            a2lowerbound = a2bound1 if a2bound1<a2bound2 else a2bound2
            a2upperbound = a2bound2 if a2bound2>=a2bound1 else a2bound1
            a1bound1 = 0.9*right_fit[1]
            a1bound2 = 1.1*right_fit[1]
            a1lowerbound = a1bound1 if a1bound1<a1bound2 else a1bound2
            a1upperbound = a1bound2 if a1bound2>=a1bound1 else a1bound1
            
            if ((left_fit[0] > a2lowerbound) & (left_fit[0] < a2upperbound) &
                (left_fit[1] > a1lowerbound) & (left_fit[1] < a1upperbound)):
                pass
            else:
                left_fit, _ = curve_fit(poly_ord2, lefty, leftx, 
                                        bounds=([a2lowerbound,a1lowerbound,-np.inf],
                                                [a2upperbound,a1upperbound,np.inf]))
        
        ## Update lane line data ##
        self.leftline.update(left_fit, leftx, lefty, img.shape[1]/2, img.shape[0])
        self.rightline.update(right_fit, rightx, righty, img.shape[1]/2, img.shape[0])
        
        ## Visualization ##
        # Mark on left, right line pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        
        # Mark on lane area
        window_img = np.zeros_like(out_img)
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = poly_ord2(ploty, self.leftline.best_fit[0], 
                              self.leftline.best_fit[1], self.leftline.best_fit[2])
        right_fitx = poly_ord2(ploty, self.rightline.best_fit[0],
                               self.rightline.best_fit[1], self.rightline.best_fit[2])
        
        lane_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        lane_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_pts = np.hstack((lane_window1, lane_window2))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([lane_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Unwarp to original image via reverse perspective transform
        result_unwarped = self.unwarp_perspective(result)
        
        return result_unwarped
    
    def search_laneline_nearpoly(self, img, binaryimg, margin=100):
        '''
        Search lane lines using polynominal fitting of previous step
        '''
        ## Pipeline ##
        # Perspective Transform
        self.obtain_perspective(binaryimg)
        binary_warped = self.warp_perspective(binaryimg)
        
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox > (poly_ord2(nonzeroy,self.leftline.best_fit[0],
                                                 self.leftline.best_fit[1],self.leftline.best_fit[2]) - margin)) &
                          (nonzerox < (poly_ord2(nonzeroy,self.leftline.best_fit[0],
                                                 self.leftline.best_fit[1],self.leftline.best_fit[2]) + margin)))
        right_lane_inds = ((nonzerox > (poly_ord2(nonzeroy,self.rightline.best_fit[0],
                                                  self.rightline.best_fit[1],self.rightline.best_fit[2]) - margin)) &
                           (nonzerox < (poly_ord2(nonzeroy,self.rightline.best_fit[0],
                                                  self.rightline.best_fit[1],self.rightline.best_fit[2]) + margin)))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        # Fit a second order polynomial to each using `curve_fit`
        left_fit, _ = curve_fit(poly_ord2, lefty, leftx)
        right_fit, _ = curve_fit(poly_ord2, righty, rightx)
        
        # Select 'reliable lane line' and
        # re-fit 'unreliable lane line' to be parallel if its 2nd order term is out of 10% bound
        if left_lane_inds.shape[0] > right_lane_inds.shape[0]:
            a2bound1 = 0.9*left_fit[0]
            a2bound2 = 1.1*left_fit[0]
            a2lowerbound = a2bound1 if a2bound1<a2bound2 else a2bound2
            a2upperbound = a2bound2 if a2bound2>=a2bound1 else a2bound1
            a1bound1 = 0.9*left_fit[1]
            a1bound2 = 1.1*left_fit[1]
            a1lowerbound = a1bound1 if a1bound1<a1bound2 else a1bound2
            a1upperbound = a1bound2 if a1bound2>=a1bound1 else a1bound1
            
            
            if ((right_fit[0] > a2lowerbound) & (right_fit[0] < a2upperbound) &
                (right_fit[1] > a1lowerbound) & (right_fit[1] < a1upperbound)):
                pass
            else:
                right_fit, _ = curve_fit(poly_ord2, righty, rightx, 
                                         bounds=([a2lowerbound,a1lowerbound,-np.inf],
                                                 [a2upperbound,a1upperbound,np.inf]))
        else:
            a2bound1 = 0.9*right_fit[0]
            a2bound2 = 1.1*right_fit[0]
            a2lowerbound = a2bound1 if a2bound1<a2bound2 else a2bound2
            a2upperbound = a2bound2 if a2bound2>=a2bound1 else a2bound1
            a1bound1 = 0.9*right_fit[1]
            a1bound2 = 1.1*right_fit[1]
            a1lowerbound = a1bound1 if a1bound1<a1bound2 else a1bound2
            a1upperbound = a1bound2 if a1bound2>=a1bound1 else a1bound1
            
            if ((left_fit[0] > a2lowerbound) & (left_fit[0] < a2upperbound) &
                (left_fit[1] > a1lowerbound) & (left_fit[1] < a1upperbound)):
                pass
            else:
                left_fit, _ = curve_fit(poly_ord2, lefty, leftx, 
                                        bounds=([a2lowerbound,a1lowerbound,-np.inf],
                                                [a2upperbound,a1upperbound,np.inf]))
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.zeros_like(img)
        window_img = np.zeros_like(out_img)
        
        ## Update lane line data ##
        self.leftline.update(left_fit, leftx, lefty, img.shape[1]/2, img.shape[0])
        self.rightline.update(right_fit, rightx, righty, img.shape[1]/2, img.shape[0])
        
        ## Visualization ##
        # Mark on left, right line pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        
        # Mark on lane area
        window_img = np.zeros_like(out_img)
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = poly_ord2(ploty, self.leftline.best_fit[0], 
                              self.leftline.best_fit[1], self.leftline.best_fit[2])
        right_fitx = poly_ord2(ploty, self.rightline.best_fit[0],
                               self.rightline.best_fit[1], self.rightline.best_fit[2])
        
        lane_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        lane_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_pts = np.hstack((lane_window1, lane_window2))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([lane_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Unwarp to original image via reverse perspective transform
        result_unwarped = self.unwarp_perspective(result)
        
        return result_unwarped
    
    def process_image(self, img):
        undistimg = self.cc.undistort_image(img)
        binaryimg = self.threshold_image(undistimg)
        if self.leftline.framenum == 0 & self.rightline.framenum == 0:
            out_img = self.search_laneline_window(img, binaryimg)
        else:
            out_img = self.search_laneline_nearpoly(img, binaryimg)
            
        result = cv2.addWeighted(undistimg, 1, out_img, 0.5, 0)
        
        font = cv2.FONT_ITALIC
        fontScale = 0.75
        thickness = 1
        location = (100, 100)
        cv2.putText(result, 'Radius of Curvature (Left) = '+str(self.leftline.radius_of_curvature)+'m', 
                    location, font, fontScale, (255,255,255), thickness)
        location = (100, 125)
        cv2.putText(result, 'Radius of Curvature (Right) = '+str(self.rightline.radius_of_curvature)+'m', 
                    location, font, fontScale, (255,255,255), thickness)
        location = (100, 150)
        offset = np.absolute(self.leftline.line_base_pos - self.rightline.line_base_pos)/2
        if self.leftline.line_base_pos > self.rightline.line_base_pos:
            cv2.putText(result, 'Offset from Lane Center = '+str(offset)+'m to the Right', 
                        location, font, fontScale, (255,255,255), thickness)
        else:
            cv2.putText(result, 'Offset from Lane Center = '+str(offset)+'m to the Left', 
                        location, font, fontScale, (255,255,255), thickness)
        
        return result
            
        