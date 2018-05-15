import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
#%matplotlib inline
# file : findLaneUtils.py
#
# project files:
# './calibrationImages/   -- images to calibrate
# AdvancedLaneFindingReport.ipny
# findLaneUtils.py

import pickle
def save_camera_info (pickle_file, CameraMat, distCoeff, M, Minv):
    dist_pickle = {}
    dist_pickle["CameraMat"] = CameraMat
    dist_pickle["distCoeff"] = distCoeff
    dist_pickle["M"]         = M
    dist_pickle["Minv"]      = Minv
    pickle.dump( dist_pickle, open( pickle_file, "wb" ) )

def load_camera_info (pickle_file):
    dist_pickle = pickle.load( open( pickle_file, "rb" ) )
    CameraMat   = dist_pickle["CameraMat"]
    distCoeff   = dist_pickle["distCoeff"]
    M           = dist_pickle["M"]
    Minv        = dist_pickle["Minv"]                             
    return CameraMat, distCoeff, M, Minv

def plot_2_images(img1, img2, title1='Original Image', title2='Thresholded S' ):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title( title1, fontsize=50)
    ax2.imshow(img2)
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
def plot_3_images(img1, img2, img3, title1='Original Image', title2='Thresholded S', title3='title 3' ):
    # Plot the result
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    
    ax1.imshow(img1)
    ax1.set_title( title1, fontsize=50)
    
    ax2.set_title(title2, fontsize=50)
    ax2.imshow(img2)
    
    ax3.set_title(title3, fontsize=50)
    ax3.imshow(img3)
    
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def CalculateCameraMatrix (calibration_dir):
    #import numpy as np
    nx = 9 #  8#7#4#8 #0#TODO: enter the number of inside corners in x
    ny = 6 #  6#4#6 #0#TODO: enter the number of inside corners in y
    objpoints = [] #3d points in real world space
    imgpoints = [] #2D points in image plane
    objp = np.zeros((6*9,3), np.float32)
    #objp = numpy.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)# x,y coordinates
    
    # read in and make a list of calibration images
    image_files = calibration_dir + '/calibration*.jpg'
    # read in and make a list of calibration images
    images = glob.glob('./calibrationImages/calibration*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        #plt.imshow(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #plt.imshow(img)     
    
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3,thresh=(0,255)):
    thresh_min=thresh[0]
    thresh_max=thresh[1]
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separatel(Sobel x and y gradients)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    #   Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # 5) Create a binary mask where mag thresholds are met
    # Create a binary image of ones where threshold is met, 
    # zeros otherwise
    binary_output = np.zeros_like(gradmag)
    # 6) Return this mask as your binary_output image
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    binary_output =  np.zeros_like(absgraddir)
    # 5) Create a binary mask where direction thresholds are met
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    
    return binary_output

# Threshold color channel s_thresh=(170, 255)
# Threshold x gradient sx_thresh=(20, 100)

def threshold_s_sx (img0, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img0)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    #l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    #plt.imshow(l_channel)
    #plt.imshow(s_channel)
    
    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.imshow(gray)
    
    # Sobel x # Take the derivative in x
    #sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    #plt.imshow(sobelx)
    
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx) 
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    #plt.imshow(scaled_sobel)
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    #plt.imshow(sxbinary)
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    #plt.imshow(s_binary)
    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. 
    # It might be beneficial to replace this channel with something else.
    # Stack each channel to view their individual contributions in green and
    # blue respectively
    # This returns a stack of the two binary images, whose components you 
    # can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    cv2.imwrite('prj_img/color_binary2.jpg', color_binary )
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return color_binary, combined_binary


def superImposeImage (image, left_fitx, right_fitx, ploty ):
    print (image.shape)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    implot = plt.imshow(image)
    #print (implot)
    #fig.savefig('/tmp/test.png')
    return fig

def laneDetection (binary_warped):
   
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #plt.imshow(out_img)
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #print (leftx_base, midpoint, rightx_base)
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current  = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
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
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
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
     
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit  = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx  = left_fit[0] *ploty**2 + left_fit [1]*ploty + left_fit [2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return out_img, left_fit, right_fit


def laneDetectionNext (binary_warped, left_fit, right_fit): 
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 100

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin))) 
    
    # Again, extract left and right line pixel positions
    leftx  = nonzerox[left_lane_inds]
    lefty  = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]   
    
    # Fit a second order polynomial to each
    left_fit  = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)   
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx  = left_fit [0]*ploty**2 + left_fit [1]*ploty + left_fit [2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 
    
    #print (len(leftx), len(lefty), len(ploty))
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)  
    
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]]   = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly (window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly (window_img, np.int_([right_line_pts]), (0,255, 0))
    cv2.polylines(window_img, np.int_([right_line_pts]),True,(0,255,255), thickness=10)
                  
    fin_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)


# Example values: 632.1 m    626.2 m

    return fin_img, leftx, lefty, rightx, righty, ploty

def discoverCurvature (ploty, leftx, lefty,  rightx, righty):
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    y_eval = np.max(ploty)
    #print (y_eval)
    # Fit new polynomials to x,y in world space
    #print(len (ploty * ym_per_pix), len(leftx *xm_per_pix))
    #left_fit_cr  = np.polyfit(ploty * ym_per_pix, leftx *xm_per_pix, 2)
    #right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx*xm_per_pix, 2)
    
    left_fit_cr  = np.polyfit(lefty * ym_per_pix, leftx *xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad  = ((1 + (2*left_fit_cr[0] *y_eval*ym_per_pix + left_fit_cr[1]) **2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Now our radius of curvature is in meters
    #print("left curve:", left_curverad, 'm', "right curve:", right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return left_curverad, right_curverad

# def warpback_display (undistorted_img, Minv, ploty, right_curverad ):
# Minv
# undistorted_img
# y_eval = np.max(ploty)
# right_curverad
# midx = 650
#     ym_per_pix = 30/720 # meters per pixel in y dimension
#    xm_per_pix = 3.7/700 # meters per pixel in x dimension
#
#def warpback_display (undistorted_img, binary_warped, left_fit, right_fit,Minv, ploty, right_curverad ):
def warpback_display (undistorted_img, binary_warped, left_fit, right_fit,Minv,  right_curverad ):
    #undistorted_img = img1 # dst
    #plt.imshow (undistorted_img)
    ploty      = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero  = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left   = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right  = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts        = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp    = cv2.warpPerspective(color_warp, Minv, (undistorted_img.shape[1], undistorted_img.shape[0])) 

    # Combine the result with the original image
    result     = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
    curvature = right_curverad
    cv2.putText(result,'Radius of Curvature: %.2fm' % curvature,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    ################
    y_eval = np.max(ploty)
    midx = 650
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ################   
    x_left_pix  = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
    x_right_pix = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
    position_from_center = ((x_left_pix + x_right_pix)/2 - midx) * xm_per_pix
    if position_from_center < 0:
        text = 'left'
    else:
        text = 'right'
    cv2.putText(result,'Distance From Center: %.2fm %s' % (np.absolute(position_from_center), text),(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    #plt.imshow(result)
    return result

def create_binary_swarped (img1, CameraMat, distCoeff, M):
    undistorted_img = cv2.undistort(img1, CameraMat, distCoeff, None, CameraMat)
    #color_binary, combined_binary = threshold_s_sx(undistorted_img, s_thresh, sx_thresh)
    color_binary, combined_binary = threshold_s_sx(undistorted_img)
    img_size = (combined_binary.shape[1], combined_binary.shape[0])
    binary_warped = cv2.warpPerspective(combined_binary,M, img_size)
    return binary_warped, undistorted_img
