#
# proj_pipeline.py
#
import numpy as np
import cv2
from    findLaneUtils import *
from proj_lane import *

class DrivingPipeLine:
    imageCount = 0
    refeshRate = 7
    def __init__(self):      
        self.line       = None
        self.M          = None
        self.Minv       = None
        self.cameraMat  = None
        self.distCoeffs = None
        self.another_count = 5
        #print ("Driving PipeLine Initialized :", self.imageCount)
        
    #@staticmethod
    def __init__(self, cameramtx, distcoeff, m, minv ):
        self.line          = Lane()
        refeshRate         = 7
        self.M             = m
        self.Minv          = minv
        self.cameraMat     = cameramtx
        self.distCoeffs    = distcoeff
        
   
        
    def set_values(self, line, cameramtx, distcoeff, m, minv, refresh):
        self.line          = line
        self.M             = m
        self.Minv          = minv
        self.cameraMat     = cameramtx
        self.distCoeffs    = distcoeff
        self.refeshRate    = refresh

        
    def print_values (self):
        print ("imageCount    : ", self.imageCount)
    def test_set (self):
        self.imageCount +=1
        
    #@staticmethod
    def pipeline(self, img):
        self.imageCount += 1
        #print ("Driving PipeLine Started :", self.imageCount)
        binary_warped, undistorted_img = create_binary_swarped (img, self.cameraMat, self.distCoeffs, self.M)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255 ### <<<
        if ((self.imageCount % self.refeshRate)==1):
            #print ("Driving PipeLine Started :", self.imageCount)
            #print ("First Image: ", self.imageCount)
            out_img, left_fit, right_fit = laneDetection (binary_warped)
            self.line.update_fits(left_fit, right_fit)
            self.line.first_frame_processed = True
        else :
            #print ("Next Image: ", self.imageCount)
            out_img2,  leftx, lefty, rightx, righty, ploty = laneDetectionNext (binary_warped, self.line.left_fit, self.line.right_fit)
            self.line.update_fits(np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2))
            #left_curverad, right_curverad = discoverCurvature (ploty, leftx, lefty, rightx, righty )  
        right_curverad = self.line.curvature    
        result = warpback_display (undistorted_img, binary_warped, 
                                       self.line.left_fit, self.line.right_fit,self.Minv,  right_curverad )          
        
        return result
