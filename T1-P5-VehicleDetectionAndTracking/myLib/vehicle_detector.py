import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

import sys, os

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

import time

from myLib.featureExtraction  import *
from myLib.search_window      import *
from collections import deque

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
class VehicleDetector:
    def __init__(self, model_file):
        # Loading Model Parameters
        dist_pickle    = pickle.load( open(model_file, "rb" ) )
        self.svc            = dist_pickle["svc"]
        self.X_scaler       = dist_pickle["X_scaler"]
        self.color_space    = dist_pickle["color_space"]
        self.orient         = dist_pickle["orient"]
        self.pix_per_cell   = dist_pickle["pix_per_cell"]
        self.cell_per_block = dist_pickle["cell_per_block"]
        self.spatial_size   = dist_pickle["spatial_size"]
        self.hist_bins      = dist_pickle["hist_bins"]
        self.hog_channel    = dist_pickle["hog_channel"]      

        # Current Frame Count
        self.frame_count = 0

        # Various Scales
        self.ystart_ystop_scale = [   (400, 600, 1.0), (500, 600, 1.0) ] 
        # Current HeatMap
        self.heatmap     = None
        self.queueLength = 8
        self.heatQ       = deque(maxlen=self.queueLength) 

        self.box_list    = []
        # Threshold for Heatmap
        self.threshold          = 2
        self.heat_scan_interval = 20    # NOT USE?
        self.scan_interval      = 5     # NOT USE?
        self.all_heatmap        = None  # NOT USE?       

        
#------------------------------------------------------------------------
# Working and no heap map.
# introduce concept of heatMapQueue
#------------------------------------------------------------------------
    def find_other_cars(self, img):
        svc            = self.svc
        X_scaler       = self.X_scaler
        color_space    = self.color_space 
        orient         = self.orient
        pix_per_cell   = self.pix_per_cell
        cell_per_block = self.cell_per_block
        spatial_size   = self.spatial_size
        hist_bins      = self.hist_bins
        hog_channel    = self.hog_channel

        heat = np.zeros_like(img[:,:,0]).astype(np.float) # Why float.  Should be integer?
        
        self.box_list = []
        draw_img = np.copy(img)

        if self.frame_count % self.heat_scan_interval == 0:
            self.all_heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
      
        #if ( (self.frame_count % self.scan_interval)== 0):
        loop = 0
        for (ystart, ystop, scale) in self.ystart_ystop_scale:
            self.box_list.extend ( find_cars (img, 
                    ystart, ystop, scale, 
                    svc, X_scaler, 
                    color_space, orient, 
                    pix_per_cell, cell_per_block, 
                    spatial_size, hist_bins, hog_channel))      
            loop +=1
        #print (loop, len(self.box_list), len(self.ystart_ystop_scale))
        heat = add_heat(heat, self.box_list)        
        heat = apply_threshold(heat, (self.threshold + loop -1 ))
        
        # operation before drawing image. need np.clip
        
        self.heatmap = np.clip(heat, 0, 255)
        
        labels   = label(self.heatmap)
        draw_img = draw_labeled_bboxes(draw_img, labels)   
        
        self.frame_count += 1
        return draw_img        

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------   
def auto_test_detector (imageFile, scale, modelFile='modelRGB.pk' ):

    detector = VehicleDetector (modelFile) # ('model-params-hsv2.pk')
    detector.ystart_ystop_scale = scale
    #detector.threshold = 3
    img = mpimg.imread(imageFile)
    out = detector.find_other_cars(img)
    plot_2_images (out, detector.all_heatmap,"Label Image", "Total Heat Map")
    plot_2_images (out, detector.heatmap    ,"Label Image", "Local Image Heat Map")
    return detector
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------   
def auto_test_detector2 (imageFile, scale, modelFile, detector):
    img = mpimg.imread(imageFile)
    out = detector.find_other_cars(img)
    plot_2_images (out, detector.all_heatmap,"Label Image", "Total Heat Map")
    plot_2_images (out, detector.heatmap    ,"Label Image", "Local Image Heat Map")
    return detector 
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def auto_test_detector1 (imageFile, scale, modelFile, detector):
    img = mpimg.imread(imageFile)
    out = detector.find_other_cars(img)
    #plot_2_images (out, detector.all_heatmap,"Label Image", "Total Heat Map")
    plot_2_images (out, detector.heatmap    ,"Label Image", "Local Image Heat Map")
    return detector