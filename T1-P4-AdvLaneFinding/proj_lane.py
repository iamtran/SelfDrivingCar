#
# proj_lane.py
#
import numpy as np

class Lane:
    def __init__(self):
        # if the first frame of video has been processed
        self.first_frame_processed = False       
        self.img           = None
        self.y_eval        = 700
        self.midx          = 640
        self.ym_per_pix    = 3.0/72.0 # meters per pixel in y dimension
        self.xm_per_pix    = 3.7/660.0 # meters per pixel in x dimension        
        self.mse_tolerance = 0.01
        self.left_fit      = [np.array([False])] 
        self.right_fit     = [np.array([False])] 
        self.curvature     = 0
       
    def set_values(first_frame_processed, img,mse_tolerance, left_fit, right_fit, y_eval, midx, ym_per_pix, xm_per_pix, curvature ):
        self.first_frame_processed = first_frame_processed  
        self.img = img
        self.mse_tolerance = mse_tolerance
        self.left_fit   = [np.array([False])] 
        self.right_fit  = [np.array([False])] 
        
        self.y_eval     = y_eval
        self.midx       = midx
        self.ym_per_pix = ym_per_pix # meters per pixel in y dimension
        self.xm_per_pix = xm_per_pix # meters per pixel in x dimension
        self.curvature  = curvature     
        
    def retrieve_values():
        return  self.mse_tolerance, \
                self.left_fit, \
                self.right_fit, \
                self.y_eval, \
                self.midx, \
                self.ym_per_pix, \
                self.xm_per_pix, \
                self.curvature

    def update_curvature(self, fit):
  
        y1 = (2*fit[0]*self.y_eval + fit[1])*self.xm_per_pix/self.ym_per_pix
        y2 = 2*fit[0]*self.xm_per_pix/(self.ym_per_pix**2)
        curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)
        
        if self.first_frame_processed:
            self.curvature = curvature
        
        elif np.absolute(self.curvature - curvature) < 500:
            self.curvature = 0.75*self.curvature + 0.25*(((1 + y1*y1)**(1.5))/np.absolute(y2)) 
            
            
    def update_fits(self, left_fit, right_fit):
        if self.first_frame_processed:
            left_error  = ((self.left_fit[0]  - left_fit[0])  ** 2).mean(axis=None)      
            right_error = ((self.right_fit[0] - right_fit[0]) ** 2).mean(axis=None)
            
            if left_error < self.mse_tolerance:
                self.left_fit = 0.75 * self.left_fit + 0.25 * left_fit  
                
            if right_error < self.mse_tolerance:
                self.right_fit = 0.75 * self.right_fit + 0.25 * right_fit
        else:
            self.right_fit = right_fit
            self.left_fit = left_fit
        
        self.update_curvature(self.right_fit)
     
    