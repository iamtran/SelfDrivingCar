import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from myLib.featureExtraction  import *
#from myLib.vehicle_detector   import *

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
# Define a single function that can extract features using hog sub-sampling and make predictions


# Missing the control of
# hog_channel  -- always on ALL can not deal with 0, 1,2,
# spatial_feat -- always on ?
# hist_feat    -- always on ?
# hog_feat     -- always on ?

def find_cars (img,  
                    ystart, ystop, scale, 
                    svc, X_scaler, 
                    color_space, 
                    orient, 
                    pix_per_cell, cell_per_block, 
                    spatial_size, hist_bins, hog_channel ):
    #draw_img = np.copy(img)
    #img = img.astype(np.float32)/255 # why you want to do this ?
    
    img_tosearch = img[ystart:ystop,:,:]
    # color_space = 'HSV' 'LUV' 'HLS' 'YUV' 'YCrCb'
    ctrans_tosearch = change_color_space(img_tosearch, color_space)
    #print (type (ctrans_tosearch))
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    #window = 64
    window = pix_per_cell * pix_per_cell
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        if   (hog_channel == 1):
            hogx = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        elif (hog_channel == 2):
            hogx = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        elif (hog_channel == 3):
            hogx = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        else :
            hogx = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        
    window_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else :
                hog_feat1 = hogx[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1))
                
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features    = color_hist (subimg, nbins=hist_bins)
            
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
   
            test_prediction = svc.predict(test_features)  

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop  * scale)
                win_draw  = np.int(window* scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                window_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    #return draw_img, window_list
    return  window_list
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def build_test_data (files):
    images = glob.glob(files)
    cars    = []
    notcars = []
    #print(files)
    for image in images:
        if 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)    

    return cars, notcars, len(cars), len(notcars)
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def build_simple_data (files):
    images = glob.glob(files)
    cars    = []
    for image in images:
        cars.append(image)    
    return cars,len(cars)
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
# all the pixel in the box will have counter increase by 1
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        #print (box)
        #print (box[0][1],box[1][1], box[0][0],box[1][0] )
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def manage_heatmap (image, box_list, threshold):
        heat    = np.zeros_like(image[:,:,0]).astype(np.float)
        heat    = add_heat(heat,box_list)
        #print ("manage_heatmap")
        heat    = apply_threshold(heat,threshold)
        heatmap = np.clip(heat, 0, 255)
        return heatmap
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def draw_multi_scale_windows(img, ystart, ystop, scale, pix_per_cell):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    imshape = img_tosearch.shape
    img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # Define blocks and steps as above
    nxblocks = (img_tosearch.shape[1] // pix_per_cell) - 1
    nyblocks = (img_tosearch.shape[0] // pix_per_cell) - 1
    #nfeat_per_block = orient * cell_per_block ** 2

    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    rect_start = None
    rect_end = None
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            xbox_left = np.int(xleft * scale)
            ytop_draw = np.int(ytop * scale)
            win_draw = np.int(window * scale)
            rect_start = (xbox_left, ytop_draw + ystart)
            rect_end = (xbox_left + win_draw, ytop_draw + win_draw + ystart)
            cv2.rectangle(draw_img, rect_start, rect_end, (0, 0, 255), 6)
    
    cv2.rectangle(draw_img, rect_start, rect_end, (255, 0, 0), 6)

    return draw_img
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------

def automate_training (cars, notcars, my_features, model_name):
    car_features    = extract_features_param (cars   , my_features)
    notcar_features = extract_features_param (notcars, my_features)
    
    X_train, X_test, y_train, y_test, X_scaler = create_test_training_data (car_features, notcar_features)
    svc = train_model (X_train, X_test, y_train, y_test) ## shuffle data in train_model()
    
    color_space    = my_features['color_space']
    orient         = my_features['orient']
    pix_per_cell   = my_features['pix_per_cell']
    cell_per_block = my_features['cell_per_block']
    hog_channel    = my_features['hog_channel']
    spatial_size   = my_features['spatial_size']
    hist_bins      = my_features['hist_bins']
    
    spatial_feat   = my_features['spatial_feat']
    hist_feat      = my_features['hist_feat']
    hog_feat       = my_features['hog_feat']
    
    pFile = model_name 
    #print (model_name)

    data={
        'svc'           : svc,
        'X_scaler'      : X_scaler,
        'color_space'   : color_space,
        'orient'        : orient,
        'pix_per_cell'  : pix_per_cell,
        'cell_per_block': cell_per_block,
        'spatial_size'  : spatial_size,
        'hist_bins'     : hist_bins,
        'hog_channel'   : hog_channel
         }

    with open(pFile, 'wb') as pFile:
        pickle.dump(data, pFile) 
    return  X_train, X_test, y_train, y_test, model_name  
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------        
def automate_testing (image, ystart, ystop, scale, my_features, model_name, X_train, X_test, y_train, y_test ):
    pfile = model_name
    dist_pickle    = pickle.load( open(pfile, "rb" ) )
    svc            = dist_pickle["svc"]
    X_scaler       = dist_pickle["X_scaler"]
    
    color_space    = dist_pickle["color_space"]
    orient         = dist_pickle["orient"]
    pix_per_cell   = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size   = dist_pickle["spatial_size"]
    hist_bins      = dist_pickle["hist_bins"]
    hog_channel    = dist_pickle["hog_channel"]
    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    acc = 100* round(svc.score(X_test, y_test), 4)
    print('%s Test Accuracy  = %f %% '% (model_name, acc ))
    #print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    test_model_prediction (svc, X_test, y_test, n_predict = 10 )
 
    # note color space will be RGB2YCrCb and all for hog_channel
    # no nog_channel
    # no color_space
    
    window_list= find_cars(image, \
                ystart, ystop, scale, \
                svc, X_scaler, \
                color_space, orient, pix_per_cell, cell_per_block,  spatial_size, hist_bins, hog_channel)
    out_img = draw_boxes(image, window_list , color=(0, 0, 255), thick=6)
    plot_2_images(image, out_img, "ORIGINAL", color_space)
    return out_img, window_list

