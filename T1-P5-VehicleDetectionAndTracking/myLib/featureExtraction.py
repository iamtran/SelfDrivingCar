import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def change_color_space (img, color_space='RGB'):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)  
    return feature_image
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    feature_image = change_color_space (img, color_space)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features


#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, 
                                  orientations    = orient, 
                                  pixels_per_cell = ( pix_per_cell, pix_per_cell), 
                                  cells_per_block = (cell_per_block, cell_per_block), 
                                  transform_sqrt  = True,  
                                  visualise       = vis, 
                                  feature_vector  = feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, 
                       orientations    = orient, \
                       pixels_per_cell = (pix_per_cell, pix_per_cell),\
                       cells_per_block = (cell_per_block, cell_per_block), \
                       transform_sqrt  = True, \
                       visualise       = vis, \
                       feature_vector  = feature_vec)
        return features

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def get_hog_features_special (feature_image, hog_channel, orient, 
                                        pix_per_cell, 
                                        cell_per_block, 
                                        vis=False, 
                                        feature_vec=True):
    if not ((hog_channel == 'ALL') or (hog_channel == 0) or (hog_channel == 1) or (hog_channel ==2) ):
        print ("hot_channel is : %d . Not Correct. Reset to 0" % (hog_channel))
        hog_channel = 0
        
    if hog_channel == 'ALL':            
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, 
                                        pix_per_cell, 
                                        cell_per_block, 
                                        vis        =False, 
                                        feature_vec=True))
        hog_features = np.ravel(hog_features)        
    else:

                
            hog_features = get_hog_features(feature_image[:,:,hog_channel], 
                                                orient, 
                                                pix_per_cell, 
                                                cell_per_block, 
                                                vis        =False, 
                                                feature_vec=True)
    return hog_features
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def read_image (file):
    #print (file)
    end=file.split('.')[1]
    #print (end)
    if (end =="png"):
        src = cv2.imread(file)
        image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    elif (end == "jpg"):
        image = mpimg.imread(file)
    else :
        print ("Reading image problem : ", file)
        print ("Expect file ending jpg or png but receive: ", end)
        image = None
    return image
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, 
                     color_space    = 'RGB', 
                     spatial_size   = (32, 32),
                     hist_bins      = 32, 
                     orient         = 9, 
                     pix_per_cell   = 8, 
                     cell_per_block = 2, 
                     hog_channel    = 0,
                     spatial_feat   = True, 
                     hist_feat      = True, 
                     hog_feat       = True):
    
    # Create a list to append feature vectors to
    if not ((hog_channel == 'ALL') or (hog_channel == 0) or (hog_channel == 1) or (hog_channel ==2) ):
        print ("hot_channel is : %d . Not Correct. Reset to 0" % (hog_channel))
        hog_channel = 0
        
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = read_image (file)
         
        # apply color conversion if other than 'RGB'
        feature_image = change_color_space (image, color_space)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
            
        # Apply color_hist()
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
            
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_feat == True:
            hog_features = get_hog_features_special (feature_image, 
                                                     hog_channel, 
                                                     orient, 
                                                     pix_per_cell, 
                                                     cell_per_block, 
                                                     vis        =False, 
                                                     feature_vec=True)               
                
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def extract_features_param (imgs, my_features):
    color_space   = my_features["color_space"]
    orient        = my_features["orient"]
    pix_per_cell  = my_features["pix_per_cell"]
    cell_per_block= my_features["cell_per_block"]
    hog_channel   = my_features["hog_channel"]
    spatial_size  = my_features["spatial_size"]
    hist_bins     = my_features["hist_bins"]
    spatial_feat  = my_features["spatial_feat"]
    hist_feat     = my_features["hist_feat"]
    hog_feat      = my_features["hog_feat"]
    
    data_features =  extract_features(imgs, 
                            color_space   = color_space, 
                            spatial_size  = spatial_size,
                            hist_bins     = hist_bins, 
                            orient        = orient, 
                            pix_per_cell  = pix_per_cell, 
                            cell_per_block= cell_per_block, 
                            hog_channel   = hog_channel,
                            spatial_feat  = spatial_feat,
                            hist_feat     = hist_feat, 
                            hog_feat      = hog_feat)
        
    return data_features
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
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
# Define a function to draw bounding boxes
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
def create_feature_parameters (color_space = 'YCrCb', 
        spatial_feat = True, spatial_size = (32, 32), 
        hist_feat = True, hist_bins = 32, 
        hog_feat = True, orient = 9, pix_per_cell = 8,
        cell_per_block = 2, hog_channel = 'ALL'    ):
    
    feature_dict = { 
        'color_space' : color_space,
        'spatial_size' : spatial_size,
        'hist_bins' : hist_bins,
        'orient' : orient,
        'pix_per_cell' : pix_per_cell,
        'cell_per_block' : cell_per_block,
        'hog_channel' : hog_channel,
        'spatial_feat' : spatial_feat,
        'hist_feat' : hist_feat,
        'hog_feat' : hog_feat
    }
    return feature_dict
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def create_test_training_data (car_features, notcar_features):
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))  
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    return X_train, X_test, y_train, y_test, X_scaler
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def train_model(X_train, X_test, y_train, y_test):
    # Use a linear SVC 
    
    # Check the training time for the SVC
    #t=time.time()
    svc = LinearSVC()
    X_train, y_train = shuffle(X_train, y_train)
    svc.fit(X_train, y_train)
    #t2 = time.time()
    #print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    #print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc   
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def test_model_prediction (svc, X_test, y_test, n_predict = 10 ):
    # Check the prediction time for a single sample
    t=time.time()
    print('My SVC predicts    : ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')    
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------    
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    if not ((hog_channel == 'ALL') or (hog_channel == 0) or (hog_channel == 1) or (hog_channel ==2) ):
        print ("hot_channel is : %d . Not Correct. Reset to 0" % (hog_channel))
        hog_channel = 0  
    #1) Define an empty list to receive features
    img_features = []
    
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)  
        
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        
        #4) Append features to list
        img_features.append(spatial_features)
        
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                                     orient, 
                                                     pix_per_cell, 
                                                     cell_per_block, 
                                                     vis=False, 
                                                     feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], 
                                            orient, 
                                            pix_per_cell, 
                                            cell_per_block, 
                                            vis=False, 
                                            feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
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
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------    
def plot_2_images(img1, img2, title1='Original Image', title2='Thresholded S' ):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title( title1, fontsize=50)
    ax2.imshow(img2)
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    