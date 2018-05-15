# BehavioralCloning_DataSet.py
from myLib.BehavioralCloning_ImportFile import *

"""
The driving_log.csv 's format:
1. CenterImage
2. LeftImage
3. RightImage
4. Steering (-1..1)
5. Throttle (0..1)
6. Break    (0)
7. Speed    (0..30)

"""
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def get_data (data_dir) :
    cvs_file=data_dir+"/driving_log.csv"
    lines = []
    with open (cvs_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    return lines

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
# BUG: Why I am reading center right and left image at the same time ?
#

def load_data (data_dir):
    images = []
    measurements = []
    lines = get_data (data_dir)
    for line in lines:
        for i in range(3):
            source_path = line[i]
            #print (source_path)
            filename = source_path.split('\\')[-1]
            #print(filename)
            current_path = data_dir+'/IMG/' + filename
            #print (current_path)
            image = cv2.imread(current_path)
            images.append(image)
            measurement = float (line[3])
            measurements.append(measurement)
            
    print ("Data Set Length is : ", len(lines), len(images), len(measurements))
    return images, measurements
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def get_file_name (data_dir, line, index):
    source_path  = line[index]
    filename     = source_path.split('\\')[-1]
    current_path = data_dir + '/IMG/' + filename
    return current_path

def process_image (filename):
    image = cv2.imread(filename)
    return image 

def load_data2 (data_dir):
    car_images      = []
    steering_angles = []
    lines = get_data (data_dir)
    for row in lines:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction     = 0.2 # this is a parameter to tune
        steering_left  = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = "..." # fill in the path to your training IMG directory
        
        img_center = process_image(get_file_name(data_dir, row,0))
        img_left   = process_image(get_file_name(data_dir, row,1))
        img_right  = process_image(get_file_name(data_dir, row,2))

        # add images and angles to data set
        car_images.append(img_center)
        car_images.append(img_left)
        car_images.append(img_right)
        steering_angles.append(steering_center);
        steering_angles.append(steering_left);
        steering_angles.append(steering_right)
    print ("Data Set Length is : ", len(lines), len(car_images), len(steering_angles))
    return car_images, steering_angles
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def augmentation_data(images, measurements):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip (images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)
    #------------------------------------------------
    print ("Data Set Length is : ", len(augmented_images), len(augmented_measurements))
    
    return augmented_images, augmented_measurements

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
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------    

    
    
    
    
    """
def load_data (lines, data_dir):
    images = []
    measurements = []
    for line in lines:
        for i in range(3):
            source_path = line[i]
            #print (source_path)
            filename = source_path.split('\\')[-1]
            #print(filename)
            current_path = data_dir+'/IMG/' + filename
            #print (current_path)
            image = cv2.imread(current_path)
            images.append(image)
            measurement = float (line[3])
            measurements.append(measurement)
    return images, measurements
"""