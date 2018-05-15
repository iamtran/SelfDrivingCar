from myLib.vehicle_detector import *

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def test_get_data (car_image_files, non_car_images_file):
    notcars, len_noncars = build_simple_data (non_car_images_file)
    cars, len_cars = build_simple_data (car_image_files)
    return cars, len_cars, notcars, len_noncars

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------

def create_model_name (my_features, data_str):
    color_space = my_features['color_space']
    pk_dir = "PK_" + str(my_features['hist_bins'])
    model_name = pk_dir + "/" + "model_"+ color_space + "_" + \
        str(my_features['hist_bins']) + "_"+ data_str +".pk"

    return model_name
def video_output_name (video_input, model_file):
    p1 = video_input.split('.')[0]
    p2 = (model_file.split ('.')[0]).split('/')[1]
    video_output = p1 + "_" + p2  + ".mp4"
    print (video_output)
    return video_output
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
my_features1 = lect_features = create_feature_parameters \
(
    color_space    = 'RGB', # # RGB HSV LUV HLS YUV YCrCb
    spatial_size   = (16, 16),
    hist_bins      = 16,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)
my_features2 = lect_features = create_feature_parameters \
(
    color_space    = 'HSV', # # RGB HSV LUV LUV YUV YCrCb
    spatial_size   = (16, 16),
    hist_bins      = 16,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)
my_features3 = lect_features = create_feature_parameters \
(
    color_space    = 'LUV', # # RGB HSV LUV HLS YUV YCrCb
    spatial_size   = (16, 16),
    hist_bins      = 16,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)
my_features4 = lect_features = create_feature_parameters \
(
    color_space    = 'HLS', # # RGB HSV LUV HLS YUV YCrCb
    spatial_size   = (16, 16),
    hist_bins      = 16,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)

my_features5 = lect_features = create_feature_parameters \
(
    color_space    = 'YUV', # # RGB HSV LUV HLS YUV YCrCb
    spatial_size   = (16, 16),
    hist_bins      = 16,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)
my_features6 = lect_features = create_feature_parameters \
(
    color_space    = 'YCrCb', # # RGB HSV LUV HLS YUV YCrCb
    spatial_size   = (16, 16),
    hist_bins      = 16,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)

pk_list_16 = [my_features1, my_features2, my_features3, my_features4,my_features5, my_features6]
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
my_features7 = lect_features = create_feature_parameters \
(
    color_space    = 'RGB', # # RGB HSV LUV HLS YUV YCrCb
    spatial_size   = (32, 32),
    hist_bins      = 32,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)
my_features8 = lect_features = create_feature_parameters \
(
    color_space    = 'HSV', # # RGB HSV LUV LUV YUV YCrCb
    spatial_size   = (32, 32),
    hist_bins      = 32,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)
my_features9 = lect_features = create_feature_parameters \
(
    color_space    = 'LUV', # # RGB HSV LUV HLS YUV YCrCb
    spatial_size   = (32, 32),
    hist_bins      = 32,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)
my_features10 = lect_features = create_feature_parameters \
(
    color_space    = 'HLS', # # RGB HSV LUV HLS YUV YCrCb
    spatial_size   = (32, 32),
    hist_bins      = 32,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)

my_features11 = lect_features = create_feature_parameters \
(
    color_space    = 'YUV', # # RGB HSV LUV HLS YUV YCrCb
    spatial_size   = (32, 32),
    hist_bins      = 32,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)
my_features12 = lect_features = create_feature_parameters \
(
    color_space    = 'YCrCb', # # RGB HSV LUV HLS YUV YCrCb
    spatial_size   = (32, 32),
    hist_bins      = 32,
    orient         = 9, 
    pix_per_cell   = 8, 
    cell_per_block = 2, 
    hog_channel    = 'ALL' , #0, #'ALL',
    spatial_feat   = True,
    hist_feat      = True,
    hog_feat       = True
)
pk_list_32 = [my_features7, my_features8, my_features9, my_features10,my_features11, my_features12]

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
pk_dict_16 = {
    'hls'  : 'PK_16/model_HLS_16_MoreData.pk',
    'hsv'  : 'PK_16/model_HSV_16_MoreData.pk',
    'luv'  : 'PK_16/model_LUV_16_MoreData.pk',
    'rgb'  : 'PK_16/model_RGB_16_MoreData.pk',
    'yuv'  : 'PK_16/model_YUV_16_MoreData.pk',
    'ycrcb': 'PK_16/model_YCrCb_16_MoreData.pk'
}


pk_dict_16_jpeg = {
    'hls'  : 'PK_16/model_HLS_16_DataJpeg.pk',
    'hsv'  : 'PK_16/model_HSV_16_DataJpeg.pk',
    'luv'  : 'PK_16/model_LUV_16_DataJpeg.pk',
    'rgb'  : 'PK_16/model_RGB_16_DataJpeg.pk',
    'yuv'  : 'PK_16/model_YUV_16_DataJpeg.pk',
    'ycrcb': 'PK_16/model_YCrCb_16_DataJpeg.pk'
}

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------

pk_dict_32 = {
    'hls' : 'PK_32/model_HLS_32_MoreData.pk',
    'hsv' : 'PK_32/model_HSV_32_MoreData.pk',
    'luv' : 'PK_32/model_LUV_32_MoreData.pk',
    'rgb' : 'PK_32/model_RGB_32_MoreData.pk',
    'yuv' : 'PK_32/model_YUV_32_MoreData.pk',
    'ycrcb': 'PK_32/model_YCrCb_32_MoreData.pk'
}

pk_dict_32_jpeg = {
    'hls'  : 'PK_32/model_HLS_32_DataJpeg.pk',
    'hsv'  : 'PK_32/model_HSV_32_DataJpeg.pk',
    'luv'  : 'PK_32/model_LUV_32_DataJpeg.pk',
    'rgb'  : 'PK_32/model_RGB_32_DataJpeg.pk',
    'yuv'  : 'PK_32/model_YUV_32_DataJpeg.pk',
    'ycrcb': 'PK_32/model_YCrCb_32_DataJpeg.pk'
}
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def debug_find_cars (d12, image):
    draw_image = mpimg.imread(image)
    window_img = draw_boxes(draw_image, d12.box_list, color=(0, 255, 255), thick=8) 
    print (len(d12.box_list))
    print (d12.box_list)
    plt.imshow (window_img)
    

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def to_get_hog_features (orient, pix_per_cell, cell_per_block ) :
    test_images='TestImages/*.jpeg'
    cars, notcars, len_cars, len_notcars =  build_test_data (test_images)
    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(cars))
    # Read in the image
    image = mpimg.imread(cars[ind])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)      
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient, 
                            pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=False)
    plot_2_images (image, hog_image, 'Example Car Image', 'HOG Visualization')
        
#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def to_find_cars (image_file, model_file, ystart, ystop, scale):
    
    dist_pickle    = pickle.load( open(model_file, "rb" ) )
    svc            = dist_pickle["svc"]
    X_scaler       = dist_pickle["X_scaler"]
    color_space    = dist_pickle["color_space"]
    orient         = dist_pickle["orient"]
    pix_per_cell   = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size   = dist_pickle["spatial_size"]
    hist_bins      = dist_pickle["hist_bins"]
    hog_channel    = dist_pickle["hog_channel"]

    image = mpimg.imread(image_file)
    window_list= find_cars(image, \
                ystart, ystop, scale, \
                svc, X_scaler, \
                color_space, orient, pix_per_cell, cell_per_block,  spatial_size, hist_bins, hog_channel)
    out_img = draw_boxes(image, window_list , color=(0, 0, 255), thick=6)
    plot_2_images(image, out_img, "ORIGINAL", "Vehicle Detection")

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def to_multiple_scale_search (imageFile, window_scale_list3 ):
    img = mpimg.imread(imageFile)    
    i = 1
    pix_per_cell = 8
    imageList = []
    for (ystart, ystop, scale) in window_scale_list3:
        image = draw_multi_scale_windows(img, ystart, ystop, scale, pix_per_cell)
        imageList.append(image)
        i+=1
    plot_3_images(imageList[0], imageList[1], imageList[2], "(350, 600, 1.8)", "(300, 600, 2.0)", "(300, 700, 3.0)")

    