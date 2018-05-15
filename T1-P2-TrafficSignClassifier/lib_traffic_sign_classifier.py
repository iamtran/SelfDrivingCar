# lib_traffic_sing_classifier.py
# Load pickled data
import cv2
import pickle
import numpy as np
import random
import statistics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten


def load_training_data ():
    training_file   = "train.p"
    validation_file = "valid.p"
    testing_file    = "test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test   = test ['features'], test ['labels']

    assert(len(X_train) == len(y_train))
    assert(len(X_valid) == len(y_valid))
    assert(len(X_test)  == len(y_test))
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def summary_data_set (X_train, y_train, X_valid, y_valid, X_test, y_test):

    n_train = len(X_train)

    # TODO: Number of validation examples
    n_validation = len(X_valid)

    # TODO: Number of testing examples.
    n_test = len(X_test)

    # TODO: What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # TODO: How many unique classes/labels there are in the dataset.
    n_classes = len(np.bincount(y_train))

    print("Number of training examples   =", n_train)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples    =", n_test)
    print("Image data shape              =", image_shape)
    print("Number of classes             =", n_classes)
    
    return n_classes
    
def display_dataset_distributions(mydata, labels):
    
    fig, (ax0) = plt.subplots(figsize=(16, 8))
    final_data_bins = []    
    for dataset, label_desc in zip(mydata, labels):
        # Create a histogram of the classes
        data_bins = np.bincount(dataset)
        # Convert to percent
        data_bins = data_bins / len(dataset) * 100
        #final_data_bins.append(data_bins)
        
        ax0.hist(data_bins, bins=40, label=label_desc)
        ax0.set_title('data distribution')
        ax0.set_xlabel('% samples')
        ax0.set_ylabel('# classes')
    
    ax0.legend(loc=1)
    plt.show()
    
def preprocessing_data (X):
    return ((X - 128)/ 128)

def plot_image(image, nr, nc, i):
    plt.subplot(nr, nc, i)
    plt.imshow(image, cmap="gray")
    
def random_plot (data1, data2=None, num_images=1):
    nr = 2
    nc = num_images
    plt.figure(figsize=(nc,nr))             
    for i in range(num_images):
        index = random.randint(0, len(data1))
        image1 = data1[index].squeeze()
        image2 = data2[index].squeeze()
        plot_image(image1, nr, nc,  i+1)
        plot_image(image2, nr, nc,  5+(i+1))
        


def calculate_accuracy(X_data, y_data, x, y, accuracy_operation, BATCH_SIZE):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]

        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
    
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = n_classes = 43
    n_classes = 43
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes ), mean = mu, stddev = sigma))
    #
    
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


def conv_layer(x, iChannel, oChannel, k=5, s=1, mu=0, sigma=0.1):
    #print ("conv2d >> n_InCh={}, oChannel={}, k={}, s={}, mu={}, sigma={} ".format(iChannel, oChannel, k, s, mu, sigma))
    W = tf.Variable(tf.truncated_normal([k, k, iChannel, oChannel], mu, sigma))

    b = tf.Variable(tf.truncated_normal([oChannel]))
    #print ("conv2d >> strides[1,{},{},1]  padding=VALID".format(s,s))
    conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='VALID')
    #print (conv)
    ## x = tf.nn.bias_add(x, b)
    result = tf.nn.relu(conv + b)
    #print ("conv2d >>", result)
    return result 

def maxpool_layer(x, k=2, s=2):
    #print ("maxpool2d >> k={}, s={}".format(k, s))
    #print ("maxpool2d >> strides[1,{},{},1]  ksize=[1, {}, {}, 1]padding=VALID".format(s,s,k,k))
    result = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID')
    #print ("maxpool2d >> ", result)
    return result 

def fullyconnected_layer(x, iChannel, oChannel, activation=False, mu=0, sigma=0.1):
    #print ("fullyconnected >> iChannel{}, oChannel={}, activation={}, mu={}, sigma={}".format( iChannel, oChannel, activation, mu, sigma))
    W = tf.Variable(tf.truncated_normal([iChannel, oChannel], mu, sigma))

    b = tf.Variable(tf.truncated_normal([oChannel]))
    result = fc = tf.matmul(x, W) + b
    if activation: result =  tf.nn.relu(fc)
    #print ("fullyconnected >> ", result)
    return result

def Lenet2(x, n_channels = 3, n_classes = 43, mu = 0, sigma = 0.1, keep_probability=0.5 ): 
    
    keep_probability_tf = tf.constant(keep_probability)
    # 32x32x3 => 28x28x25
    conv1     = conv_layer(x,        iChannel=n_channels, oChannel=25, k=5, s=1, mu=mu, sigma=sigma)  
    # 28x28x25 => 14x14x25
    pooling1  = maxpool_layer(conv1, k=2, s=2)  
    # 14x14x25 => 10x10x40
    conv2     = conv_layer(pooling1, iChannel=25,         oChannel=40, k=5, s=1, mu=mu, sigma=sigma)  
    # 10x10x40 => 5x5x40
    pooling2  = maxpool_layer(conv2, k=2, s=2)                     
    # 5x5x40 => 3x3x60
    conv3     = conv_layer(pooling2, iChannel=40,         oChannel=60, k=3, s=1)   
    # 3x3x60 => 540
    flatP2    = flatten(conv3)
    #print ("flatP2", flatP2)
    # 540 => 120
    drop1 = fullconn1 = fullyconnected_layer(flatP2, 540, 120, True)     
    # 120
    #drop1     = tf.nn.dropout(fullconn1, keep_probability_tf)
    #print ("drop", drop1)
    # 120 => 43
    
    fullconn2 = fullyconnected_layer(drop1, 120, n_classes )         
    return fullconn2

def Lenet3(x, n_channels = 3, n_classes = 43, mu = 0, sigma = 0.1, keep_probability=0.5 ): 
    
    keep_probability_tf = tf.constant(keep_probability)
    # 32x32x3 => 28x28x25
    conv1     = conv_layer(x, iChannel=n_channels, oChannel=25, k=5, s=1, mu=mu, sigma=sigma)  
    # 28x28x25 => 14x14x25
    pooling1  = maxpool_layer(conv1, k=2, s=2)  
    # 14x14x25 => 10x10x40
    conv2     = conv_layer(pooling1, iChannel=25, oChannel=40, k=5, s=1, mu=mu, sigma=sigma)  
    # 10x10x40 => 5x5x40
    pooling2  = maxpool_layer(conv2, k=2, s=2)                     
    # 5x5x40 => 3x3x60
    conv3     = conv_layer(pooling2, iChannel=40, oChannel=60, k=3, s=1)   
    # 3x3x60 => 540
    flatP2    = flatten(conv3)
    #print ("flatP2", flatP2)
    # 540 => 120
    fullconn1 = fullyconnected_layer(flatP2, 3*3*60, 120, True)     
    # 120
    drop1     = tf.nn.dropout(fullconn1, keep_probability_tf)
    #print ("drop", drop1)
    # 120 => 43
    
    fullconn2 = fullyconnected_layer(drop1, 120, n_classes )         
    return fullconn2

def load_external_traffic_sign():
    ext_result = []

    file_list = ["p1_caution.jpg",  "p2_slippery.jpg", "p3_roadNarrowOnLeft.jpg",            
                 "p6_slipperyRoad.jpg",  
                 "p7_roadwork.jpg",  "p8_trafficSignal.jpg"]
    dim = (32,32)
    image_dir  = "./ext_pict_2/"
    # "p1_caution.jpg"
    i = 0
    image_file = image_dir + file_list[i]
    image = mpimg.imread(image_file)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    plot_image(image, 3, 3, i+1)
    ext_result.append([18, resized])
    #  "p2_slippery.jpg"
    i = 1
    image = mpimg.imread(image_dir + file_list[i])
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    plot_image(image, 3, 3, i+1)
    ext_result.append([23, resized])
    # "p3_roadNarrowOnLeft.jpg"  Keep right ?
    i = 2
    image = mpimg.imread(image_dir + file_list[i])
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    plot_image(image, 3, 3, i+1)
    ext_result.append([38, resized])

    # "p6_slipperyRoad.jpg"
    i = 3
    #image = mpimg.imread(image_dir + file_list[i])
    #resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    #plot_image(image, 3, 3, i+1)
    #ext_result.append([23, resized])
    # "p7_roadwork.jpg"
    i = 4
    image = mpimg.imread(image_dir + file_list[i])
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    plot_image(image, 3, 3, i+1)
    ext_result.append([25, resized])
    #  "p8_trafficSignal.jpg"
    i = 5
    image = mpimg.imread(image_dir + file_list[i])
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    plot_image(image, 3, 3, i+1)
    ext_result.append([26, resized])

    return ext_result

def distribution_valid_traffic_sign(y_valid):
    x=range(0,43)
    y = list( x)
    plt.hist(y_valid, bins=y)

    # Display the first 43 sign we see in the y_valid
def sample_valid_traffic_sign (X_valid, y_valid): 
    sign_list = []
    for i in range (0, 42):
        for j in range (0, len(y_valid)):
            if (i == y_valid[j]):
                sign_list.append(X_valid[j])
                break
                
    return sign_list
def display_valid_traffic_sign(sign_list):  
    dim = (32,32)
    plt.figure(figsize=(20, 20))
    for i in range (0, len(sign_list)):
        image = sign_list[i]
        #resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        plot_image(image, 10, 5, i+1)

def perform_prediction (trained_model_file, ext_result):
    tf.reset_default_graph()
    X_ext_result = np.array([col[1] for col in ext_result])
    y_ext_result = [col[0] for col in ext_result]
    #sess = tf.Session()
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
    logits = Lenet2(x)
    n_classes          = 43
    rate               = 0.001; EPOCHS             = 100; BATCH_SIZE         = 128;
    one_hot_y          = tf.one_hot(y, n_classes)

    cross_entropy      = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation     = tf.reduce_mean(cross_entropy)

    optimizer          = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, trained_model_file)
        predictions = sess.run(logits, feed_dict={x: X_ext_result})

    errors = 0
    for prediction,i in zip(predictions, range(len(predictions))):
        print("expected/predicted class: {}/{}".format(y_ext_result[i], np.argmax(prediction)))
        if np.argmax(prediction) != y_ext_result[i]:
            errors += 1
    print("correct:", len(predictions) - errors)
def load_external_traffic_sign2(ext_result, image_file, code ):
    dim = (32,32)
    image = mpimg.imread(image_file)
    #resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    resized = cv2.resize(image, dim)

    ext_result.append([code, resized])
    return ext_result

def build_external_traffic_sign (ext_image_list1):

    image_file='test_data2/70_km_hr_sign.jpg'  ; code = 4
    ext_image_list1 = load_external_traffic_sign2(ext_image_list1, image_file, code )

    image_file='test_data2/road_work1_sign.jpg' ; code = 25
    ext_image_list1 = load_external_traffic_sign2(ext_image_list1, image_file, code )

    image_file='test_data2/road_work2_sign.jpg' ; code = 25
    ext_image_list1 = load_external_traffic_sign2(ext_image_list1, image_file, code )
    len(ext_image_list1)
    image_file='test_data2/road_work1_sign_misc.jpg'   ; code = 25
    ext_image_list1 = load_external_traffic_sign2(ext_image_list1, image_file, code )
    image_file='test_data2/road_work1_sign_contrast.jpg'  ; code = 25
    ext_image_list1 = load_external_traffic_sign2(ext_image_list1, image_file, code )
    image_file='test_data2/road_work1_sign_mirror.jpg'  ; code = 25
    ext_image_list1 = load_external_traffic_sign2(ext_image_list1, image_file, code )
    image_file='test_data2/road_work1_sign_rotate.jpg' ; code = 25
    ext_image_list1 = load_external_traffic_sign2(ext_image_list1, image_file, code )
    
    image_file='test_data2/road_work1_sign_flares.jpg' ; code = 25
    ext_image_list1 = load_external_traffic_sign2(ext_image_list1, image_file, code )
    row = len (ext_image_list1) % 5
    for i in range (0, len (ext_image_list1)):
        
        image = ext_image_list1[i][1]
        plot_image(image,  row, 5, i+1)
    return ext_image_list1