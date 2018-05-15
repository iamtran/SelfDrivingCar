# BehavioralCloning_NN_Model.py

from myLib.BehavioralCloning_ImportFile import *

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def display_history (history_object):
    ### print the keys contained in the history object
    print(history_object.history.keys()) 
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


#------------------------------------------------------------------------
#
#------------------------------------------------------------------------

def lenet_model ():
    print ("Model Name : lenet_model")
    model = Sequential()
    # Need to crop and nomalized data
    #model.add(Cropping2D(cropping=((50,20),(0,0))))
    #model.add(Lambda(lambda x:(x/255.0 )- 0.5, input_shape = (160,320,3)))
    
    model.add(Lambda(lambda x:(x/255.0 )- 0.5, input_shape = (160,320,3)))
    model.add(Convolution2D(6,5,5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    print(model.summary())
    return model
#------------------------------------------------------------------------
# Base on LeNet.With slight modification
#------------------------------------------------------------------------
def lenet_model2(input_shape=(160, 320, 3), dropout=.35 ):
    print ("Model Name : lenet_model2")
    model = Sequential()
    # cropping before normalization to save time. BUG WILL NOT WORK
    # ValueError: The first layer in a Sequential model must get an `input_shape` or 
    #`batch_input_shape` argument.
    #model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # normalization
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(6, 5, 5, activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, 5, 5, activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    print(model.summary())
    return model


#------------------------------------------------------------------------
#
#------------------------------------------------------------------------
def lenet_model3 ():
    print ("Model Name : lenet_model3")
    model = Sequential()
    # Need to crop and nomalized data
    #model.add(Cropping2D(cropping=((50,20),(0,0))))
    #model.add(Lambda(lambda x:(x/255.0 )- 0.5, input_shape = (160,320,3)))
    
    model.add(Lambda(lambda x:(x/255.0 )- 0.5, input_shape = (160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(6,5,5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    print(model.summary())
    return model

#------------------------------------------------------------------------
# Nvida model for driving cars
#------------------------------------------------------------------------

def nvidia_model():
    print ("Model Name : nvidia_model")
    model = Sequential()
    #pixel_normalized = pixel / 255
    #pixel_mean_centered = pixel_normalized - 0.5

    model.add(Lambda(lambda x:(x/255.0 )- 0.5, input_shape = (160,320,3)))

    model.add(Cropping2D(cropping=((50,20),(0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(10))
    model.add(Dense(1))
    print(model.summary())
    return model
#------------------------------------------------------------------------
# Read Driving data + Load Old Model + Train Model + save New model file
#------------------------------------------------------------------------
def retrain_model (data_dir, oldModelUsed, newModelfilesaved, epoch=5):
    images, measurements =  load_data2 (data_dir)
    
    augmented_images, augmented_measurements = augmentation_data(images, measurements)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    #model = modelUsed()
    model     = load_model(oldModelUsed)
    #model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epoch, verbose=1)
    model.save(newModelfilesaved)
    return history_object

#------------------------------------------------------------------------
#
#------------------------------------------------------------------------

def create_train_model_need_data (images, measurements, modelUsed, modelfilesaved, epoch=5):
    X_train = np.array(images)
    y_train = np.array(measurements)
    model = modelUsed()
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epoch, verbose=1)
    model.save(modelfilesaved)
    return history_object

def create_data_set (data_dir):
    images, measurements =  load_data2 (data_dir)
    #images, measurements =  load_data2 (data_dir)
    augmented_images, augmented_measurements = augmentation_data(images, measurements)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements) 
    
    return X_train, y_train
#------------------------------------------------------------------------
# Read Driving data + Create Model + Train Model + save model file
#------------------------------------------------------------------------
from keras.callbacks import ModelCheckpoint
def create_train_model (X_train, y_train, modelUsed, modelfilesaved, epoch=5):   
    model = modelUsed()
    model.compile(loss='mse', optimizer='adam')
    #history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=epoch, verbose=1)
    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)    
    history_object = model.fit(X_train, y_train, 
                               validation_split=0.2, shuffle=True, 
                               nb_epoch=epoch, verbose=1, callbacks=[checkpointer])
    model.save(modelfilesaved)
    return history_object

def model_run (data_dir, modelName, modelfile, epoch):
    if  (modelName=="lenet1"):
        X_train, y_train = create_data_set (data_dir)
        hist = create_train_model (X_train, y_train, lambda: lenet_model(),   modelfile, epoch)
    elif (modelName=="lenet2"):
        X_train, y_train = create_data_set (data_dir)
        hist = create_train_model (X_train, y_train, lambda: lenet_model2(), modelfile, epoch)
    elif (modelName=="lenet3"):
        X_train, y_train = create_data_set (data_dir)
        hist = create_train_model (X_train, y_train, lambda: lenet_model3(),  modelfile, epoch)
    elif (modelName=="nvidia"):  
        X_train, y_train = create_data_set (data_dir)
        hist = create_train_model (X_train, y_train, lambda: nvidia_model(), modelfile, epoch)
    elif (modelName=="all"):
        X_train, y_train = create_data_set (data_dir)
        hist = create_train_model (X_train, y_train, lambda: lenet_model (), modelfile, epoch)
        hist = create_train_model (X_train, y_train, lambda: lenet_model2(), modelfile, epoch)
        hist = create_train_model (X_train, y_train, lambda: lenet_model3(), modelfile, epoch)
        hist = create_train_model (X_train, y_train, lambda: nvidia_model(), modelfile, epoch)
    elif (modelName=="glenet1"): 
        hist = create_train_model_generator (data_dir, lambda: lenet_model() , modelfile, epoch)
    elif (modelName=="glenet2"):  
        hist = create_train_model_generator (data_dir, lambda: lenet_model2(), modelfile, epoch)
    elif (modelName=="glenet3"): 
        hist = create_train_model_generator (data_dir, lambda: lenet_model3(), modelfile, epoch)
    elif (modelName=="gnvidia"): 
        hist = create_train_model_generator (data_dir, lambda: nvidia_model(), modelfile, epoch)
    else :
        hist = create_train_model (data_dir, lambda: nvidia_model(), modelfile, epoch)                                    
    return hist

#------------------------------------------------------------------------
# Main part of generator to handle large amount of data
#------------------------------------------------------------------------

from sklearn.utils import shuffle
def generator(samples, batch_size=32, data_dir="/tmp"):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                row = batch_sample
                steering_center = float(row[3])
                correction     = 0.2 # this is a parameter to tune
                steering_left  = steering_center + correction
                steering_right = steering_center - correction
                img_center = process_image(get_file_name(data_dir, row,0))
                img_left   = process_image(get_file_name(data_dir, row,1))
                img_right  = process_image(get_file_name(data_dir, row,2))                

                images.append(img_center)
                images.append(img_left)
                images.append(img_right)
                angles.append(steering_center)
                angles.append(steering_left)
                angles.append(steering_right)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #yield tuple(sklearn.utils.shuffle(X_train, y_train))
            yield (sklearn.utils.shuffle(X_train, y_train))


#------------------------------------------------------------------------
# Use Generator Read Driving data + Create Model + Train Model + save model file
#------------------------------------------------------------------------
          
def create_train_model_generator (data_dir, modelUsed, modelfilesaved, epoch):
    samples = get_data (data_dir) # print(len(samples))
    print (len(samples))
    # compile and train the model using the generator function

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator      = generator(train_samples, batch_size=32, data_dir=data_dir)
    validation_generator = generator(validation_samples, batch_size=32, data_dir=data_dir)
    #print (next (train_generator) ) #print (next (validation_generator) )
    
    model = modelUsed()
    model.compile(loss='mse', optimizer='adam')
    
    hist = model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=epoch)
    model.save(modelfilesaved)
    return hist
