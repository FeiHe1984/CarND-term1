# Import packages
import os
import csv
import numpy as np
import sklearn
import argparse
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import cv2
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.models import Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Input, GlobalAveragePooling2D
from keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, ELU, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16


# Resized shape and input shape of the model 
resized_shape = (80, 80)
INPUT_SHAPE = (80, 80, 3)



def read_csv(args):
    """
    # Reading the csv file and store in the list
    """
    samples = []
    # Using the supplied data of udacity
    if args.selected_dataset == 'udacity':
        with open('./data/data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)
        samples = samples[1:]

    # Using my data from the simulator
    elif args.selected_dataset == 'my_data':
        with open('./data1/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)

    return samples

def normalize_data(image_data):
    """
    # Normalize the dataset
    """
    data = image_data / 255.0 - 0.5
    return data


def generator(args, samples, batch_size=32):
    """
    # The generator of the samples
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    # Selected the name path for each dataset
                    if args.selected_dataset == 'udacity':                   
                        name = './data/data/IMG/'+batch_sample[i].split('/')[-1]
                    elif args.selected_dataset == 'my_data': 
                        name = './data1/IMG/'+batch_sample[i].split('\\')[-1]
                    # Read the image and preprocess the image: resize, stree angle, augment, filp 

                    # Nvidia model: read the image only and when it run the 'model.h5', 
                    #it shoud be used the 'python drive-original.py model.h5'
                    if args.selected_model == 'Nvidia':
                        image = cv2.imread(name)           
                        images.append(image)   

                    # Vgg16 transfer learning model: read and resize the image.
                    # When it run the 'model.h5',
                    #it should be used the 'python drive.py model.h5'
                    elif args.selected_model == 'Vgg16':                   
                        image = cv2.imread(name)   
                        image = cv2.resize(image, resized_shape)              
                        images.append(image)

                    center_angle = float(batch_sample[3])
                    correction = 0.2
                    if i == 0:
                        measurment = center_angle # center
                    elif i == 1:
                        measurment = center_angle + correction #left
                    elif i == 2:
                        measurment = center_angle - correction # right
                    angles.append(measurment)
            # Augment and flip the image
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle*-1.0)

            # Form the X train and y train and shuffle the generator
            X_train = np.array(augmented_images)
            X_train = normalize_data(X_train)
            y_train = np.array(augmented_angles) 
            yield sklearn.utils.shuffle(X_train, y_train)


def Nvidia_model():
    """
    # Nvidia model:referenced https://arxiv.org/pdf/1604.07316v1.pdf
    # this model follwo the udacity lecture, and not modified the image size before traiing the model,
    # and so that when you run the simulator, you should run the 'drive-original.py' which don't change the 
    # tesing image size
    """
    print("Nvidia-------------")
    model = Sequential()
    model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


def Vgg16_transfer_model():
    """
    # Vgg16 model
    # this model's input shape is resized, so when you run the simulator, 
    # you should run the 'drive.py' which modified the testing image size
    """
    print("Vgg16---------------")
    input_shape = INPUT_SHAPE
    input_tensor = Input(shape=input_shape)

    base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='elu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='elu')(x)
    x = Dropout(0.1)(x)
    predictions = Dense(1, init='zero')(x)

    # the model we will train
    model = Model(input=base_model.input, output=predictions)

    return model


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Project')
    parser.add_argument('-d', help='input udacity or my_data', dest='selected_dataset', type=str, default='my_data')
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=5)
    parser.add_argument('-m', help='input Nvidia or Vgg16', dest='selected_model', type=str, default="Vgg16")
    args = parser.parse_args()

    print('-' * 40)
    print('Parameters')
    print('-' * 40)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 40)

    # Load dataset 
    samples = read_csv(args)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(args, train_samples, batch_size=32)
    validation_generator = generator(args, validation_samples, batch_size=32)

    # Selected model and train the model
    # Nvidia model
    if args.selected_model == 'Nvidia':
        model = Nvidia_model()
        # Default parameters follow those provided in the original paper.
        model.compile(loss='mse', optimizer='adam')
        checkpoint = ModelCheckpoint('model.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
        history_object = model.fit_generator(train_generator, samples_per_epoch= 
                        len(train_samples)*6, validation_data=validation_generator, 
                        nb_val_samples=len(validation_samples)*6, callbacks=[checkpoint], nb_epoch=args.nb_epoch)
        model.save('model.h5')

    # Vgg16 transfer learning model
    elif args.selected_model == 'Vgg16':
        model = Vgg16_transfer_model()

        # Choose top2 blocks to train
        for layer in model.layers[:11]:
            layer.trainable = False
        for layer in model.layers[11:]:
            layer.trainable = True   

        # Default parameters follow those provided in the original paper.
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.6)
        model.compile(optimizer=opt, loss='mse')
        checkpoint = ModelCheckpoint('model.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
        history_object = model.fit_generator(train_generator, samples_per_epoch= 
                    len(train_samples)*6, validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples)*6, callbacks=[checkpoint], nb_epoch=args.nb_epoch)

    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()