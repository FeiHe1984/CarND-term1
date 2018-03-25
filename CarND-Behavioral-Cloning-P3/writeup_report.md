#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a **VGG16 transfer learning model** in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/vgg16.jpg "Model Visualization"
[image2]: ./examples/center.png "Center Lane Driving"
[image3]: ./examples/recover0.jpg "Recovery Image"
[image4]: ./examples/recover1.jpg "Recovery Image"
[image5]: ./examples/recover2.jpg "Recovery Image"
[image6]: ./examples/resized.png "Resized Image"
[image7]: ./examples/flipped.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py**:  containing the script to create and train the model

* **drive.py(modified the original file to fit the resized image size)** : for driving the car in autonomous mode

* **drive-original.py(the origianl drive.py)**: when use the **Nvidia model**, run the original file and in the final submitted **VGG16 model**, this file is **useless**.

* **model.h5(159 MB)**: containing a trained **VGG16** transfer learning convolution neural network 

* **writeup_report.pdf**: summarizing the results

  ​

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```



####3. Submission code is usable and readable

* The model.py file contains the code for training and saving the **VGG16** transfer learning neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

* The model.py file used default parameters, it train myself collecting data and used the **VGG16** model  and 5 epochs for training.

  ​

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

* My model  used the **VGG16** as the bechmark or base transfer learning model. The original **VGG16** network has 16 trainable layers, contains convolution layers and fully connected layers, and in keras could import 'not top' and 'with top'. In our project, we used the 'not top' 3 fully-connected layers and kept the bottom 13 convolutional layers and attached myself fully-connected layers. 
* The model includes ELU layers to introduce nonlinearity (**Vgg16_transfer_model function()**: code from line 160), and the data is normalized in the model using a **normalize_data()** function (code from line 54). 

####2. Attempts to reduce overfitting in the model

* The model contains dropout layers in order to reduce overfitting ((in **Vgg16_transfer_model()** function from line147). 
* The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 193). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer:

```python
#Default parameters follow those provided in the original paper.
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.6)
```

There, I used the Default parameters provied in the original paper: [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8), and added a decay parameters to make initial learning rate decrease and the model will get more precise 
as the epochs increased.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving smoothly around curves and driving counter-clockwise one lap.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

- The overall strategy for deriving a model architecture was to use the **VGG16** 'not top' model as the bechmark, and attached 3 fully connected layes to regression for the steering angle.

- My steps to structure the **model.py** are as follows:

  1. Firstly, I followed the udacity lectures which used the **Nvidia** model as the bench mark and added a dropout layer(keep prob set 0.5) after the covolution layer(). After 20 epochs training, the vehicle drove just so so and could stay on the track. But it moved  swingeingly so I didn't satisfiy with the model. 
  2. Then, I decied to use the transfer learning model and at last chose the **VGG16** model because it performenced well on recognition and extraction of general features of images. 
  3. When I structured the **VGG16** model, initially added 3 fully connected layers after import the 'not top' VGG16 model and freezing all **VGG16** tainable layers. After serveral experiments, it worked not so well so I make top2 block convolotion layers of **VGG16** tranable, which names Feature-extraction (train only the top-level of the network, the rest of the network remains fixed), and the performance of the vehicle drove more precisely and smoothly, for the top2 block convolution get some features that have generalization capabilities on the ImageNet data set.
  4. After this, I found this model that had something in overfiiting, so I added 3 dropout layer between the last 3 fully connected layers and  tuned the keep prob parameters. After a couple of expteriments, the model only needed 5 epochs to make the validation loss and trainning loss low enough and the vehicle testing driving performed smoothly and stayblely.


####2. Final Model Architecture

* The final model architecture (model.py lines 147-171) consisted of a **VGG16** transfer leaning model neural network with the following layers and layer sizes:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_2 (InputLayer)             (None, 80, 80, 3)     0                                            
____________________________________________________________________________________________________
block1_conv1 (Convolution2D)     (None, 80, 80, 64)    1792        input_2[0][0]                    
____________________________________________________________________________________________________
block1_conv2 (Convolution2D)     (None, 80, 80, 64)    36928       block1_conv1[0][0]               
____________________________________________________________________________________________________
block1_pool (MaxPooling2D)       (None, 40, 40, 64)    0           block1_conv2[0][0]               
____________________________________________________________________________________________________
block2_conv1 (Convolution2D)     (None, 40, 40, 128)   73856       block1_pool[0][0]                
____________________________________________________________________________________________________
block2_conv2 (Convolution2D)     (None, 40, 40, 128)   147584      block2_conv1[0][0]               
____________________________________________________________________________________________________
block2_pool (MaxPooling2D)       (None, 20, 20, 128)   0           block2_conv2[0][0]               
____________________________________________________________________________________________________
block3_conv1 (Convolution2D)     (None, 20, 20, 256)   295168      block2_pool[0][0]                
____________________________________________________________________________________________________
block3_conv2 (Convolution2D)     (None, 20, 20, 256)   590080      block3_conv1[0][0]               
____________________________________________________________________________________________________
block3_conv3 (Convolution2D)     (None, 20, 20, 256)   590080      block3_conv2[0][0]               
____________________________________________________________________________________________________
block3_pool (MaxPooling2D)       (None, 10, 10, 256)   0           block3_conv3[0][0]               
____________________________________________________________________________________________________
block4_conv1 (Convolution2D)     (None, 10, 10, 512)   1180160     block3_pool[0][0]                
____________________________________________________________________________________________________
block4_conv2 (Convolution2D)     (None, 10, 10, 512)   2359808     block4_conv1[0][0]               
____________________________________________________________________________________________________
block4_conv3 (Convolution2D)     (None, 10, 10, 512)   2359808     block4_conv2[0][0]               
____________________________________________________________________________________________________
block4_pool (MaxPooling2D)       (None, 5, 5, 512)     0           block4_conv3[0][0]               
____________________________________________________________________________________________________
block5_conv1 (Convolution2D)     (None, 5, 5, 512)     2359808     block4_pool[0][0]                
____________________________________________________________________________________________________
block5_conv2 (Convolution2D)     (None, 5, 5, 512)     2359808     block5_conv1[0][0]               
____________________________________________________________________________________________________
block5_conv3 (Convolution2D)     (None, 5, 5, 512)     2359808     block5_conv2[0][0]               
____________________________________________________________________________________________________
block5_pool (MaxPooling2D)       (None, 2, 2, 512)     0           block5_conv3[0][0]               
____________________________________________________________________________________________________
globalaveragepooling2d_2 (Global (None, 512)           0           block5_pool[0][0]                
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 512)           262656      globalaveragepooling2d_2[0][0]   
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 512)           0           dense_5[0][0]                    
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 256)           131328      dropout_4[0][0]                  
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 256)           0           dense_6[0][0]                    
____________________________________________________________________________________________________
dense_7 (Dense)                  (None, 64)            16448       dropout_5[0][0]                  
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 64)            0           dense_7[0][0]                    
____________________________________________________________________________________________________
dense_8 (Dense)                  (None, 1)             65          dropout_6[0][0]                  
====================================================================================================
Total params: 15,125,185
Trainable params: 15,125,185
Non-trainable params: 0
____________________________________________________________________________________________________
```

- This mode's trainable layer is defined as follows (from top2 block layer to the last fully-connected layer ):

  ```
  block4_conv1 (Convolution2D)     (None, 10, 10, 512)   1180160     block3_pool[0][0]                
  ____________________________________________________________________________________________________
  block4_conv2 (Convolution2D)     (None, 10, 10, 512)   2359808     block4_conv1[0][0]               
  ____________________________________________________________________________________________________
  block4_conv3 (Convolution2D)     (None, 10, 10, 512)   2359808     block4_conv2[0][0]               
  ____________________________________________________________________________________________________
  block4_pool (MaxPooling2D)       (None, 5, 5, 512)     0           block4_conv3[0][0]               
  ____________________________________________________________________________________________________
  block5_conv1 (Convolution2D)     (None, 5, 5, 512)     2359808     block4_pool[0][0]                
  ____________________________________________________________________________________________________
  block5_conv2 (Convolution2D)     (None, 5, 5, 512)     2359808     block5_conv1[0][0]               
  ____________________________________________________________________________________________________
  block5_conv3 (Convolution2D)     (None, 5, 5, 512)     2359808     block5_conv2[0][0]               
  ____________________________________________________________________________________________________
  block5_pool (MaxPooling2D)       (None, 2, 2, 512)     0           block5_conv3[0][0]               
  ____________________________________________________________________________________________________
  globalaveragepooling2d_2 (Global (None, 512)           0           block5_pool[0][0]                
  ____________________________________________________________________________________________________
  dense_5 (Dense)                  (None, 512)           262656      globalaveragepooling2d_2[0][0]   
  ____________________________________________________________________________________________________
  dropout_4 (Dropout)              (None, 512)           0           dense_5[0][0]                    
  ____________________________________________________________________________________________________
  dense_6 (Dense)                  (None, 256)           131328      dropout_4[0][0]                  
  ____________________________________________________________________________________________________
  dropout_5 (Dropout)              (None, 256)           0           dense_6[0][0]                    
  ____________________________________________________________________________________________________
  dense_7 (Dense)                  (None, 64)            16448       dropout_5[0][0]                  
  ____________________________________________________________________________________________________
  dropout_6 (Dropout)              (None, 64)            0           dense_7[0][0]                    
  ____________________________________________________________________________________________________
  dense_8 (Dense)                  (None, 1)             65          dropout_6[0][0]                  
  ====================================================================================================
  ```

  ​

- Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


![alt text][image1]

​	The above image refer to [Image Classification](http://book.paddlepaddle.org/03.image_classification/). My model is different from the image is in 'Size 7': the 3 fully 		   connected layers is changed to:

​	fc 512-> elu -> dropout -> fc 256 -> elu -> dropout -> fc 64 -> elu -> dropout -> fc 1

####3. Creation of the Training Set & Training Process

* To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

* I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from side to center. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

* Then I driving counter-clockwise in order to get more data points and combat the bias.

* In prepocess input images data, I resized the image and flip the image to augment data set: 

  1. I resized image from (160, 320) to (80, 80), here is an image that has then been flipped:

  ![alt text][image2]
  ![alt text][image6]

  2. To augment the data sat, I also flipped images and angles thinking that this would  For example, here is an image that has then been flipped:

     ![alt text][image6]
     ![alt text][image7]

* After the collection process, I had **7682** number of data points. I then preprocessed this data by cv2.resize() function and flip the image. So in every data point(line), I read 3 images, and augmented by flip these images, so altogether, I had 7682* 6=  **46092**(X) images and **46092**(y) angels to form the train data and validation data.

  I finally randomly shuffled the data set and put 20% of the data into a validation set which had **36870** training set and **9222** validation set.

  I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5-10 as evidenced by the **training loss** and **validatiojn loss** figure is all low enough and the tesing vehicle driving smooth and stayble.