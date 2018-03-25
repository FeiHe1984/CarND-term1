#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/grayscale.jpg "Grayscaling"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/No_entry.jpg "Traffic Sign 1"
[image5]: ./examples/Speed_limit_(70km_h).jpg "Traffic Sign 2"
[image6]: ./examples/stop.jpg "Traffic Sign 3"
[image7]: ./examples/Turn_right_ahead.jpg "Traffic Sign 4"
[image8]: ./examples/Yield.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and all my files of this project are in the zip file Traffic_Sign_Classifier.zip. 

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?

  The size of trainning set is 34799.

  ​

* The size of the validation set is ?

  The size of the validation set is 4410.

  ​

* The size of test set is ?

  The size of test set is 12630.

  ​

* The shape of a traffic sign image is ?

  The shape of a traffice sign image is (32, 32, 3).

  ​

* The number of unique classes/labels in the data set is ?

  The number of unique classes/labels in the data set is 43.

  ​

####2. Include an exploratory visualization of the dataset.

Please check the  **visualization of the dataset** in the Traffic_Sign_Classifier html which heading name is **Include an exploration visualization of the dataset**. I will not plot there because these plotting images are magnitude. You will see as fllows:

*  Plotting traffic sign images for each class using subplot.

* Plotting the count of each sign for training set, validation set and test set, which use the 'barh' chart.

* We can find the distribution of classes in the training set are similar with the test set.

  ​



###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* As a first step, I decided to convert the images to grayscale because I suppose that the color information is not the most important judging  influence factor to the classified project.

  Here is an example of a traffic sign image before and after grayscaling.

![alt text][image1]

* As a second step, I normalizedthe image data with Min-Max scaling to a range of [0.1, 0.9]  because I want to make them well conditioned with zero means and eaqual variance.

  ​

* As a last step, I decided to generate additional data using Gaussian noise image from duplicated training set images. Because  I found some classes' samples of taining set  is small amount and if adding more noise data to traing set, it can performance much better than before.

  To add more data to the the training data set, I used the following techniques:

  ​	(1).I wrote a fuction to add noise to each image. (you can find the detail in 3rd pard of **Pre-process the Data Set**)

  ```python
  # Add imge with noise
  noisy_img = images_data + noise_factor * np.random.randn(*images_data.shape)
  noisy_img = np.clip(noisy_img, 0.1, 0.9)
  ```

  ​	(2). I used a loop to add every class which less than 1500 traing images data to go to 1500 samples with random noise samples. (you can find the detail in 3rd pard of **Pre-process the Data Set**)

  Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original trianing data set and the augmented data set is the following:

 -  original trianing data's shape: (34799, 32, 32, 3)

 -   augmented training data set's shape: (67380, 32, 32, 1), augmented 32581 noise samples.

 -  The test data set and validation data set are grayscale and nomalized not add noise images samples.

    ​


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |         32x32x1  grayscale image         |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x108 |
|      RELU       |                                          |
|     Dropout     |               Use kee_porb               |
|    Max pool     | Size 2x2, strides 2x2, valid padding, outputs 14x14x108 |
| Convolution 5x5 | Stride 1, valid padding.  Outputs 10x10x200 |
|      RELU       |                                          |
|     Dropout     |               Use kee_porb               |
|    Max pool     | Size 2x2, strides 2x2, valid padding, outputs 5x5x200 |
|     Flatten     |      Input = 5x5x200. Output = 5000      |
| Fully connected |        Input = 5000. Output = 400        |
|      RELU       |                                          |
|     Dropout     |               Use kee_porb               |
| Fully connected | Input = 400. Output = 43 (last classied labels) |



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer and Loss with following hyperparameters:

```
EPOCHS = 15
BATCH_SIZE = 128
rate = 0.001
```

when traing **keep_prob** is set **0.5**.

when  validation and testing  **keep_prob** is set **1.0**.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of final is 1.000, mostly 0.999.

* validation set accuracy of final is 0.959, highest 0.973

* test set accuracy of 0.958.

  ​

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

  I firtly began to do the lab with the  **LeNet-5** architecture because it proved very effective at classifying  handwritten digits in the lecture of udacity.

* What were some problems with the initial architecture?

  I exprimented many times then I found the validation accuracy never greater than 0.96 and unstable and the training epochs too many to me. In my opioion, In Convolution layers the depths of feature maps of  **LeNet-5** is small amount because there are 43 classes signs but in the classifying  handwritten digits only 10 classes.

* How was the architecture adjusted and why was it adjusted? Which parameters were tuned? How were they adjusted and why? What are some of the important design choices and why were they chosen? 

  -I ajusted the epochs to 30 or 64 or 128, but I did not see more imporovement in validation accuracy which were not greater than 0.96 and loss, though the training accuracy is high.

  I also changed the keep_prob from 0.2 to 0.7, but the 0.5 is the optimal to the architecture.

  A convolution layer work well with this problem because it can comprehend and extract the features of images well by itself.

  To overcome overfitting, I add dropout layer after fully connected layer 1 and layer 2. This made the validation accuracy boost a little.

  ​

If a well known architecture was chosen:
* What architecture was chosen?

  Then, I referenced to the[Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), I decided to modify the **LeNet-5** architecture.

* Why did you believe it would be relevant to the traffic sign application?

  This model don't change more compared with the  **LeNet-5**, it just add more depthsof feature maps in each Convolution layer, and only use one Fully connected layer. I believed it can performance better because after adding more noise training images the model need to more strong capacities to identify the different features of the traffic signs.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

  Finally, I found the architecture used only **10-20** epochs the loss of the model didn't change a lot, and the validation accuracy went to **0.96-0.97**. The training accuracy is **1.000**, as it were overfiiting,  that I added dropout  in  Convolution layer and Fully connected layer to combat the overfitting. And after many experiments by tuned parameters, my test data accuracy went to **0.958**.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

 Challenges:  a lot noise information

![alt text][image4] 



 Challenges: easily mixted with 30,50,60

![alt text][image5] 



 Challenges: No changllenge

![alt text][image6] 



 Challenges: No changllenge

![alt text][image7] 



Challenges: No changllenge

![alt text][image8]

* The first image might be difficult to classify because there are a lot noise information to disturb the classification.



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|        Image         |      Prediction      |
| :------------------: | :------------------: |
|       No entry       |       No entry       |
|   Turn right ahead   |   Turn right ahead   |
| Speed limit (70km/h) | Speed limit (60km/h) |
|        Yield         |        Yield         |
|         Stop         |         Stop         |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.958 (95.8%).

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 28th cell of the Ipython notebook.

For the 1st image(No_entry.jpg), the model is definitely sure that this is a No entry sign (probability of 100%), and the image does contain a No entry sign. The top five soft max probabilities were

| Probability |          Prediction          |
| :---------: | :--------------------------: |
|    1.00     |           No entry           |
|    0.00     |       Turn right ahead       |
|    0.00     | Dangerous curve to the right |
|    0.00     |     Go straight or left      |
|    0.00     |          No passing          |


For the 2nd image(Turn_right_ahead.jpg), the model is definitely sure that this is a Turn right ahead(probability of 100%), and the image does contain a Turn right ahead sign. The top five soft max probabilities were

| Probability |              Prediction               |
| :---------: | :-----------------------------------: |
|    1.00     |           Turn right ahead            |
|    0.00     |              Ahead only               |
|    0.00     | Right-of-way at the next intersection |
|    0.00     |         Roundabout mandatory          |
|    0.00     |            Turn left ahead            |

For the 3rd image(Speed_limit_(70km_h).jpg), the model identifies that uncertainly this is a Speed limit (60km/h) (probability of 55.23%), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were

| Probability |      Prediction       |
| :---------: | :-------------------: |
|    .5523    | Speed limit (60km/h)  |
|    .2025    |      Bumpy road       |
|    .1030    |       Road work       |
|    .0962    |   Bicycles crossing   |
|    .0296    | Wild animals crossing |

For the 4th image(Yield.jpg), the model identifies that this is definitely sure a Yield (probability of 100%), and the image does contain Yield sign. The top five soft max probabilities were

| Probability |             Prediction              |
| :---------: | :---------------------------------: |
|    1.00     |                Yield                |
|    0.00     |            Priority road            |
|    0.00     |        Go straight or right         |
|    0.00     |           Turn left ahead           |
|    0.00     | End of all speed and passing limits |

For the 5th image(stop.jpg), the model identifies that this is definitely sure a Stop(probability of 96.76%), and the image does contain Stopsign. The top five soft max probabilities were

| Probability |      Prediction      |
| :---------: | :------------------: |
|   0.9676    |         Stop         |
|   0.0161    | Speed limit (30km/h) |
|   0.0045    | Speed limit (20km/h) |
|   0.0036    | Speed limit (60km/h) |
|   0.0020    |        Yield         |

And there is also pollting these softmax predictions using barh chart.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I finished this part using the original architecture **LeNet-5** because the output feature maps are only have 6 in first Convolution layer and 16 in second Convolution layer. 

The characteristics of the neural network use to make classifications are in the previous Convolution layer the feature maps only can understand simple form concept, such as line, simple form, arc, etc. , and  in the later Convolution layer they can hold the complex idea so that the output feature maps of images are black and white grids.