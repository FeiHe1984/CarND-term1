##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_windows2.png
[image5]: ./examples/sliding_windows3.png
[image6]: ./examples/heat_map1.png
[image7]: ./examples/heat_map2.png
[image8]: ./examples/heat_map3.png
[image9]: ./examples/heat_map4.png
[image10]: ./examples/pipeline_test.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!  This project code are all in  "./CarND-Vehicle-Detection.ipynb" .  

###Part1: Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the **2-3** code cell of the IPython notebook located in "./CarND-Vehicle-Detection.ipynb" . 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then defined a `get_hog_features` function to extract HOG features which is in **3** code cell.

Here is an example using the `Gray` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

The code for this step is contained in the **4-5** code cell of the IPython notebook located in "./CarND-Vehicle-Detection.ipynb" . 

I tried various combinations of parameters and extract features from color way(spatial intensity or color channel intensity histogram features) and HOG way and the combinations of the two ways, which in the  `extract_features_color`,  `extract_features_hog`, `extract_features`, and in my pespective, the combinations of the two ways would perform better than other two. But after many experiment, this combinations way is time consuming and often extrat too many features and in the later accuracy of training classifier on test dataset is not more stable than only HOG ways. The only color way performed not well on training classifier and then at last I chose only the HOG way to extract features because it is enough to extract features to train the classifier efficiently and acurately.

The evaluation metrics of my settled final HOG parameters is performance of the SVM classifier produced using them, which include the time of training and accuracy of the test dataset. Trough the test on differnt `color space `, `orient`, `pix_per_cell`, `cell_per_block` and `hog_channel` parameters, I found following paramers performed best:

```python
# Feature extraction parameters
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
```

the **'RGB'** and **'Ycrcb'** colorspace also had good accuracy but the forecast time is logger than **'YUY'**, and in the later **'Sliding Window Search'** part has more false positives search than the **'YUY'**.

In this step, I also prepare the traing data by splitting the dataset to train data and test data and nomorlizing them by  `StandardScaler()`  method.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using **' LinearSVC() of sklearn.svm'**  by default parameters and using HOG features only. At last, the classifier took **1.32** seconds to train SVC and aurracy is **0.9848**(about **98.5%**).



###Part2: Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the **9-13** code cell of the IPython notebook located in "./CarND-Vehicle-Detection.ipynb" . 

firstly, I follwed the lecture **'Hog Sub-sampling Window Search'** to extract features and make predictions using `find_cars` method by fixted **scale 1.5**, y axis start 440 and end 660. The `find_cars` only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. The result images as follows:

![alt text][image3]

Secondly, I search in different scales , y start point and y end point for 2 rows region of windows in single image. `find_all_scales` function used the `find_cars` several times and aggregated all return rectangles.

For long distanse of **small** scale is **1.0 and 1.5 scales**, **medium **is  **2.0 scales** and **near large** is **3.0 scales** and the window overlap 75% in y axis(50% in x axis). The details in code cell **11** and the interpration  is in the title **Search in different scales**. The images are following(**12code cell**):

![alt text][image4]

Thridly, I plotted the region of the scales windows search, the result as follows(**13 code cell**):

![alt text][image5]

The final searching considered 186 window locations, which is proved enough to reliably detect vehicles robustly while maintaining a high speed of execution.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The code for this step is contained in the **14-18** code cell of the IPython notebook located in "./CarND-Vehicle-Detection.ipynb" . 

In the first step of this part, I found the sliding window searching had a few false positives dectection, and so I used Heat Map to tackle them. The steps of Heat Map are from the lecture of udacity:

* 1.Add heat map(**14 code cell**):

![alt text][image6]

* 2.Apply Threshold(**15 code cell**):

![alt text][image7]

* 3.Apply  SciPy Labels(**16 code cell**):

  ![alt text][image8]

* 4.Draw Bounding Boxes for Lables(**17 code cell**):

  ![alt text][image9]

  ​

Ultimately I put all previous steps to form a Pipline to process the frame image(**18 code cell**):.  Here are some example test images:

![alt text][image10]

---

### Part3: Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)

The code for this step is contained in the **22** code cell of the IPython notebook located in "./CarND-Vehicle-Detection.ipynb" . 

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the **19-21** code cell of the IPython notebook located in "./CarND-Vehicle-Detection.ipynb" . 

firstly, I ran my **pipeline of part 2** on the test video. The result is in **19** code cell. The rectangle is not stable and flashed in different scales.

Secondly, I refined the pipeline(**20 code cell**): 

* Adding a deque(maxlen=10) to store the previous frames' rectangles and using all of them to form the heat map, the treshold is set to `len(recent_frames) // 2 + 1`(treshold will be equal or more than half the number of rectangle sets contained in the history).
* Defined a `region_of_interest` method to mask the interested region (the left other side cars are better not to be dectcted).
* In every start of process the video, clear the deque: `recent_frames.clear()`

At last, using the refined pipeline to process the test video and project video(**21-22 code cell**).

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The proble I encountered is the `GridSearchCV` to find the best parameters to train the classifier, because it was time consuming and although the test and train accuracy is high, but then the predition in `find_cars` performed not well.
* The techniques of sliding window robustly and video processing are tricky and took a lot of time.
* The pipeline will be failed when the vehichels that are not in the trainning dataset come out because it can't identify the features.
* I believe the AI and Machine Learning will dectect  vehicles more robustly and efficiently.
* Finally , even though this part makes me tired, but I enjoy it and have a fun.