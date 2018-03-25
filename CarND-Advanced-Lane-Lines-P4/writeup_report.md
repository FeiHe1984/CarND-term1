## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/test4_undistort.png "Road Transformed"
[image3]: ./examples/binary_combo_example.png "Binary Example"
[image4]: ./examples/warped_binary.png "Warp Example"
[image5]: ./examples/color_fit_lines.png "Fit Visual"
[image6]: ./examples/color_fit_nextframe.png "Fit Next Frame"
[image7]: ./examples/plot_back_down.png "Plot Back Down"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Part 1: Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the **2-4** code cell of the IPython notebook located in "./P4-CarND-Advanced-Lane-Lines.ipynb" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]



### Part2: Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for this step is contained in the **5** code cell of the IPython notebook located in "./P4-CarND-Advanced-Lane-Lines.ipynb" .  

In part 1, I save the camera calibration result for later use: `./calibration_pickle.p` and then in part 2, I opened the file and undistored the `test4.image`. The result as follows: 
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in the **6-7** code cell of the IPython notebook located in "./P4-CarND-Advanced-Lane-Lines.ipynb" .  

* Fistly, I used a combination of color and gradient thresholds to generate a binary image(**6** code cell), but the result with which I didn't satisfiy.  
* Then, I thresholded white and yellow color pixels from HLS channels(**7** code cell). It just mask the white and yellow color. The result is as follows: 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained in the **8-9** code cell of the IPython notebook located in "./P4-CarND-Advanced-Lane-Lines.ipynb" .  

The code for my perspective transform includes a function called `warper_binary()`, which appears in 9th code cell.  The `warper_binary()` function takes as inputs an image (`undistored_img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
   offset = 200
  	src = np.float32([
    [  588,   446 ],
    [  691,   446 ],
    [ 1126,   673 ],
    [  153 ,   673 ]])
    
	dst = np.float32([
    [offset, 0], 
    [img_size[0] - offset, 0], 
    [img_size[0] - offset, img_size[1]], 
    [offset, img_size[1]]])
```

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 588, 446  |   200, 0    |
| 691, 446  |   1080, 0   |
| 1126, 673 |  1080, 720  |
| 1126, 673 |  200, 720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is contained in the **10-11** code cell of the IPython notebook located in "./P4-CarND-Advanced-Lane-Lines.ipynb" .  

* First, I followed the udacity lecture of this part to use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.
* Secondly, In the next frame of video I don't need to do a blind search again, but instead I can just search in a margin around the previous line position.
* These two images shows as follows:

![alt text][image5]

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in the **12-13** code cell of the IPython notebook located in "./P4-CarND-Advanced-Lane-Lines.ipynb" .  

* Calculated the radius of curvature(code cell **12**):

  ```python
      # Calculate the new radii of curvature    
      left_y1 = (2*left_fit[0]*y_eval + left_fit[1])*xm_per_pix/ym_per_pix
      left_y2 = 2*left_fit[0]*xm_per_pix/(ym_per_pix*ym_per_pix)
      left_curverad = ((1 + left_y1*left_y1)**(1.5))/np.absolute(left_y2)
      
      right_y1 = (2*right_fit[0]*y_eval + right_fit[1])*xm_per_pix/ym_per_pix
      right_y2 = 2*right_fit[0]*xm_per_pix/(ym_per_pix*ym_per_pix)
      right_curverad = ((1 + right_y1*right_y1)**(1.5))/np.absolute(right_y2)
      
      aver_curverad = (left_curverad + right_curverad) / 2
  ```

* Caclulated the position of the vehicle with respect to center(code cell **13**):

  ```python
  	# Caclulated the position of the vehicle with respect to center
  	x_left_pix = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
      x_right_pix = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
      position_from_center = ((x_left_pix + x_right_pix)/2 - x_middle) * xm_per_pix
  ```



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is contained in the **13** code cell of the IPython notebook located in "./P4-CarND-Advanced-Lane-Lines.ipynb" . Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* This code perfomed not well on challenge as my function of warped binary can't adapt to various enviroment such as the variation of lightness, shadows or illumination etc.
* This project is a little complicated and tedious especially tuning parameters and make all steps to pipeline video.
* Maybe the machine leaning and AI will make the lane dectection more robustly and automatically.
* Finally , even though this part makes me tired, but I enjoy it and have a fun.