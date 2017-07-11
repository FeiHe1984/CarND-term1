# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

06.28.2017

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of serveral steps as follows:

* 1.Made a Grayscale to the image.
* 2.Define a kernel size and apply Gaussian smothing to the gray image.
* 3.Using Canny function to find the edges of the image.
* 4.Define a four sided polygon to mask which in order to process the region of interest.
* 5.Run Hough transform to find lane lines.
* 6.Add the finded images using red color to the initial image.

At the 5th step, the original draw_lines() function could only draw the line segments, in order to connect/average/extrapolate line segments to draw right and left single solid lines, i modified it to draw_lines_connect() function. And i will list the detail algorithm:

* 1.Firstly, extract left x ,y and right x,y separately and stroe in the lists.
* 2.Secondly, calculate the average num of x,y coordinates of left and right lines.
* 3.Thirdly, calculate the slope and intercept for left and right direction through those avg numbers.
* 4.Fouthly, find the endpoint to define the whole straight line which has the average slope and intercept.
* 5.Fifthly, draw the right and left line using the end points.


### 2. Identify potential shortcomings with your current pipeline

* 1.One potential shortcoming would be what would happen when a new direction road are processed by my pipeline. Because the  pipeline is a fixed way that can not generalize to the most  of analogous problems. 
* 2.Another shortcoming could be using many hard code in the functions.  
* 3.In  the optional chanlleng, my pipeline does not work well for the intersections with sharp angles.


### 3. Suggest possible improvements to your pipeline

* 1.A possible improvement would be to smooth the maked lines. The current line just can work but not perfectly fit the lane lines and it would be improved by tuning the parameters of the algorithm or using a filter to simplize the slope and interception.
* 2.Another potential improvement could be to refine the code more elegant and strong.
* 3.Using more powerful opencv tools to make my pipeline robust and automatic.