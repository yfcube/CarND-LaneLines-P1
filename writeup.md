# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)
[image1]: ./test_images_output/gray.png "Grayscale"
[image2]: ./test_images_output/gaussian_blur.png "Blur"
[image3]: ./test_images_output/edges.png "Edges"
[image4]: ./test_images_output/region_selected.png "Region"
[image5]: ./test_images_output/hough_ave.png "Region"
---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps:  
1. Images are converted to grayscale as follows.
![alt text][image1]
2. Grascale imagese are put into a gaussian filter to be smoothened.
3. Pixels belong to edges are selected using Canny Edge Detector provided by opencv. Result is shown as follows.
![alt text][image3]
4. A manually tuned region mask is apply to edge pixel images. After this, only pixels in region of interest remain. Result is shown as follows.
![alt text][image4]
5. Using Hough line transform algorithm, a set of lines consisted of edge pixels is botained. Inside Hough line transform function, I implemented a draw_lines() function to calculate two average lines representing left and right lane borders. Basic idea of how i modified the draw_lines() is decribed as follows: First, the set of lines obtained after Hough transform is classified into two groups according to their slope. Then, in each group, caculate average slope and intercept and find the y_min value. y_max is simply assigned to the height of image. The average line function can be obtained by average slope and intersect. Then x_min and x_max is obtained by this function using y_min and y_max. The set of lines and average lines are shown in the following image:
![alt text][image5]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the car is turning at a corner. Because in this pipeline only straight lines are extracted and the parameters using in Hough transform are fixed. So when the car is turning where boarder of lane is a curve, this pipeline is very likely to fail.

Another shortcoming could be the region seletion. Now the shape of polygon of region of interest is manually tunned and the rejection of other objects relied hugely on the selection of this region. When the width of the lane changed(suppose you drive to another county) or the installation angle of the camera changed, this pipeline is bound to fail.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to detect the points on lane boarder lines and represent the line as Bézier curve. 

