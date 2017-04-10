
# Udacity Self-Driving Car Engineer Nanodegree Program


## Vehicle Detection Project

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
    * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
    * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[im01]: ./output_images/car_visualize.png
[im02]: ./output_images/noncar_visualize.png
[im03]: ./output_images/hog_nonhog_image2.png
[im04]: ./output_images/different_config.png
[im05]: ./output_images/model_accuracy.png
[im06]: ./output_images/top_config.png
[im07]: ./output_images/top15_accuracy.png
[im08]: ./output_images/selected_config.png
[im09]: ./output_images/selected_config2.png
[im10]: ./output_images/config_61.png
[im11]: ./output_images/config_49.png
[im12]: ./output_images/config_13.png
[im13]: ./output_images/config_50.png
[im14]: ./output_images/slidingwindow1.png
[im15]: ./output_images/slidingwindow2.png
[im16]: ./output_images/slidingwindow3.png
[im17]: ./output_images/slidingwindow4.png
[im18]: ./output_images/slidingwindow_combine.png
[im19]: ./output_images/heatmap1.png
[im20]: ./output_images/heatmap_threshold.png
[im21]: ./output_images/applying_labels.png
[im22]: ./output_images/labelled_img.png


[//]: # (Image References)

[image1]: ./output_images/01_random_data_grid.png
[image2]: ./output_images/02_hog_visualization.png
[image3]: ./output_images/03_detections.png
[image4]: ./output_images/04_boxes_1.png
[image5]: ./output_images/05_boxes_2.png
[image6]: ./output_images/06_boxes_3.png
[image6a]: ./output_images/06a_boxes_4.png
[image7]: ./output_images/07_all_detections.png
[image8]: ./output_images/08_heatmap.png
[image9]: ./output_images/09_heatmap_threshold.png
[image10]: ./output_images/10_label_heatmap.png
[image11]: ./output_images/11_final_boxes.png
[image12]: ./output_images/12_all_test_detects.png
[video1]: ./test_video_out.mp4
[video2]: ./test_video_out_2.mp4
[video3]: ./project_video_out.mp4


### Writeup / README


#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

All of the code for the project is contained in the Jupyter notebook Vehicle Detection..Project 5.ipynb

---------

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I loaded all the images provided from `vehicle` and `non-vehicle` folders by Udacity. I loaded all the images from these folder and visualize a random sample of car and non-car imagesbegan by loading all of the vehicle and non-vehicle image paths from the provided dataset. The figure below shows a random sample of images from both classes of the dataset.
The code for this part is present in `Vehicle Detection..Project 5.ipynb` cell 3-5

![alt tag][im01]

![alt tag][im02]

For extracting HOG features from an image, I defined a function `get_hog_features` using the code provided in the Udacity lectures. The code is present under the section *`Functions`* and visulized in section *`Histogram of Oriented Gradients (HOG)`* in `Vehicle Detection..Project 5.ipynb` cell 2 and 6. 
The figure below shows a comparison of a car and a non car image chosen at random and there associated histogram of oriented gradients.
**Note:** As the the image is chosen as random, therefore the example image shown below may be different form what you see in the code.

![alt tag][im03]

#### 2. Explain how you settled on your final choice of HOG parameters.

The function `extract_features` in the section *"Function to extract Hog Features from a list of images"* as explained in the code, takes in a list of images path, reads the images and calculates `binned color image features`, `different color histogram features` and `Hog Features`. But currently we can focusing only on Hog Features therefore, other two features are kept "False". The code for calculating other two features are present in "Functions" section in cell 2.

The function consist of various parameters with different values, Therefore I created a loop to test all combinations and stored the new configuration and features in variable called **df_configuration**, **dict_carfeatures ** and **dict_noncarfeatures**.

A snapshot of different configurations are shown below:

![alt tag][im04]



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Using the above configuration and features, I train a linear support vector machine (SVM) classifier, other models can also be tried, but currently I am focussing on "Linear SVC".The output is stored in a dataframe 'df_stats'. 

I give importance to both the time taken for training and Accuracy, with higher importance to Accuracy. Therefore, Below is the snapshot of top 15 values of Acuracy with their Training times. The code for this question is written under question 3 in `Vehicle Detection..Project 5.ipynb` cell 10-13.

![alt tag][im06]

** I choose configuration 61, 49, 50 and 13 as the Best in Accuracy and Training Time.**
The configuration is :

![alt tag][im08]
![alt tag][im09]


-----------

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

In the section titled `Function that can extract features using hog sub-sampling and make predictions` utilizing the materials provided in the lesson, I wrote a function `find_car`, that can extract HOG features(spatial bin and color histogram can also be added) with a sliding window search, But rather than extracting HOG features on each window(which is very time consuming), the method extract hog features for the entire image(of the image selected, based on `ystart` and `ystop` values). The Linear SVM classifier is than used to make prediction on HOG features for each window and then returns whether than window consists of car (positive) or not(zero).

**NOTE:** The below images uses 4 different model as explained above.

                                             Configuration: 61
![alt tag][im10]

                                             Configuration: 49
![alt tag][im11]

                                             Configuration: 13
![alt tag][im12]

                                             Configuration: 50
![alt tag][im13]



I also utilizes several differnt values of window postion and scale, as car distance from the camera can be increase or decrease with respect to image scaling. We can observe various overlaps in X and Y directions can. The following images show the configurations of all search windows to be used in final implementation for small(1x), medium(2x) and large windows(3x). 
**NOTE:** For the below images I have used configuration 50.

Parameters:
 
| Ystart    | Ystop   | Scale    | 
|:---------:|:-------:|:--------:| 
| 400       |  464    |  1.0     |
| 416       |  480    |  1.0     |

![alt tag][im14]

Parameters:
 
| Ystart    | Ystop   | Scale    | 
|:---------:|:-------:|:--------:| 
| 400       |  496    |  1.5     |
| 432       |  528    |  1.5     |

![alt tag][im15]

Parameters:
 
| Ystart    | Ystop   | Scale    | 
|:---------:|:-------:|:--------:| 
| 400       |  528    |  2.0     |
| 432       |  560    |  2.0     |

![alt tag][im16]

Parameters:
 
| Ystart    | Ystop   | Scale    | 
|:---------:|:-------:|:--------:| 
| 400       |  596    |  3.0     |
| 464       |  660    |  3.0     |

![alt tag][im17]

Finally, I combine all the window scale and windows returned from `find_cars` function are aggreagated. 

In previous implementations ti was found to return too many false positives, and originally the window overlap was set to 50% in both X and Y directions, but an overlap of 75% in the Y direction (yet still 50% in the X direction) produced more redundant true positive detections. By choosing, a appropriate y range for windows to reduce false positives where the car cannot appear seems a more likely approach as compared to setting x range, as the cars are constantly moving.

The image below: shows the implementation and returns windows returned by the `find_cars` function drawn on to test image. We can observe, there are several prediction travelling in the same direction and few on the incoming cars in different side of the lane.

![alt tag][im18]


Along with true positives, we also get some false positives as well. While, as observed, false positives gets only 2-3 detections, therefore, a combined heatmap and then applying a threshold can used to eliminate the false positives. The `add_heat` function creates a an image with incremental pixel value(called as "heat") on a all black images, the size of the original image at the detection window locations. So, area overlapped with more detections will be brighter as compared to area with lower detections(typically false positives).

We can see an example of the result in the following heat map. This part of the code is present in a section titled `HeatMap` in `Vehicle Detection..Project 5.ipynb` cell ...

**Note:** the model used in the image below is configuraton 50
![alt tag][im19]

A threshold(in this case "3") is applied to the heatmap (in the section titled "Applying the Threshold" in `Vehicle Detection..Project 5.ipynb`), setting all pixels that don't exceed the threshold to zero.

![alt tag][im20]

The label() function from scipy.ndimage.measurements collects spatially contiguous areas of the heatmap. as shown in the image below. the code for this part is present in section tilted `"Applying Labels"`.

![alt tag][im21]

In the section `"Draw Bounding Boxes for Labels"`, the `labeled_boxes_img` functions finds the pixel location of the labels obtained above and draw a bounding box on the image. Below is an example:

![alt tag][im22]

#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?



Passing all the images in the test folder to a pipeline function `"process_images"` that takes in images, function and parameters like, pixels_per_cell, cell_per_block, orientation etc. I draw the boxes on each of the 6 test images.

                                   Linear SVM model with Parameter Configuration 50
                
![alt tag][im23]
                
                                   Linear SVM model with Parameter Configuration 13
                                                                      
![alt tag][im24]
                
                                   Linear SVM model with Parameter Configuration 49
                                                                                                         
![alt tag][im25]                                                                                                         
                                                                                                                                                          Linear SVM model with Parameter Configuration 61

![alt tag][im26]

The final implementations with Configuration 61 and Configuration 50 performs very well, identifying the near-field vehicles in each of the images. ***But for the purpose of the implementation I am going with config 50.***

**PipeLine Working:**
1. I begin the implementation using the `find_cars` function and try to optimize this function using different hog features like color space i.e. rather from naive 'RGB' I try 'HSV', 'YUV' and 'YCrCb'. Also, which helps increase the test accuracy.
2. I use all the 3 channels( "ALL"), it does increase the training time by 2-3 secs but compensates that by increase accuracy by 3-4%.
3. I went with default pixel_per_cell as 8. As the above two paramters, already increased the accuracy to 98%+, and to avoid the risk of overfitting. But pixel_per_cell = 16, is also something which can be worth trying.

** Other Implementation Techniques:**
1. Playing with window sizes and Overlapping factor is described and used in my implementation, but further efforts are worth putting in that area.
2. Lowering the threshold, is also one way to improve detection(but increases the risk of false positives), but higher threshold tends to lose the size of the object. Therefore, maintaining that balance is something worth working on.

## Video Implementation

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here is the [link to my Test Video](https://youtu.be/RX-G5Lj_dpw)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the above code, we can notice the flickering of the boxes, which takes place, as the code is finding the boxes are each frame in the video, Therefore, to improve on the previous iteration of the function "process_image50", I create a class `Detect_Vehicle` that stores the previous 16 frames and using prev_boxes parameter. Therefore, rather than performing heatmap->threshold->label step in each frame, the detected boxes of the past 16 frames are combines and added to the heatmap and threshold is set to 1+ len(det.prev_boxes)//2(i.e (one more than half the number of rectangle sets contained in the history), this value is found to perform better than a single scaler value.

This is the [link](https://youtu.be/Nmpji9c5RuA) to the new test video.

This is the [link](https://youtu.be/VNI5Krd_N8I) to the final project Video

#### For Fun I have mixed the video output for Advance Lane Detection Algorithm and run the new Vehicle Detection Algorithm on it.

This is the [link](https://youtu.be/IKP0MRtYHgs) to my Video showcasing Advance Lane Detection with Vehicle Detection.


----------

## Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

**Problem Faced:**
The problems I faced were mainly related to prediction accuracy.
1. Balancing the value of threshold is crucial, as higher value can lead to smaller size boxes while lower value leads to number of false positives.
2. The Accuracy is directly propotional to time of implementation, while keeping in mind that time is a crucial element while implementing the classifier on road. Scanning a large number of images can provide higher accuracy but would take longer
3. I calculated the Hog features first and then implementing sliding window search to find correct detections, this can be reversed and might increase the accuracy but the time of implementation was increase drastically.
4. By Integrating detections from previous frames, it helps in removing some misclassfication, but introduces another problem, ie. vehicle with change position drastically with respect to previous frames will not be detection(specially incoming traffic for left lane, cannot be detection).

The pipeline is most likely to fail in cases where vehicles (or their HOG features) don't resemble those in the training dataset, or produce false positives where other objects in the image consist same hog features as those in trainig dataset, lighting and environmental conditions might also play a role (e.g. car fences color). 
Along with incoming cars, distant cars are also an issue, as I have used a smaller window( range of ystart and ystop) 

Some of the other approaches would be:
1. To increase the training data significantly
2. Mix Detection from previous frames and also detection in each image to remove false positive.
3. More focus on accuracy and less of time, which would provide more intelligent vehicle detection technique.
4. Higher overlapping of the frames
5. Use other classification approaches like Convolutional Neural Nets.



```python

```
