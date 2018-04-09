# **Behavioral Cloning** 

## Project Writeup  

### Yulong Li

---

## Goals

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report  

---
## Rubric Points
 
This writeup will include all the [Rubric Points](https://review.udacity.com/#!/rubrics/432/view).  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* project_writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and the original drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I started with the NVIDIA CNN architecture:  

![nvidia_cnn](https://github.com/yulongl/p3_BehavioralCloning/blob/master/pic/nvidia_cnn.png)  
  
From https://devblogs.nvidia.com/deep-learning-self-driving-cars:  

_"The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture, and to be accelerated via GPU processing._

_"The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers._

_"We follow the five convolutional layers with three fully connected layers, leading to a final output control value which is the inverse-turning-radius. The fully connected layers are designed to function as a controller for steering, but we noted that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor, and which serve as controller."_  

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 70 and 72). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an RMSprop optimizer. I made the learning rate adjustable but I didn't change it - still using the default 0.001 learning rate.  

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used only used the center lane driving and didn't use the images from left and right cameras. Because I think the driving behavior will not be natural and smooth by adding an fixed offset to the steering angle. Under different conditions, there might different angle offsets.  

Instead, I recorded more data, including teaching the vehicle to drive back to the center.  

For details about how I created the training data, see the next section. 

---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to modify and implement the NVIDIA CNN architecture mentioned in **"An appropriate model architecture has been employed"** above.   

Two more fully connected layers were added because the size of the flatten layer was way more than the NVIDIA CNN.

Image and steering angle data were split into a training and validation set in a ratio of 8:2. I found that the first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added two dropout layers with a dropout rate of 0.5, which significantly improved the performance.  

When I run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I generated more training data which taught the model how to drive back to center.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The model starts with three 5x5 convolutional layers with a stride size of (2, 2) and VALID padding, following by two 3x3 convolutional layers with a stride size of (1, 1) and also VALID padding. RELU has been applied to all the convolutional layers. After a flatten layer, there are five fully connected layers. Two dropout layers have been applied among them to reduce overfitting. Below is the model visualization.  

![model](https://github.com/yulongl/p3_BehavioralCloning/blob/master/pic/model.png)  

#### 3. Creation of the Training Set & Training Process

Even though the tutorial suggested to drive in center, I still wanted the car to be driven like a professional racer, who always picked the best and the most efficient routes. Below is an example:

![center_2018_04_03_20_48_50_878.jpg](https://github.com/yulongl/p3_BehavioralCloning/blob/master/pic/center_2018_04_03_20_48_50_878.jpg)

I then recorded the vehicle driven on center lane on track one:

![center_2018_04_03_21_16_48_321.jpg](https://github.com/yulongl/p3_BehavioralCloning/blob/master/pic/center_2018_04_03_21_16_48_321.jpg)  

![center_2018_04_03_21_17_10_787.jpg](https://github.com/yulongl/p3_BehavioralCloning/blob/master/pic/center_2018_04_03_21_17_10_787.jpg)  


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to center:

![center_2018_04_07_22_45_51_208.jpg](https://github.com/yulongl/p3_BehavioralCloning/blob/master/pic/center_2018_04_07_22_45_51_208.jpg)  

![center_2018_04_07_22_45_51_986.jpg](https://github.com/yulongl/p3_BehavioralCloning/blob/master/pic/center_2018_04_07_22_45_51_986.jpg)  


Then I repeated this process on track two in order to get more data points:  

![center_2018_04_03_21_27_27_846.jpg](https://github.com/yulongl/p3_BehavioralCloning/blob/master/pic/center_2018_04_03_21_27_27_846.jpg)  

![center_2018_04_03_23_26_32_307.jpg](https://github.com/yulongl/p3_BehavioralCloning/blob/master/pic/center_2018_04_03_23_26_32_307.jpg)  


After the collection process, I had **16468** number of data points. I then preprocessed this data by normalization.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

A recorded video of driving in autonomous mode on track one can be found in links below. Because the video size exceeded the GitHub limit, so I split it into two parts.  
https://github.com/yulongl/p3_BehavioralCloning/blob/master/run2_part1.mp4  
https://github.com/yulongl/p3_BehavioralCloning/blob/master/run2_part2.mp4  

