# Lane Detection

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Authors
Arpit Aggarwal
Shantam Bajpai


### Introduction to the Project
In this project, we used Homography and Histogram of Lane Pixels for Lane Detection. The algorithm is described in more detail in the "Report.pdf" file.


### Software Required
To run the .py files, use Python 3. Standard Python 3 libraries like OpenCV, Numpy, scipy and matplotlib are used.


### Instructions for running the code
To run the code for problem 1 and problem 2, follow the following commands:

```
cd Code
python problem1.py 'video_path(in .mp4 format)'
```
where, video_path is the path for input video. For example, running the python file on my local setup was:

```
cd Code/
python problem1.py /home/arpitdec5/Desktop/enpm673/projects/Project2/data/Night.mp4
```


```
cd Code/
python problem2_data1.py 'folder_path (containing images)'
```
where, folder_path is the path for the folder. For example, running the python file on my local setup was:

```
cd Code/
python problem2_data1.py /home/arpitdec5/Desktop/enpm673/projects/Project2/data/data_1/data
```


```
cd Code/
python problem2_data2.py 'video_path(in .mp4 format)'
```
where, folder_path is the path for the folder. For example, running the python file on my local setup was:

```
cd Code/
python problem2_data2.py /home/arpitdec5/Desktop/enpm673/projects/Project2/data/data_2/challenge_video.mp4
```


### Credits
The following links were helpful for this project:
1. https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
2. https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
3. https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
4. https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
