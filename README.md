Hand pose detection using camera image
======================================

* Done using OpenCV in Python 3
* Project is an attempt to detect same gestures as [https://support.getmyo.com/hc/en-us/articles/202647853](Myo EMG tracking device) detects.

Based on [lzane/Fingers-Detection-using-OpenCV-and-Python](https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python):

* changed code structure
* upgraded to use OpenCV 3.4.0
* adjusted parameters for tested device
* removed all trackbars, etc.
* added logic need to detect hand gestures

Developed on: Mac OS Sierra, Python: 3.6.1 (with NumPy: 1.14.0), OpenCV: 3.4.0


Processing flow
---------------

### Capturing image

Video is captured directly from camera connected to computer.
Images are processed in real time, but algorithm causes limits processing to about 10 FPS on tested device.

### Removing background

Background is removed from processing using [Gaussian Mixture-based Background/Foreground Segmentation Algorithm](http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf).
Method is based on finding pixels which colors are static between video frames.

More information: <https://docs.opencv.org/3.2.0/d1/dc5/tutorial_background_subtraction.html>

### Extracting shapes

Shapes on the image is smoothed using gaussian blur and then extracted using threshold method.

More information: [blur](https://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html) and [threshold](https://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/threshold/threshold.html).

### Finding shape of hand and locating fingertips

We assume that hand will be the biggest shape on the frame. We find the convex hull of the shape and then convexity defects.

### Detecting hand pose

Hand posed is detected by calculating number of fingers and angles in the hand shape.