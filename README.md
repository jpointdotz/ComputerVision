Repo consists:

- Detection of pose on bicycle to perform proper fitting. Script uses Mediapipe library and OpenCV functions. Now only for static pictures, video ongoing.

 - Some results of Object detection trained models. I was using standard TF object detection API (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/#) and training done locally or on Google Colab. The data to create the dataset has been collected manually from internet (Fork lift) but as well by taking the pictures manually (Ford, Gesture and Nokia) in order to better understand what works, what not, and what type of data is suitable for convolution filters when dataset is hard to find. Transfer learning has been done on top of Single Shot Mobilenet V2 FPNLite 640x640 DNN which works the best for me. I "played" as well with Faster R-CNN, Mask R-CNN and Detectron, however "live on camera" detection was too slow for my HW. I was using Yolo v5 along to Mediapipe (hand detection in my case) to detect the mechanical part along with hand position exact position during assembly of parts. The results are obviously not shown - part is not in production. The internal project has been stopped.  
 
 - In order to avoid missing component within assembly process, I have created the script that compares the actual frame with desired frame. The HSV color space is more suitable for "real life", however based on my results using standard camera I got better results with RGB color space that is quite faster. The light conditions are essential for effective live analysis on production line and in the best case the camera with fixed focal length to be used. You can see two models of NIKE Jordan. Analyser divides the picture to chosen number of segments, and compares histogram of analyzed frame with respective segment of desired frame. Correlation is shown as heatmap for every segment. Another alternative is to use only gray scale, however the results were not OK under hard light conditions. 
 
 - Some videos of working with ArUco markers. 
 
 - You can see some scripts in "Scripts" folder. 
 
 - Passenger detections - Detection of passengers through the HOG OpenCV function to quick detect human body with some tweaks (changed padding, definition of ROI to get good precision by keeping input dimension for HOG, ...). Considered two additional tasks: Kalman's tracking and/or object classification of ROI (decreased size for classificator, to be trained by transfer learing on top of some pre-trained DNNs to recognize humans behaviour). 
