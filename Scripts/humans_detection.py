import cv2
from datetime import datetime 
import numpy as np
import os
import time 

# Globally used variables
time_font = cv2.FONT_HERSHEY_SIMPLEX
grid_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
white = (255,255,255)
black = (0,0,0)
green = (0,255,0)
red = (0,0,255)
blue = (255,0,0)
precision_of_detections = 0.8

# HOG human body detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Flags for various functions
show_grid_flag = False
show_transparent_area_flag = True
show_ROI_frame_flag = True

# Video source
video = '.\Violation_detection\Video.mp4'

### Function to get actual time and date
def get_actual_time():
    now = datetime.now()
    time_now = str(datetime.now())
    current_time = now.strftime('%H:%M:%S')
    current_date = now.strftime('%d.%m.%y')

    return current_time, current_date
###

### Function to get grid for beter definition of violation area
def get_grid(frame, width, height):
    
    x = y = 0

    for i in range(0,height,50):
        cv2.line(frame, (i,0),(i,width), white, 1)
        cv2.putText(frame, str(x), (i+5,15), grid_font, 0.5, white, 1)
        x = x + 50
         
    for k in range(0,width,50):
        cv2.line(frame, (0,k),(height,k), white, 1)
        cv2.putText(frame, str(y), (10, k+10), grid_font, 0.5, white, 1)
        y = y + 50

    return frame
###

### Function to get transparent area within ROI
def get_transparent_area(frame,w,h,d1,d2,color):
    
    transparent_area = frame[w:w+d1, h:h+d2]
    cv2.rectangle(frame, (w,h), (w+d1, h+d2), color ,2)

    # Transparency for transparent area 
    overlay = frame.copy()
    alpha = 0.4       # transparency
    cv2.rectangle(overlay,(w,h), (w+d1, h+d2), color, -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    
    return frame
###

### Function to get detections and frame ROI
def get_detections(ROI, padding, n):
    gray_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    (regions, weights) = hog.detectMultiScale(gray_ROI, padding=padding, scale = 1.02)

    detection_flag = False
    
    for i, (x,y,w,h) in enumerate(regions):
        if weights[i] > precision_of_detections:
            cv2.rectangle(ROI, (x,y), (x+w, y+h), (0,0,255), 2)
            if show_ROI_frame_flag == True:
                cv2.putText(ROI, f'Detection found.', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
            detection_flag = True

            # show transparent area if True in flags
            if show_transparent_area_flag == True:
                ROI = get_transparent_area(ROI, x, y, w, h, red)
            
    # show frame around ROI if True in flags
    if show_ROI_frame_flag == True:
        cv2.rectangle(ROI, (0,0), (ROI.shape[1], ROI.shape[0]), blue, 4, cv2.LINE_AA)
        cv2.putText(ROI, f'ROI n.{n}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 2)
        # if flag == True:
        #     cv2.putText(ROI, f'{stop_watch}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 2)    

    return ROI, detection_flag
###

def main():
       
    read_video = cv2.VideoCapture(video)
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        
        ret, frame = read_video.read()  

        width, height, _ = frame.shape  
         
        # resized frame to decraese expanse of computation
        resized_frame = cv2.resize(frame, (height//2, width//2), interpolation=cv2.INTER_AREA)
        resized_width, resized_height, _ = resized_frame.shape  

        print(resized_frame.shape)

        # calculation of fps
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)

        # definitions of ROIs (manual for now)
        ROI = resized_frame[180:360, 320:640]    

        # get detections within ROIs based on hyperparameters
        padding = (8,8)
        stride = None     # needed to be put manually into function ('winStride = (4,4)') - slower but higher precission
        detection_ROI, detection_flag_ROI = get_detections(ROI, padding, 1)
          
        # get and draw actual time onto frame
        current_time, current_date = get_actual_time()

        # count seconds to save image
        counter = 0
        print(type(current_time))

        # put time stamp onto ROI
        # if detection_flag_ROI == True:
        #     cv2.putText(detection_ROI, 'Flag True', (10,60), time_font, 0.5, red, 2)     

        # final frame with detections
        resized_frame[180:360, 320:640]   = detection_ROI

        # text put onto the frame
        cv2.putText(resized_frame, f'Detector: HOG SVM Descriptor', (30,18), time_font, 0.4, green, 1)
        cv2.putText(resized_frame, current_time, (30,30), time_font, 0.4, green, 1)
        cv2.putText(resized_frame, current_date, (30,42), time_font, 0.4, green, 1)
        cv2.putText(resized_frame, f'FPS: {fps}', (30,54), time_font, 0.4, green, 1)
        cv2.putText(resized_frame, f'Origin: {height}x{width}', (30,66),time_font, 0.4, green, 1)
        cv2.putText(resized_frame, f'Sliced: {resized_height}x{resized_width}', (30,78),time_font, 0.4, green, 1)
        cv2.putText(resized_frame, f'ROI: 180x320', (30,90),time_font, 0.4, green, 1)
        cv2.putText(resized_frame, f'No. of ROIs: {1}', (30,102),time_font, 0.4, green, 1)
        cv2.putText(resized_frame, f'Confidence level > {precision_of_detections}', (30,114),time_font, 0.4, green, 1)
        cv2.putText(resized_frame, f'Hyperparamter: padding = {padding}', (30,126),time_font, 0.4, green, 1)
        cv2.putText(resized_frame, f'Hyperparamter: stride = {stride}', (30,138),time_font, 0.4, green, 1)

        # show grid if True in switchers
        if show_grid_flag == True:
            resized_frame = get_grid(resized_frame, width, height) 

        if ret == True:  

            # cv2.imshow(f'Violation original frame.', frame)
            # cv2.imshow(f'Violation sliced frame.', sliced_frame)
            cv2.imshow(f'Humans detection.', resized_frame)          

            if cv2.waitKey(30) == ord('q'):
                break

        else:
            break

if __name__ == '__main__':
    main()
    