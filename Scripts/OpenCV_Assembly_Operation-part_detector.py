# all imports of dlls
from utils import *
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime 

# setting of variables
project = "Hyundai BN7R"
part = 'Bowl cop assembly'
width = 300
height = 540  # frame.shape[0]
width_v = 960
height_v = 540
green = (0,255,0)
red = (255,0,0)
blue = (255,0,0)
white = (255,255,255)
color = (0,0,0)
black = (0,0,0)
default_frame = np.ones((540,960,3), dtype='uint8')*255 # if frame not available, to be used 

# loading variables  
model = '.\best.onnx'
label = '.\classes.txt'
image = '.\IMG_5770.JPG'
video = '.\Hyundai_assembly_540.mp4'
# net = cv2.dnn.readNet(model)

# instances 
hyundai_detector = Detector(model, label) # instance only from testing purposes without the frame, see in main fuction
hyundai_side_panel = SidePanel(height, width)
#hyundai_mask = Masks(default_frame) # to be used if frame is not available
hyundai_background = Background(color, height_v, width_v)

def Hyundai_panel():
    # get labels for Hyundai YOLO detection
    labels = hyundai_detector.read_label()
    # get side panel for visualization       
    side_panel = hyundai_side_panel.get_panel()
    # default text
    h_font = cv2.FONT_HERSHEY_SIMPLEX

    now = datetime.now()
    time_now = str(datetime.now())
    current_time = now.strftime('%H.%M.%S')
    current_date = now.strftime('%d.%m.%y')

    cv2.putText(side_panel, current_time, (10,15), h_font, 0.4, (0,0,0), 1)
    cv2.putText(side_panel, current_date, (230,15), h_font, 0.4, (0,0,0), 1)

    cv2.putText(side_panel, f'Project: {project}', (10,50), h_font, 0.4, black, 1)
    cv2.putText(side_panel, f'Part: {part}', (10,80), h_font, 0.4, black, 1)

    cv2.putText(side_panel, 'Position', (120,120), h_font, 0.4, black, 1)
    cv2.putText(side_panel, 'Color', (220,120), h_font, 0.4, black, 1)

    cv2.putText(side_panel, labels[0][:7], (10,150), h_font, 0.4, color, 1)
    cv2.putText(side_panel, labels[1][:7], (10,180), h_font , 0.4, color, 1)
    cv2.putText(side_panel, labels[2][:7], (10,210), h_font, 0.4, color, 1)

    return side_panel

# main function
def main():

    read_video = cv2.VideoCapture(video)
 
    while (read_video.isOpened()):
        
        ret, frame = read_video.read() 

        # hyundai_preprocess = hyundai_detector.pre_process(net, frame)
        # hyundai_postproces = hyundai_detector.post_process(hyundai_preprocess)

        # frame = hyundai_postproces        

        if ret == True:
            x = 0
            y = 0


            # Vertical and horizontal axis
            #    
            for i in range(0,950,50):
                 cv2.line(frame, (i,0),(i,540), black, 1)
                 cv2.putText(frame, str(x), (i+5,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, black, 1)
                 x = x + 50
         
           
            for k in range(0,600,50):
                cv2.line(frame, (0,k),(960,k), black, 1)
                cv2.putText(frame, str(y), (10, k+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, black, 1)
                y = y + 50
            
            # mask for clip
            cv2.circle(frame,(400,270), 25, white,1)
            cv2.putText(frame, 'Clip', (385,240), cv2.FONT_HERSHEY_COMPLEX, 0.5, white, 1)

            # mask for pad
            corners_p = np.array([[[575,125],[755,220],[660,390],[490,310]]], np.int32)
            cv2.polylines(frame, [corners_p], True, blue, 1)
            cv2.putText(frame, 'Pad', (565,125), cv2.FONT_HERSHEY_COMPLEX, 0.5, blue, 1)

            # mask for cowl
            corners_c = np.array([[[350,150],[650,100],[830,180],[750,470],[500,400],[300,490],[200,430]]], np.int32)
            cv2.polylines(frame, [corners_c], True, green, 1)
            cv2.putText(frame, 'Cowl', (630,90), cv2.FONT_HERSHEY_COMPLEX, 0.5, green, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, baw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

            
            # hist = cv2.calcHist([gray], [0], None, [256],[0,256])
            # plt.plot(hist)
            # plt.show()

            

            # initialization of mask for color analysis of ROI, initialization after the frame is available
            # if not exists - possible to check out with default frame - at the begining 
            hyundai_mask = Masks(baw)
            mask = hyundai_mask.cowl()
            #frame = hyundai_mask.pad()
            #frame = hyundai_mask.clip()

            # hist = cv2.calcHist([gray], [0], mask, [256], [0,256],)
            # plt.plot(hist)
            # plt.show()

            final = cv2.hconcat([frame, Hyundai_panel()])    
         
            
            cv2.imshow(f'{project}', final)
            cv2.imshow(f'{project} Threshold', mask)
            # plt.hist(baw.ravel(), 2, [0,256])
            
            # plt.draw()
            # plt.pause(0.1)
            # plt.clf()
            
                       

            if cv2.waitKey(30) == ord('q'):
                break
        
        else:
            break
                        
    #read_video.release()
    #cv2.destroyAllWindows()
    

# run the main
if __name__ == '__main__':
    print(" ahoj ")
    print(video)
    main()


