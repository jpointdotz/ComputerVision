from platform import release
import numpy as np
import cv2

'''
Code for object detection official OPENCV documentation code.
Code for actual application done by Jan Zarnay.
'''

class Detector():
    def __init__(self, model, label):
        self.model = model
        self.label = label
        
    
    # Reading of lables for YOLO detection
    def read_label(self):
        with open(self.label, 'r') as f:
            class_names = f.read().split('\n')
            return class_names

    # Creating a 4D blob from a frame
    def pre_process(self, net, frame):

        self.frame = frame 
        
        # Standard dimensions for inference in YOLO unless trained otherwise
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        
        blob = cv2.dnn.blobFromImage(self.frame, 1/255,  (self.INPUT_WIDTH, self.INPUT_HEIGHT), [0,0,0], 1, crop=False)

        net.setInput(blob)

        # Runs the forward pass to get output of the output layers.
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        return outputs

    def post_process(self, outputs):
        
        # Standard threshold values / to be put into main later on
        SCORE_THRESHOLD = 0.5
        NMS_THRESHOLD = 0.45
        CONFIDENCE_THRESHOLD = 0.45

        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []
        # Rows.
        rows = outputs[0].shape[1]
        image_height, image_width = self.frame.shape[:2]
        # Resizing factor.
        x_factor = image_width / self.INPUT_WIDTH
        y_factor =  image_height / self.INPUT_HEIGHT
        # Iterate through detections.
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                # Get the index of max class score.
                class_id = np.argmax(classes_scores)
                #  Continue if the class score is above threshold.
                if (classes_scores[class_id] > SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)
    # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]             
            # Draw bounding box.             
            cv2.rectangle(self.frame, (left, top), (left + width, top + height), (255,0,0), 3)
            # Class label.                      
            label = "{}:{:.2f}".format(self.classes[class_ids[i]], confidences[i])             
            # Draw label.             
            self.draw_label(self.frame, label, left, top)
    
        return self.frame

    

class SidePanel():

    def __init__(self,width,height):
        self.width = width
        self.height = height
        
    
    #returns side panel based with size based on input data
    def get_panel(self):
 
        panel = np.ones((self.width, self.height, 3), dtype='uint8')*255
        panel[0:20, 0:500] = (panel[0:20, 0:500] * 100).astype('uint8')
        
        return panel

class Masks():

    def __init__(self, frame):
        self.frame = frame
        

    def cowl(self):
        corners_c = np.array([[[350,150],[650,100],[830,180],[750,470],[500,400],[300,490],[200,430]]], np.int32)
        mask = np.zeros(self.frame.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, corners_c, 255)
        masked_c = cv2.bitwise_and(self.frame, self.frame, mask=mask)

        return masked_c
    
    def pad(self):
        corners_p = np.array([[[575,125],[755,220],[660,390],[490,310]]], np.int32)
        mask = np.zeros(self.frame.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, corners_p, (255,0,0))
        masked_p = cv2.bitwise_and(self.frame, self.frame, mask=mask)

        return masked_p

    def clip(self):
        mask = np.zeros(self.frame.shape[:2], dtype="uint8")
        cv2.circle(mask, (400,270), 25, 255, -1)
        masked_cl = cv2.bitwise_and(self.frame, self.frame, mask=mask)

        return masked_cl


# class for definition of background to compare the color with

class Background():

    def __init__(self, color, width, height):

        self.color = color
        self.width = width
        self.height = height     

    def colored_background(self): 

        colored_background = np.ones((self.width ,self.height,3), dtype="uint8")
        colored_background[::] = colored_background[::] * self.color   

        return colored_background



      


      
