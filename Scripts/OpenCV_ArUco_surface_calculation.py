import cv2
import math
import numpy as np
from collections import deque

# Definition of exact ArUco dictionary and Params
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
ArucoParams = cv2.aruco.DetectorParameters_create()

# Declaration of variables and lists
None 

# Calibration for length, input frame, output list with coordinates 
# ID12 x,y and ID13 x,y + ratio length pixels / cm
# length pixels, length cm
# not proper calibration --> just infomative in X axis
def calibration(video_stream):

    (corners, ids, rejected) = cv2.aruco.detectMarkers(video_stream, arucoDict,
        parameters=ArucoParams)

    real_length = 317

    try:
        ids = ids.flatten()

        # extract the top-left marker
        i = np.squeeze(np.where(ids == 13))
        topLeft = np.squeeze(corners[i])[0]
        
        # extract the top-right marker
        i = np.squeeze(np.where(ids == 12))
        topRight = np.squeeze(corners[i])[1]
        
        length = math.sqrt((topLeft[0]-topRight[0])**2 + (topLeft[1]-topRight[1])**2)

        ratio = length / real_length
        
        return topLeft, topRight, length, ratio


    except:
        return None, None, None, None 


# Fucntion finds new IDS and put coordianates into dynamic list for further processing
# on the same time removes ID1 and ID3 .. no better solution found :(
def new_arucos_for_processing(video_stream):

    (corners, ids, rejected) = cv2.aruco.detectMarkers(video_stream, arucoDict,
        parameters=ArucoParams)

    try:
        ids = ids.flatten()
        polygon_edges = deque(maxlen=len(ids))

        for aucorner, idsn in zip(corners, ids):
         
            # extract the top-left marker
            k = np.squeeze(np.where(ids==idsn))
            topLeft = np.squeeze(corners[k])[0]

            # avoid the fixed ArUcos to be calculated
            if idsn !=13 and idsn !=12:
                polygon_edges.appendleft(topLeft)

        return polygon_edges
 

    except:
        return None


# function to sort the given list with centorid and atan2
def sort(list_of_edges):

    try:
        def centeroidpython(list_of_edges):
            x, y = zip(*list_of_edges)
            l = len(x)
            return sum(x) / l, sum(y) / l
        
        cent_1, cent_2 = centeroidpython(list_of_edges)

        xy_sorted = sorted(list_of_edges, key = lambda x: math.atan2((x[1]-cent_2),(x[0]-cent_1)))
        xy_sorted_int = np.array(xy_sorted, dtype='int32')
    
    except:
        xy_sorted_int= None 
        cent_1 = None
        cent_2 = None

    return xy_sorted_int, cent_1, cent_2


# function to draw elemnts inside the frame
def drawings (video_stream, topLeft, topRight, length, ratio, xy_sorted, cent_1, cent_2):

    try:
        cv2.fillPoly(video_stream, pts=[xy_sorted], color= (150,150,150))
        cent_1 = int(cent_1) 
        cent_2 = int(cent_2)
        cv2.circle(video_stream, (cent_1, cent_2), 3, (0,0,255), -1)
        
        area = cv2.contourArea(xy_sorted)
        cv2.putText(video_stream, 'Area = {:.01f} pixels'.format(area), (cent_1+30, cent_2-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,230), 1)
        area_mm = area / ratio  
        cv2.putText(video_stream, 'Area around {:.01f} mm2'.format(area_mm), (cent_1+30, cent_2 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,230), 1)
                      
        x1, y1 = int(topLeft[0]), int(topLeft[1])
        x2, y2 = int(topRight[0]), int(topRight[1])

        cx1 = (x1 + x2) // 2
        cy1 = (y1 + y2) // 2

        cv2.circle(video_stream, (x1, y1), 3, (0,0,255), -1)
        cv2.circle(video_stream, (x2,y2), 3, (0,0,255), -1)

        cv2.putText(video_stream, 'Calibration only for reference:', (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,230), 1)
        cv2.putText(video_stream, 'Calibration X-line length: {:.01f} pixels'.format(length), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,230), 1)
        length_mm = length / ratio
        cv2.putText(video_stream, 'Calibration X-line length: {:.01f} mm'.format(length_mm), (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,230), 1)

        for i, k in enumerate(xy_sorted):
            cv2.putText(video_stream, '{}'.format(i), k, cv2.FONT_HERSHEY_COMPLEX, 1 , (0,0,255), 2)
            cv2.circle(video_stream, k, 3, (0,0,0), -1)
            cv2.line(video_stream, k, (cent_1, cent_2), (0,0,0,),2) 

    except:
        cv2.putText(video_stream, '[ERROR] Calibration not done.', (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,230),1)
        return video_stream 

    return video_stream


# Main funciton to start video and show results
def __main__():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1600)
    cap.set(4, 1200)
   

    while True:
        ret, frame = cap.read()
        cv2.flip(frame, 0)

        if ret == False:
            print('Camera failed.')
            break
        
        # --> calibration with two ArUco points ID12 and ID13
        topLeft, topRight, length, ratio = calibration(frame)

        # --> getting list of edges
        list_edges = new_arucos_for_processing(frame)

        # --> sort the list_edges acc. to centroid and atan2
        xy_sorted, cent_1, cent_2 = sort(list_edges)

        # --> drawing objects in frame
        frame_with_drawings = drawings(frame, topLeft, topRight, length, ratio, xy_sorted, cent_1, cent_2)
             
        cv2.imshow('Video', frame_with_drawings)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    __main__()