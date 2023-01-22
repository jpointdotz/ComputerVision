import numpy as np
import cv2 as cv

'''
Parts of script:
haar_face.xml
features.npy
lables.npy folder
validation folder
'''

haar_cascade = cv.CascadeClassifier('haar_face.xml')
features = np.load('features.npy', allow_pickle = True)
labels = np.load('./lables.npy/lables.npy')


def haar():

    faces_recognizer = cv.face.LBPHFaceRecognizer_create()
    faces_recognizer.read('face_traine.yml')

    people = ['Ben Affleck', 'Madonna', 'Wolfs']

    img = cv.imread('validation\Ben Affleck\q2.jpg')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Person', img)

    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        label, confidence = faces_recognizer.predict(faces_roi)
        print(f'Label = {people[label]}, with a confidence of {confidence}')

        cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0 , (0,255,0), thickness = 2)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    return img

def show_image(img):
    cv.imshow('Detected face', img)
    cv.waitKey(0)


if __name__ == "__main__":
    image = haar()
    show_image(image)