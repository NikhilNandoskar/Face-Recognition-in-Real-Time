# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:42:48 2020

@author: Nando's Lenovo
"""

from mtcnn import MTCNN
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

#import matplotlib.pyplot as plt
#mat_img = plt.imread('images/Nikhil2.jpg',format=float)
#plt.imshow(mat_img)
#test_img = cv2.imread('images/Nikhil2.jpg')
#print(test_img.shape)
#Mohit test_img[1500:2200,1200:2000]
# AMisha = test_img[1200:1900,1800:2700]
# Nikhil test_img[100:850,1000:1900]
# Nikhil2 test_img[0:750,180:600]
# Jaski test_img[650:1600,900:1600]
# Nikhil_test test_img[0:300,290:450]
#crop_img =  test_img[0:750,180:600]
#cv2.imshow("crop", crop_img)

#resized = cv2.resize(crop_img, (96,96))
#cv2.imshow("test",resized)
#cv2.waitKey(0)
"""
test_img = cv2.imread('images/Nikhil2.jpg')
crop_img =  test_img[0:750,180:600]
cv2.imwrite('images/Nikhil2.jpg',crop_img)"""
"""
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = cv2.imread(filename)
	# plot the image
	cv2.imshow("img",data)
	# get the context for drawing boxes
	#ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, x2, y2 = result['box']
		# create the shape
		rect = cv2.Rectangle(data,(x, y),(x+x2,y+y2), (0,0,255), 2)
		# draw the box
		#ax.add_patch(rect)
	# show the plot
	return rect

def draw_image_with_boxes(filename, result_list):
    
	data = cv2.imread(filename)
	
	cv2.imshow("img",data)
    
	for result in result_list:
		x, y, x2, y2 = result['box']
		rect = cv2.Rectangle(data,(x, y),(x+x2,y+y2), (0,0,255), 2)
        
    print("u")
    return rect
    #return ret

if __name__ == "__main__":
    filename = 'Nikhil.jpg'
    # load image from file
    pixels = cv2.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    # display faces on the original image
    result = draw_image_with_boxes(filename, faces)"""
"""    
detector = MTCNN()
cap = cv2.VideoCapture(0)
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
    
            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
    
            cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()"""



    
    
detector = MTCNN()
cap = cv2.VideoCapture(0)
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    img = frame
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
    
            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
    
            
    
        
            key = cv2.waitKey(100)
            cv2.imshow("preview", frame)

    if key == 27: # exit on ESC
        break
cv2.destroyAllWindows()
cap.release()


