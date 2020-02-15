# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 05:51:21 2020

@author: Nando's Lenovo
"""


#%%
#FRmodel = load_model('triplet_loss_model.h5',custom_objects={'triplet_loss': triplet_loss})  
from mtcnn import MTCNN
detector = MTCNN()
#%%

import numpy as np
import glob
import cv2
import fr_utils
from keras.models import load_model
from triplet_loss_inception import triplet_loss
#%%
FRmodel = load_model('triplet_loss_model.h5',custom_objects={'triplet_loss': triplet_loss}) 

#%%
def creating_database():
    
    #This function creates a dictionary for our database and stores in pickle format 
    
    #Returns: pickled dictionary
    
    database = {}  # Python dictionary, {key:name, value:encodings}
    for images in glob.glob('images/*'):
        name = images.strip('.jpg')[7:]
        database[name] = fr_utils.img_to_encoding(images,FRmodel,path=True)
    return database        

database = creating_database()
#%%
#databases = np.load('database.npy',allow_pickle='TRUE').item()  

#img = cv2.cvtColor(cv2.imread("Nikhil2.jpg"), cv2.COLOR_BGR2RGB)
#img = cv2.imread('images/Nikhil_test.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    #detector = MTCNN()
    cap = cv2.VideoCapture(0)
    while True: 
        #Capture frame-by-frame
        __, frame = cap.read()
        img = frame
        #Use MTCNN to detect faces
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray_img.ndim==2:
            w, h = gray_img.shape
            gray = np.empty((w, h, 3), dtype=np.uint8)
            gray[:, :, 0] = gray[:, :, 1] = gray[:, :, 2] = gray_img
            result = detector.detect_faces(gray)
        
        for person in result:
            bounding_box = person['box']
    
            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (155,155,255),2)
            height,width,channels = frame.shape
            #print('frame',frame.shape)
            crop_image = frame[max(0, bounding_box[1]):min(height, bounding_box[1] + 
                               bounding_box[3]), max(0, bounding_box[0]):min(width, bounding_box[0]+bounding_box[2])]
    
            encoding = fr_utils.img_to_encoding(crop_image,FRmodel,path=False)
            
            min_dist = 50
            
            # Loop over the database dictionary's names and encodings.
            for (name, db_enc) in database.items():
                
                # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
                dist = np.linalg.norm(encoding-db_enc)
                #print('dist',dist)
                # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
                if dist < min_dist:
                    min_dist = dist
                    identity = name
                    
            
            if min_dist > 0.9:
                #print(" Not in the database.")
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,"Not in the database", (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                #print ("It's " + str(identity) + ", the distance is " + str(min_dist))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,str(identity), (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
    #display resulting frame
            cv2.imshow('frame',crop_image)
            cv2.imshow('output',img)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break
    #When everything's done, release capture
    cap.release()
    cv2.destroyAllWindows()
            
#%%    
main()