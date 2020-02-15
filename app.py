#%%
from mtcnn import MTCNN
detector = MTCNN()
#%%
import numpy as np
import glob
import cv2
import fr_utils
from keras.models import load_model
from triplet_loss_inception import triplet_loss
import flask
from flask import render_template, Response
#%%
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = False
#%%
app = flask.Flask(__name__)
cap = cv2.VideoCapture(0)
#%%
FRmodel = load_model('triplet_loss_model.h5',custom_objects={'triplet_loss': triplet_loss})

#%%
def creating_database():
    
    #This function creates a dictionary for our database
    
    database = {}  # Python dictionary, {key:name, value:encodings}
    for images in glob.glob('images/*'):
        name = images.strip('.jpg')[7:]
        database[name] = fr_utils.img_to_encoding(images,FRmodel,path=True)
    return database        

database = creating_database()

#%%

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        _,frame = cap.read()
        img = frame
        
        #Use MTCNN to detect faces        
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
            
            min_dist = 100
            
            # Loop over the database dictionary's names and encodings.
            for (name, db_enc) in database.items():
                
                # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
                dist = np.linalg.norm(encoding-db_enc)
                #print('dist',dist)
                # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
                if dist < min_dist:
                    min_dist = dist
                    identity = name
                    
            
            if min_dist > 0.7:
                #print(" Not in the database.")
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,"Not in the database", (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imwrite('img.jpg', frame)
            else:
                #print ("It's " + str(identity) + ", the distance is " + str(min_dist))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,str(identity), (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imwrite('img.jpg', frame)
    
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('img.jpg', 'rb').read() + b'\r\n')
        
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

#%%
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, threaded=False)
