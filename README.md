# Face-Recognition-in-Real-Time
Create your own databse, compile tripletloss with pre-trained FaceNet model, run real-time face recognition on local host by typing "flask run" in command prompt

**Classification vs One Shot Learning**

In classic Convolutional Neural Network (CNN) we train layers of CNN to learn features of objects we want to classify. For this we require
a lot of data during during training. For example classifying various animals, vehicles etc. 
Now suppose we want to classify a new animal we have to re-train the entire network or use transfer learning.

In case of "One Shot Learning" we need only one training example per class/ category. In this we use "Siamese" network, Siamese mean twins.
Input to Siamese network is an image and the output is "encoding of the input image". Similarly another image is fed to this network to 
calculate the image encodings.
Now we calculate the distance between these encodings. If this distance is less than threshold means same person else not.
The distance is calculated by the following formula

**d(x(1), x(2)) = || f(x(1)) – f(x(2) ||<sup>2**

Here we use **triplet loss function**
During training we have three images: an anchor image of a person (A), a positive image of the same person (P), negative image of a different person (N)

Triplet loss function is calculated by following formula

**L(A,P,N) = max(|| f(A) – f(P) ||2 - || f(A) – f(N) ||2 + α, 0)**

α: Hyperparameter, helps in preventing trivial solution.

For example α = 0.2, we want this α (margin) difference between positive and negative image distance

The above implementation is given in triplet_loss_inception.py

This program compiles the triplet loss with the pre-trained FaceNet model and store the model as triplet_loss_model.h5

Next step is to create your own database, for this you need one single image of every individual. The function "def creating_database()" generated a dictionary which stores "name" as "key" and "image embeddings" as "value".

The above function is in app.py. After creating the database you can grab an image from your webcam using OpenCV. Multitask Cascaded Convolutional Network (MTCNN) is used to detect faces from images.
After detecting face you crop the image and feed it into FcaeNet model for calculating encodings. This encoding is compared with the encodings from the database. L2 distance norm is used for comparison. 
If the distance between the calculated encoding and stored encoding is less than a minimum threshold, the program outputs the name of the person else "Not in database".

References:
Coursera: https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning
More on MTCNN: https://towardsdatascience.com/how-does-a-face-detection-program-work-using-neural-networks-17896df8e6ff
