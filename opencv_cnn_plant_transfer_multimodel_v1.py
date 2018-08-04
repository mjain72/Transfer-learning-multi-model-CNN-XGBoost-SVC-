#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 08:47:54 2018

@author: mohit

"""


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm #to show progress
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import xgboost as xgb
from sklearn import svm
from sklearn.externals import joblib


#Define Image Directory
image_dir_test = 'images/plants/test/'
image_dir_train = 'images/plants/train/'

#define the senstivity to control the selection of green color
sensitivity = 30
#define final image size for processing
image_size = 64
'''
define a function to remove background from the image to only leave the green leaves, blurring and normalizing. Followed by resizing the images
to 64 x 64 size

'''

def image_transformation(imageName, sensitivity):
    
    imagePlatCV = cv2.imread(imageName) #read image
    hsvImage = cv2.cvtColor(imagePlatCV, cv2.COLOR_BGR2HSV)
    #define the range for green color
    lower_green = np.array([60 - sensitivity, 100, 50])
    upper_green = np.array([60 + sensitivity, 255, 255])
    # threshold the hsv image to get only green colors
    mask = cv2.inRange(hsvImage, lower_green, upper_green)
    #apply bitwise_and between mask and the original image
    greenOnlyImage = cv2.bitwise_and(imagePlatCV, imagePlatCV, mask=mask)
    #lets define a kernal with ones
    kernel0 = np.ones((15,15), np.uint8)
    #lets apply closing morphological operation
    closing0 = cv2.morphologyEx(greenOnlyImage, cv2.MORPH_CLOSE, kernel0)
    #blur the edges
    blurImage = cv2.GaussianBlur(closing0, (15,15), 0)
    blurImageColor = cv2.cvtColor(blurImage, cv2.COLOR_BGR2RGB)#to make it work with right color
    #resize image
    resizeImage = cv2.resize(blurImageColor, (image_size, image_size), interpolation=cv2.INTER_AREA)
    resizeImage = resizeImage/255 #normalize
    #resizeImage = resizeImage.reshape(image_size,image_size,3) #to make it in right dimensions for the Keras add 3 channel
    print(resizeImage.shape)
    return resizeImage

#define list of plant species
classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed'
           , 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
'''
Data extraction: The loop below will create a data list containing image file path name, the classifcation lable 
(0 -11) and the specific plant name

'''
train = [] #data list
for species_lable, speciesName in enumerate(classes):
    for fileName in os.listdir(os.path.join(image_dir_train, speciesName)):
        train.append([image_dir_train + '{}/{}'.format(speciesName, fileName), species_lable, speciesName])
        
        
#convert the list into dataframe using Pandas
trainigDataFrame = pd.DataFrame(train, columns=['FilePath', 'PlantLabel', 'PlantName'])

#Suffle the data
seed = 1234 #define seed to get consistent results
trainigDataFrame = trainigDataFrame.sample(frac=1, random_state=seed)
trainigDataFrame = trainigDataFrame.reset_index()

#Prepare the images for the model by preprocessing

X = np.zeros((trainigDataFrame.shape[0], image_size, image_size, 3)) #array to store image after image_transformfunction

for i, fileName in tqdm(enumerate(trainigDataFrame['FilePath'])):
    print(fileName)
    newImage = image_transformation(fileName, sensitivity)
    X[i] = newImage

#Convert lables to categorical 
y = trainigDataFrame['PlantLabel']
y = np.array(y)
y_labels_cat, _ = pd.factorize(y)

'''
Define performance reporting function
'''

def model_performance(y_val, y_pred_prob, model_name, classes):
    #confusion matrix and classification report
    
    cm = confusion_matrix(y_val, y_pred_prob.argmax(axis=1))
    
    print(classification_report(y_val, y_pred_prob.argmax(axis =1), target_names=classes))
    
    #plot F1-score vs. Classes
    
    f1Score = f1_score(y_val, y_pred_prob.argmax(axis=1), average=None)
    
    y_pos = np.arange(len(classes))
    plt.bar(y_pos, f1Score)
    plt.xticks(y_pos, classes)
    plt.ylabel('F1 Score')
    plt.title('F1 Score of various species after classification (' + model_name + ')')
    plt.show()
    return cm, f1Score

'''
Build the initial network from pretrained model that will be used for transfer learning. We will use first two blocks from 
VGG19 pretrained model
'''
#input layer
imageInput = Input(shape=(image_size, image_size, 3), name='input_1')
#Block 1 - layers name same as the layers in pretrained model
model = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(imageInput)
model = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(model)
model = MaxPool2D(pool_size= (2,2), strides=(2,2), name='block1_pool')(model)

#Block 2

model = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(model)
model = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(model)
model = MaxPool2D(pool_size= (2,2), strides=(2,2), name='block2_pool')(model)
model_initial = Model(inputs=imageInput, outputs=model)
model_initial.summary()

#define the dictionary of layers
layer_dict = dict([(layer.name, layer) for layer in model_initial.layers])
print(layer_dict)

#load weights from VGG19 model
weights_path = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'#from https://github.com/fchollet/deep-learning-models/releases/
model_initial.load_weights(weights_path, by_name=True)

#set weights for initial layers
for i, layer in enumerate(model_initial.layers):
    weights = layer.get_weights()
    model_initial.layers[i].set_weights(weights)


model_final = Model(inputs = model_initial.input, outputs = model_initial.output)

features = [] #List to extract features using pretrained model weights
for i in tqdm(X):
    image = np.expand_dims(i, axis=0)
    featurePredict = model_final.predict(image)
    features.append(featurePredict)
    
#convert to numpy array from list    
featuresArray = np.array(features)

#reshape to get in the right shape for training
featuresReshape = np.reshape(featuresArray, (featuresArray.shape[0], featuresArray.shape[2], featuresArray.shape[3], featuresArray.shape[4]))
print(featuresReshape.shape)


#split dataset into Train, Dev and Validation
X_train_dev, X_val, y_train_dev, y_val = train_test_split(featuresReshape, y_labels_cat, test_size=0.10, random_state=seed)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev,y_train_dev, test_size=0.10, random_state=seed)


#reshape the data for XGBoost and SVC

X_train_XG_SVC = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3]))
X_dev_XG_SVC = np.reshape(X_dev, (X_dev.shape[0], X_dev.shape[1]*X_dev.shape[2]*X_dev.shape[3]))
X_val_XG_SVC = np.reshape(X_val, (X_val.shape[0], X_val.shape[1]*X_val.shape[2]*X_val.shape[3]))

'''
Build classifiers on top of model build using transfer learning
'''

'''
XGBosst Model
'''

xgb1 = xgb.sklearn.XGBClassifier(
        learning_rate =0.1,
         n_estimators=1000,
         max_depth=5,
         min_child_weight=11,
         gamma=0.1,
         subsample=0.8,
         colsample_bytree=0.7,
         objective='multi:softprob',
         n_jobs=-1,
         scale_pos_weight=1,
         seed=seed)

xgb1.fit(X_train_XG_SVC, y_train).score(X_dev_XG_SVC, y_dev)

y_pred_xgb = xgb1.predict_proba(X_val_XG_SVC)

joblib.dump(xgb1, 'xgbClassifier2.pkl')#save model

'''
End XGBoost Model
'''


'''
CNN Model
'''
#define a classifier function with extracted features as input
def cnn_model():
    classifier = Sequential()
    classifier.add(Conv2D(256, kernel_size=(3,3), padding='same', strides=(1,1), input_shape=(features[1].shape[1], features[1].shape[2],features[1].shape[3]), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.1))
    classifier.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same',  activation='relu'))  
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.1))
    classifier.add(Conv2D(1024, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.1))
    classifier.add(Flatten())
    classifier.add(Dense(units=1024, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=512, activation='relu'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units=12, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()    
    return classifier

#define classifier
classifier_CNN = cnn_model()

batch_size = 64
epochs = 50
checkpoint = ModelCheckpoint('model_vgg19_final2.h5', verbose=1, save_best_only=True)

y_train_cnn = pd.get_dummies(y_train)
y_train_cnn = y_train_cnn.values
y_dev_cnn = pd.get_dummies(y_dev)
y_dev_cnn = y_dev_cnn.values
y_val_cnn = pd.get_dummies(y_val)
y_val_cnn = y_val_cnn.values

#train model
trainingModel_CNN = classifier_CNN.fit(x=X_train, y=y_train_cnn, batch_size=batch_size, 
                           epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_dev, y_dev_cnn))

'''
End CNN model
'''


'''
SVC Model
'''
svcClassifier = svm.SVC(kernel='rbf', probability=True, verbose=True, decision_function_shape='ovr')
svcClassifier.fit(X_train_XG_SVC, y_train).score(X_dev_XG_SVC, y_dev)

y_pred_svc = svcClassifier.predict_proba(X_val_XG_SVC)

joblib.dump(svcClassifier, 'svcClassifier1.pkl')

'''
End SVC Model
'''

'''
Model comparison
'''
#XGBoost
clfXGB = joblib.load('xgbClassifier2.pkl')
y_pred_xgb = clfXGB.predict_proba(X_val_XG_SVC)
cmXGB, f1XGB = model_performance(y_val= y_val, y_pred_prob=y_pred_xgb, model_name='XGBoost', classes=classes)

#SVC
clfSVC = joblib.load('svcClassifier1.pkl')
y_pred_svc = clfSVC.predict_proba(X_val_XG_SVC)
cmSVC, f1SVC = model_performance(y_val= y_val, y_pred_prob=y_pred_svc, model_name='SVC', classes=classes)

#CNN
final_model_CNN = load_model("model_vgg19_final2.h5")
y_pred_CNN = final_model_CNN.predict(X_val)
cmCNN, f1CNN = model_performance(y_val, y_pred_CNN, model_name='CNN', classes=classes)


'''
Ensemble of the models
'''

y_ensemble = (y_pred_xgb + y_pred_svc + y_pred_CNN)/3

cmEnsemble, f1Ensemble = model_performance(y_val, y_ensemble, model_name='Ensemble', classes=classes)

