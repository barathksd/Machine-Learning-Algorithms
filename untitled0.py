# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 22:27:23 2019

@author: Lenovo
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import numpy as np
import scipy
import cv2
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import GaussianDropout
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, jaccard_score
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.datasets import mnist 

file = 'C:\\Users\\Lenovo\\Desktop\\data\\demo.png'


#Reference libraries 
#sklearn.model_selection -> Gridsearch, train_test_split
#sklearn.preprocessing -> StandardScalar, One hot encoder, .. 
#sklearn.linear_model -> lasso, linear regression, stochastic gradient descent,
#sklearn.ensemble -> RandomForestClassifier, RandomForestRegressor,
#sklearn.metrics -> max_error, mean_absolute_error, f1_score, jaccard_score, roc_curve, auc

#keras.models -> model_from_json, model_from_yaml, save_model, Input, Sequential, Model
#keras.metrics -> accuracy, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy
#keras.layers -> Dense, Maxpooling2D, Dropout, Conv2D, concatenate, Conv2DTranspose...


#loads data and splits it into test and train
def load():
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = load()


def save_model(model):    
    json_string = model.to_json()
    open('models.json', 'w').write(json_string)
    
def load_model():
    model = model_from_json(open('models.json').read())
    model.load_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weightss.hdf5')
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=sm.losses.bce_jaccard_loss,
                  optimizer=opt,
                  metrics=[sm.metrics.iou_score,f1])
    return model

# self defined F1 metric
def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    def recall(y_true, y_pred):
        
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def resunit(inp,nf=32,Ksize=3,padding='same',strides=1,BN='True',BN_first=True,activation='relu',sno='0'):
    
    
    conv = Conv2D(nf,
                  kernel_size=Ksize,
                  strides=strides,
                  padding=padding,
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  name='C'+sno)
    
    x = inp
    
    if BN_first:
        if BN:
            x = BatchNormalization(name='BN'+sno)(x)
        if activation is not None:
            x = Activation(activation,name='Act'+sno)(x)
        x = conv(x)
    else:
        x = conv(x)
        if BN:
            x = BatchNormalization(name='BN'+sno)(x)
        if activation is not None:
            x = Activation(activation,name='Act'+sno)(x)
    return x


def resnet(input_shape,n_classes=2,nf=32,nb=4):

    Ksize=3
    padding='same'
    strides=1
    ld = {}
    nf = 32
    
    
    inputs = Input(input_shape)
    x = resunit(inputs,nf=nf,Ksize=3,padding='same',strides=1,BN_first=False,activation='relu',sno='00')
    x = Conv2D(nf,kernel_size=3,strides=1,padding='same',kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name='C01')(x)
    x = add([inputs, x],name='A0')
    ld['A0'] = x
    for i in range(nb):
        nf = 2*nf
        y = resunit(x,nf,strides=2,sno=str((i+1)*10))
        y = resunit(y,nf,sno=str((i+1)*10+1))
        x = Conv2D(nf,kernel_size=1,padding='same',strides=2,name='1C'+str((i+1)*10))(x)
        x = add([x,y],name='A0'+str(i+1))
        ld['A'+str(i+1)] = x
    
#    x1 = AveragePooling2D(pool_size=2)(x)
#    y1 = Flatten()(x1)
#    outputs = Dense(num_classes,
#                    activation='softmax',
#                    kernel_initializer='he_normal')(y1)
        
    for i in range(nb):
        nf = int(nf/2)
        x = Conv2DTranspose(nf, kernel_size=Ksize, strides=2, padding=padding, kernel_initializer='he_normal')(x)
        x = concatenate([ld['A'+str(nb-i-1)],x],axis=3,name='CT'+str(i))
        y = resunit(x,nf,sno=str((i+5)*10))
        y = resunit(y,nf,sno=str((i+5)*10+1))
        x = Conv2D(nf,kernel_size=1,padding='same',strides=1,name='1C'+str((i+5)*10))(x)
        x = add([x,y],name='A0'+str(i+5))
    
    
    x = Conv2D(n_classes,kernel_size=1,padding='same',strides=1,name='out')(x)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=x)
    return model

def train():
    
    model = resnet(input_shape=(320,512,1))
    print(model.summary())
    
train()

























































