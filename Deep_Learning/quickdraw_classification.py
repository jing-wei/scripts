#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 22:53:22 2020

@author: 
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import cv2 
import sklearn 
import os 
import sys 
%matplotlib inline 




#############################################################################
# load 
#######
######### Please set wk_dir to folder where train.npz and test.npz are saved. 
######### Later output files will also be saved in the folder. 
#######
wk_dir = "[define folder path to data]"
train = np.load('/'.join([wk_dir, "train.npz"]), encoding='latin1', allow_pickle=True) 
test = np.load('/'.join([wk_dir, "test.npz"]), encoding='latin1', allow_pickle=True) 


#############################################################################
# exploration 
print(train.files) 
train['arr_0'][0] 
train['arr_1'][0] 
np.unique(train['arr_1'])
# array([0, 1, 2, 3, 4, 5])

vis = train['arr_0'][0].reshape((28,28)) 
vis = test['arr_0'][0].reshape((28,28))


img = vis.astype(np.uint8) 

# The train set is much smaller than the test set by about 40 fold. 
# In order to have decent performance, data augmentation to significantly increase train set size, while also introduce 
# variety will be needed. 

#############################################################################
# data augmentation 
# This function is built based on references from a number of resouces. 
## https://docs.opencv.org/4.1.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
## https://github.com/xn2333/OpenCV/blob/master/Seminar_Image_Processing_in_Python.ipynb
## https://colab.research.google.com/drive/1RWGmqoEQdeyh5TssoGtsXsFk8hbLGtWp#scrollTo=WtlNW7sC6xhT
## https://colab.research.google.com/drive/1iepzNI_lF1hQ9MLruwnmuHiB3YHNhwb9
## https://github.com/MeAmarP/opencv_snipps

class dataAugmentation(object): 
    def __init__(self, 
                 data_path, 
                 angle, 
                 arith_op, 
                 para_val, 
                 logical_op, 
                 func_filter, 
                 kernel_len, 
                 src_points, 
                 dst_points, 
                 orientation, 
                 x_scale, 
                 y_scale, 
                 gamma_val): 
        self.data_path = data_path 
        self.angle = angle 
        self.arith_op = arith_op 
        self.para_val = para_val 
        self.logical_op = logical_op 
        self.func_filter = func_filter 
        self.kernel_len = kernel_len 
        self.src_points = src_points
        self.dst_points = dst_points
        self.gamma_val=gamma_val 
        self.orientation=orientation 
        self.x_scale=x_scale
        self.y_scale=y_scale
        self.train = np.load(self.data_path, encoding='latin1', allow_pickle=True) 
    
    ## flip/rotate 
    # loop through self.angles 
    def rotate_img(self, img): 
        img = img.astype(np.uint8) 
        scaleFactor = 1
        rows, cols = img.shape 
        imgCenter = (cols-1)/2.0, (rows-1)/2.0 
        rotateMat = cv2.getRotationMatrix2D(imgCenter, self.angle, scaleFactor) 
        out_img = cv2.warpAffine(img, rotateMat, (cols, rows)) 
        return out_img 
    ## Arithmatic Operations 
    def arithops_on_img(self, img):
        img = img.astype(np.uint8) 
        # arith_op: + - * /
        if self.arith_op == 'add':
            s_img = (img + self.para_val)/np.max(img + self.para_val)*255.
        elif self.arith_op == 'subtract':
            s_img = img - self.para_val
        elif self.arith_op == 'multiply':
            s_img = img * self.para_val
        elif self.arith_op == 'divide':
            #Use Floor Divison operator
            s_img = img // self.para_val 
        return s_img 
    ## ligical/Bitwise operations 
    def logicalops_on_img(self, img): 
        img = img.astype(np.uint8) 
        h, w = img.shape 
        # LogicalOps = ['and','or','xor','not'] 
        """
        mask_img = np.concatenate((np.zeros(shape=[h,(w//3)],dtype=np.uint8),
                              np.ones(shape=[h,(w//3)+1],dtype=np.uint8)*64,
                              np.ones(shape=[h,(w//3)],dtype=np.uint8)*127),
                              axis=1) 
        """
        m = np.repeat([0, 0, 0, 1, 5, 0, 0, 0, 0, 1, 5, 10, 10, 30], [56])
        # noise
        np.random.shuffle(m)
        mask_img = m.reshape((h, w)) 
        if self.logical_op == 'and':
            #print('Performed Bitwise AND')
            #result_img = cv2.bitwise_and(img,mask_img)    
            result_img = (img + mask_img)/np.max(img + mask_img)*255.   
        elif self.logical_op == 'or':
            print('Performed Bitwise OR')
            result_img = cv2.bitwise_or(img,mask_img)
        elif self.logical_op == 'xor':
            print('Performed Bitwise XOR')
            result_img = cv2.bitwise_xor(img,mask_img)
        elif self.logical_op == 'not':
            print('Performed Bitwise NOT')
            result_img = cv2.bitwise_not(img)  
        return result_img 
    ## Enhancement 
    def imgLogTransform(self, img): 
        img = img.astype(np.uint8) 
        hist_img = cv2.calcHist([img],[0],None,[256],[0,256])
        img_log = (np.log2(img+1)/(np.log2(1+np.max(img))))*255
        img_log = np.array(img_log,dtype=np.uint8)
        hist_img_log = cv2.calcHist([img_log],[0],None,[256],[0,256]) 
        return hist_img 
    def imgGammaTransform(self, img ): 
        img = img.astype(np.uint8) 
        hist_img = cv2.calcHist([img],[0],None,[256],[0,256]) 
        gamma_corrected = np.array(255*(img / 255) ** self.gamma_val, dtype = 'uint8')
        hist_img_gamma = cv2.calcHist([gamma_corrected],[0],None,[256],[0,256]) 
        return hist_img_gamma 
    ## Contrast Stretching 
    # to decide whether to use
    def applyContrastStretching(self, img): 
        img = img.astype(np.uint8)  
        minmax_img = np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
        
        minmax_img[:,:] = 255*(img[:,:]-np.min(img)) / (np.max(img)-np.min(img))
        hist_img = cv2.calcHist([img],[0],None,[256],[0,256])
        hist_contrast_stretch = cv2.calcHist([minmax_img],[0],None,[256],[0,256]) 
        return hist_contrast_stretch 
    ## distort 
    # modified from https://medium.com/@schatty/image-augmentation-in-numpy-the-spell-is-simple-but-quite-unbreakable-e1af57bb50fd 
    def distort(self, img, func=np.sin):
        assert self.orientation[:3] in ['hor', 'ver'], "dist_orient should be 'horizontal'|'vertical'"
        assert func in [np.sin, np.cos], "supported functions are np.sin and np.cos"
        assert 0.00 <= self.x_scale <= 0.1, "x_scale should be in [0.0, 0.1]"
        assert 0 <= self.y_scale <= min(img.shape[0], img.shape[1]), "y_scale should be less then image size" 
        img = img.astype(np.uint8)
        img_dist = img.copy()
        
        def shift(x):
            return int(self.y_scale * func(np.pi * x * self.x_scale))
        for i in range(img.shape[self.orientation.startswith('ver')]):
            if self.orientation.startswith('ver'):
                img_dist[:, i] = np.roll(img[:, i], shift(i))
            else:
                img_dist[i, :] = np.roll(img[i, :], shift(i))
        """
        for c in range(1):
            for i in range(img.shape[orientation.startswith('ver')]):
                if orientation.startswith('ver'):
                    img_dist[:, i, c] = np.roll(img[:, i, c], shift(i))
                else:
                    img_dist[i, :, c] = np.roll(img[i, :, c], shift(i))
                    """
        return img_dist
    
    ## Spatial and linear filtering 
    # filters/blurr 
    # loop through filters and kernel lens
    def applyFilter(self, img): 
        img = img.astype(np.uint8)  
        if img is None:
            print('Unable to Read Image. Check you gave right path')
            return -1
        if self.func_filter == 'blur':
            img_fltr = cv2.blur(img,(self.kernel_len,self.kernel_len))
        if self.func_filter ==  'gaussian':
            img_fltr = cv2.GaussianBlur(img,(self.kernel_len,self.kernel_len),0)
        if self.func_filter == 'median':
            #kernal_len should be odd and greater than 1
            img_fltr = cv2.medianBlur(img,self.kernel_len)
        if self.func_filter == 'bilateral':
            img_fltr = cv2.bilateralFilter(img,self.kernel_len,self.kernel_len*2,self.kernel_len*2)
        if self.func_filter == 'arbitary':
            img_fltr =  cv2.filter2D(img,-1,self.kernel_len) 
        if self.func_filter == 'laplace':
            img_fltr = cv2.Laplacian(img,-1,3)
        if self.func_filter == 'motion-v':
            kernel_verti = np.zeros((self.kernel_len,self.kernel_len))
            kernel_verti[:, int((self.kernel_len - 1)/2)] = np.ones(self.kernel_len)
            kernel_verti /= self.kernel_len
            img_fltr = cv2.filter2D(img,-1,kernel_verti)
        if self.func_filter == 'motion-h':
            kernel_horiz = np.zeros((self.kernel_len,self.kernel_len))
            kernel_horiz[int((self.kernel_len - 1)/2),:] = np.ones(self.kernel_len)
            kernel_horiz /= self.kernel_len
            img_fltr = cv2.filter2D(img,-1,kernel_horiz)
        return img_fltr 
    ## Geometric Operations 
    # Affine 
    # provide src_points_list, dst_points_list and loop for each 
    # example: 
    # src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    # dst_points = np.float32([[0,0], [int(0.5*(cols-1)),0], [int(0.5*(cols-1)),rows-1]])

    def affineTransform(self, img): 
        img = img.astype(np.uint8)  
        cols, rows = img.shape 
        affine_matrix = cv2.getAffineTransform(self.src_points, self.dst_points)
        img_output = cv2.warpAffine(img, affine_matrix, (cols,rows)) 
        return img_output 
    ## 
    # order of augmentation workflow to be applied for each image
    # affine transformation --> rotate --> arithops
    # --> logicalops --> filters/blurr 
    # To decide: enhancement and contrast?? 
    def augment(self): 
        data = self.train
        data_out0 = self.train['arr_0'] 
        data_out1 = self.train['arr_1'] 
        for i in range(len(data['arr_0'])): 
            img = data['arr_0'][i].reshape((28,28)) 
            img_out0 = np.dstack((img, self.affineTransform(img)) ) 
            
            for l in range(img_out0.shape[2]): 
                img_out1 = np.dstack((img_out0, self.distort(img_out0[:, :, l])) )
            
            for l in range(img_out1.shape[2]): 
                img_out2 = np.dstack((img_out1, self.rotate_img(img_out1[:, :, l])))
                
            for l in range(img_out2.shape[2]): 
                img_out3 = np.dstack((img_out2, self.arithops_on_img(img_out2[:, :, l])))
                 
            for l in range(img_out3.shape[2]): 
                img_out4 = np.dstack((img_out3, self.logicalops_on_img(img_out3[:, :, l])))
                 
            for l in range(img_out4.shape[2]): 
                img_out5 = np.dstack((img_out4, self.applyFilter(img_out4[:, :, l]))) 
                
            # from second array
            for l in range(1, img_out5.shape[2]): 
                data_out0 = np.append(data_out0, img_out5[:, :, l].reshape((1, 784)), axis=0)
                data_out1 = np.append(data_out1, data['arr_1'][i]) 
        return data_out0, data_out1
        
                
## brief test of augmenter 
# rows, cols = 28, 28

## run it 



def applyAugmentation(data_augmented0, data_augmented1, data_path='/home/jx/Documents/IFT_6390/competition_2/train.npz'): 
    rows, cols = 28, 28 
    k =1
    for angle in [60, 120, 180]: 
        for arith_op in ['divide']: 
            for para_val in [2]: 
                for logical_op in ['and']: 
                    for func_filter in ['blur']: 
                        for kernel_len in [2]: 
                            for ori in ['ver']:
                                for x_param, y_param in zip([0.01, 0.03], [2, 6]):
                                    
                                        for dst_points in [np.float32([[0,0], [27,0], [0,21]]), 
                                                              np.float32([[0,0], [21,0], [0,27]]), 
                                                              np.float32([[6,6], [27,0], [0,27]]), 
                                                              np.float32([[0,0], [27,6], [0,27]]), 
                                                              np.float32([[0,0], [27,0], [6,27]]), 
                                                              np.float32([[4,4], [27,4], [4,27]]), 
                                                              np.float32([[0,7], [27,0], [0,27]])]: 
                                            for src_points in [np.float32([[0,0], [cols-1,0], [0,rows-1]])]: 
                                                dataAugmenter = dataAugmentation(
                                                                    data_path=data_path,
                                                                    angle=angle, 
                                                                    arith_op=arith_op, 
                                                                    para_val=para_val, 
                                                                    logical_op= logical_op, 
                                                                    func_filter=func_filter, 
                                                                    kernel_len = kernel_len, 
                                                                    src_points = src_points, 
                                                                    dst_points = dst_points, 
                                                                    orientation = ori, 
                                                                    x_scale = x_param, 
                                                                    y_scale = y_param, 
                                                                    gamma_val=0.8 ) 
                                                data_out0, data_out1 = dataAugmenter.augment() 
                                                data_augmented0 = np.append(data_augmented0, data_out0, axis=0) 
                                                data_augmented1 = np.append(data_augmented1, data_out1) 
                                                print (k) 
                                                k+=1
    return data_augmented0, data_augmented1 



"""
img2 = affineTransform(img)
vis = data_augmented0[33999].reshape((28, 28))
""" 

#############################################################################
# train/val split 

## prepare a val set for hyperparameter tuning 
from sklearn.model_selection import train_test_split  

X_train, X_val, y_train, y_val = train_test_split(train['arr_0'], 
                                                  train['arr_1'], 
                                                  test_size=0.3, 
                                                  shuffle=True, random_state=2) 

np.savez('/'.join([wk_dir, "partial_train_70.npz"]), X_train, y_train) 
partial_train = np.load('/'.join([wk_dir, "partial_train_70.npz"]), encoding='latin1', allow_pickle=True) 



partial_train_aug0, partial_train_aug1 = applyAugmentation(partial_train['arr_0'], partial_train['arr_1'], 
                                                     data_path='/home/jx/Documents/IFT_6390/competition_2/partial_train_70.npz') 
np.savez('/'.join([wk_dir, "partial_train_70_augmented_20201214_v2.npz"]), partial_train_aug0, partial_train_aug1) 

# validation set
np.savez('/'.join([wk_dir, "partial_val_30.npz"]), X_val, y_val) 
# Note no augmentation on the validation set. It provides a close estimate of score on the test set (according to kaggle submissions), 
# without augmentation. 




#############################################################################
# fit models and evaluate

######################################
## CNN 
# the CNN architecture was adapted from the following source: 
# https://github.com/ck090/Google_Quick_Draw 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

data_format = "channels_last" 
K.set_image_data_format(data_format)
K.image_data_format()




################### 
# redefine if needed
# 
partial_train = np.load('/'.join([wk_dir, "partial_train_70_augmented_20201214_v2.npz"]), encoding='latin1', allow_pickle=True) 
# validation set not augmented
partial_val = np.load('/'.join([wk_dir, "partial_val_30.npz"]), encoding='latin1', allow_pickle=True) 



X_train, y_train,X_val,  y_val = partial_train['arr_0']/255., partial_train['arr_1'], partial_val['arr_0']/255., partial_val['arr_1'] 

# one hot encode outputs
y_train_cnn = np_utils.to_categorical(y_train)
y_val_cnn = np_utils.to_categorical(y_val)
num_classes = y_val_cnn.shape[1]

# define the CNN model
def cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# reshape to be [samples][pixels][width][height]
X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_val_cnn = X_val.reshape(X_val.shape[0], 28, 28, 1).astype('float32')


# build the model
model_cnn = cnn_model()
# Fit the model
model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_val_cnn, y_val_cnn), epochs=30, batch_size=200) 

"""
Epoch 1/30
1549/1549 [==============================] - 40s 26ms/step - loss: 1.2054 - accuracy: 0.5321 - val_loss: 0.7173 - val_accuracy: 0.7578
Epoch 2/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.8470 - accuracy: 0.6876 - val_loss: 0.7837 - val_accuracy: 0.7511
Epoch 3/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.7243 - accuracy: 0.7358 - val_loss: 0.7531 - val_accuracy: 0.7889
Epoch 4/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.6549 - accuracy: 0.7619 - val_loss: 0.7069 - val_accuracy: 0.7889
Epoch 5/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.6054 - accuracy: 0.7801 - val_loss: 0.7818 - val_accuracy: 0.7800
Epoch 6/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.5730 - accuracy: 0.7926 - val_loss: 0.7634 - val_accuracy: 0.8133
Epoch 7/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.5454 - accuracy: 0.8021 - val_loss: 0.7744 - val_accuracy: 0.7844
Epoch 8/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.5250 - accuracy: 0.8094 - val_loss: 0.7979 - val_accuracy: 0.8022
Epoch 9/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.5099 - accuracy: 0.8149 - val_loss: 0.8770 - val_accuracy: 0.7800
Epoch 10/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.4941 - accuracy: 0.8210 - val_loss: 0.8604 - val_accuracy: 0.8022
Epoch 11/30
1549/1549 [==============================] - 41s 27ms/step - loss: 0.4804 - accuracy: 0.8253 - val_loss: 0.8705 - val_accuracy: 0.7844
Epoch 12/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.4730 - accuracy: 0.8282 - val_loss: 0.8860 - val_accuracy: 0.7933
Epoch 13/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.4634 - accuracy: 0.8326 - val_loss: 0.9091 - val_accuracy: 0.8067
Epoch 14/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.4577 - accuracy: 0.8340 - val_loss: 0.9599 - val_accuracy: 0.7778
Epoch 15/30
1549/1549 [==============================] - 45s 29ms/step - loss: 0.4491 - accuracy: 0.8376 - val_loss: 1.0555 - val_accuracy: 0.7778
Epoch 16/30
1549/1549 [==============================] - 47s 30ms/step - loss: 0.4441 - accuracy: 0.8385 - val_loss: 0.9850 - val_accuracy: 0.7978
Epoch 17/30
1549/1549 [==============================] - 43s 28ms/step - loss: 0.4397 - accuracy: 0.8404 - val_loss: 1.0237 - val_accuracy: 0.7889
Epoch 18/30
1549/1549 [==============================] - 39s 25ms/step - loss: 0.4314 - accuracy: 0.8438 - val_loss: 1.0086 - val_accuracy: 0.7867
Epoch 19/30
1549/1549 [==============================] - 42s 27ms/step - loss: 0.4272 - accuracy: 0.8453 - val_loss: 1.0988 - val_accuracy: 0.7933
Epoch 20/30
1549/1549 [==============================] - 37s 24ms/step - loss: 0.4246 - accuracy: 0.8466 - val_loss: 1.0233 - val_accuracy: 0.8089
Epoch 21/30
1549/1549 [==============================] - 36s 23ms/step - loss: 0.4195 - accuracy: 0.8483 - val_loss: 1.0408 - val_accuracy: 0.7978
Epoch 22/30
1549/1549 [==============================] - 36s 23ms/step - loss: 0.4148 - accuracy: 0.8495 - val_loss: 1.0655 - val_accuracy: 0.8022
Epoch 23/30
1549/1549 [==============================] - 36s 23ms/step - loss: 0.4112 - accuracy: 0.8513 - val_loss: 1.0599 - val_accuracy: 0.7956
Epoch 24/30
1549/1549 [==============================] - 36s 23ms/step - loss: 0.4077 - accuracy: 0.8524 - val_loss: 1.0887 - val_accuracy: 0.7844
Epoch 25/30
1549/1549 [==============================] - 36s 23ms/step - loss: 0.4040 - accuracy: 0.8536 - val_loss: 1.1198 - val_accuracy: 0.8000
Epoch 26/30
1549/1549 [==============================] - 36s 23ms/step - loss: 0.4002 - accuracy: 0.8549 - val_loss: 1.0642 - val_accuracy: 0.7844
Epoch 27/30
1549/1549 [==============================] - 36s 23ms/step - loss: 0.3971 - accuracy: 0.8565 - val_loss: 1.0754 - val_accuracy: 0.8022
Epoch 28/30
1549/1549 [==============================] - 36s 23ms/step - loss: 0.3950 - accuracy: 0.8573 - val_loss: 1.1064 - val_accuracy: 0.7956
Epoch 29/30
1549/1549 [==============================] - 36s 23ms/step - loss: 0.3941 - accuracy: 0.8572 - val_loss: 1.1201 - val_accuracy: 0.8044
Epoch 30/30
1549/1549 [==============================] - 36s 23ms/step - loss: 0.3899 - accuracy: 0.8594 - val_loss: 1.1757 - val_accuracy: 0.8022
Out[1]: <tensorflow.python.keras.callbacks.History at 0x7fdd75e961c0>

"""
# Final evaluation of the model
scores = model_cnn.evaluate(X_val_cnn, y_val_cnn, verbose=0)

print('Final CNN accuracy: ', scores[1])




###################
# apply to all 

# This will generate a lot of images, resulting a npz file of about 2.8 GBs
data_augmented0 = train['arr_0'] 
data_augmented1 = train['arr_1'] 
data_augmented0, data_augmented1 = applyAugmentation(data_augmented0, data_augmented1, 
                                                     data_path='/'.join([wk_dir, "train.npz"]))

np.savez('/'.join([wk_dir, "train_augmented_20201214_v2.npz"]), data_augmented0, data_augmented1)
# 


train_augmented = np.load('/'.join([wk_dir, "train_augmented_20201214_v2.npz"]), encoding='latin1', allow_pickle=True) 
test = np.load('/'.join([wk_dir, "test.npz"]), encoding='latin1', allow_pickle=True) 


X_train_all, y_train_all = train_augmented['arr_0']/255., train_augmented['arr_1'] 
X_test = test['arr_0']/255. 

# one hot encode outputs
y_train_all_cnn = np_utils.to_categorical(y_train_all)



# reshape to be [samples][pixels][width][height]
X_train_all_cnn = X_train_all.reshape(X_train_all.shape[0], 28, 28, 1).astype('float32')
X_test_all_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

model_cnn_all = cnn_model()
# Fit the model
model_cnn_all.fit(X_train_all_cnn, y_train_all_cnn, epochs=8, batch_size=200)

y_test_pred_cnn = model_cnn_all.predict_classes(X_test_all_cnn, verbose=0) 

test_out = pd.DataFrame(
    {'Id': [i for i in range(y_test_pred_cnn.shape[0])], 
     'Category': y_test_pred_cnn}
    ) 

# 

test_out.to_csv('/'.join([wk_dir, 'result/test_predictions_20201214_v2_8epochs.csv']), index = False) # 8 epochs
# kaggle score: 
# private 0.77852
# public 0.77111



#############################################################################
# some investigations on other CNN architectures
# 1. A smaller kernel size on the first conv2d. This is one of many hyperparameter tunings. 
# 2. A test on a deeper CNN architecture
#############################################################################
#############################################################################
# 1 

# reload if needed
partial_train = np.load('/'.join([wk_dir, "partial_train_70_augmented_20201214_v2.npz"]), encoding='latin1', allow_pickle=True) 
# validation set not augmented
partial_val = np.load('/'.join([wk_dir, "partial_val_30.npz"]), encoding='latin1', allow_pickle=True) 



X_train, y_train,X_val,  y_val = partial_train['arr_0']/255., partial_train['arr_1'], partial_val['arr_0']/255., partial_val['arr_1'] 

# one hot encode outputs
y_train_cnn = np_utils.to_categorical(y_train)
y_val_cnn = np_utils.to_categorical(y_val)
num_classes = y_val_cnn.shape[1]
# define the CNN model
def cnn_model_2():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), padding="same", activation='relu')) 
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




# reshape to be [samples][pixels][width][height]
X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_val_cnn = X_val.reshape(X_val.shape[0], 28, 28, 1).astype('float32')


# build the model
model_cnn_2 = cnn_model_2()
# Fit the model
model_cnn_2.fit(X_train_cnn, y_train_cnn, validation_data=(X_val_cnn, y_val_cnn), epochs=30, batch_size=200) 


"""
1549/1549 [==============================] - 50s 32ms/step - loss: 1.2402 - accuracy: 0.5154 - val_loss: 0.7890 - val_accuracy: 0.7489
Epoch 2/30
1549/1549 [==============================] - 48s 31ms/step - loss: 0.8708 - accuracy: 0.6782 - val_loss: 0.8118 - val_accuracy: 0.7667
Epoch 3/30
1549/1549 [==============================] - 48s 31ms/step - loss: 0.7426 - accuracy: 0.7282 - val_loss: 0.8355 - val_accuracy: 0.7800
Epoch 4/30
1549/1549 [==============================] - 48s 31ms/step - loss: 0.6668 - accuracy: 0.7569 - val_loss: 0.9220 - val_accuracy: 0.7644
Epoch 5/30
1549/1549 [==============================] - 48s 31ms/step - loss: 0.6169 - accuracy: 0.7755 - val_loss: 0.9935 - val_accuracy: 0.7644
Epoch 6/30
1549/1549 [==============================] - 48s 31ms/step - loss: 0.5808 - accuracy: 0.7891 - val_loss: 0.9403 - val_accuracy: 0.7711
Epoch 7/30
1549/1549 [==============================] - 48s 31ms/step - loss: 0.5536 - accuracy: 0.7994 - val_loss: 0.9652 - val_accuracy: 0.7600
Epoch 8/30
1549/1549 [==============================] - 48s 31ms/step - loss: 0.5329 - accuracy: 0.8065 - val_loss: 0.9911 - val_accuracy: 0.7578
Epoch 9/30
1549/1549 [==============================] - 48s 31ms/step - loss: 0.5136 - accuracy: 0.8134 - val_loss: 1.0657 - val_accuracy: 0.7467
Epoch 10/30
1549/1549 [==============================] - 48s 31ms/step - loss: 0.4972 - accuracy: 0.8196 - val_loss: 1.1910 - val_accuracy: 0.7378
Epoch 11/30
1425/1549 [==========================>...] - ETA: 3s - loss: 0.4853 - accuracy: 0.8238 Traceback (most recent call last):

"""
## model fitting killed half-way



#############################################################################
# 2 

#############################################################################
# inspired by https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as K

data_format = "channels_last" 
K.set_image_data_format(data_format)
K.image_data_format()

# 
partial_train = np.load('/'.join([wk_dir, "partial_train_70_augmented_20201216.npz"]), encoding='latin1', allow_pickle=True) 
# validation set not augmented
partial_val = np.load('/'.join([wk_dir, "partial_val_30.npz"]), encoding='latin1', allow_pickle=True) 



X_train, y_train,X_val,  y_val = partial_train['arr_0']/255., partial_train['arr_1'], partial_val['arr_0']/255., partial_val['arr_1'] 

# one hot encode outputs
y_train_cnn = np_utils.to_categorical(y_train)
y_val_cnn = np_utils.to_categorical(y_val)
    
## VGG-16
def vgg16_modified_model():
    # create model
    model = Sequential()
    model.add(Conv2D(input_shape=(28, 28, 1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    """
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    """
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2))) 
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=6, activation="softmax"))
    # Compile model
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model

model = vgg16_modified_model()

model.summary() 

# reshape to be [samples][pixels][width][height]
X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_val_cnn = X_val.reshape(X_val.shape[0], 28, 28, 1).astype('float32')


# Fit the model
model.fit(X_train_cnn, y_train_cnn, validation_data=(X_val_cnn, y_val_cnn), epochs=20, batch_size=200) 

"""
Epoch 1/20
2274/2274 [==============================] - 1883s 828ms/step - loss: 0.6715 - accuracy: 0.7422 - val_loss: 0.8932 - val_accuracy: 0.8422
Epoch 2/20
2274/2274 [==============================] - 1873s 824ms/step - loss: 0.2052 - accuracy: 0.9282 - val_loss: 1.3569 - val_accuracy: 0.8000
Epoch 3/20
2274/2274 [==============================] - 1873s 824ms/step - loss: 0.1156 - accuracy: 0.9602 - val_loss: 1.7231 - val_accuracy: 0.8111
Epoch 4/20
2274/2274 [==============================] - 1874s 824ms/step - loss: 0.0842 - accuracy: 0.9715 - val_loss: 1.5521 - val_accuracy: 0.7844
Epoch 5/20
2274/2274 [==============================] - 1872s 823ms/step - loss: 0.0667 - accuracy: 0.9779 - val_loss: 1.8934 - val_accuracy: 0.8111
Epoch 6/20
2274/2274 [==============================] - 1872s 823ms/step - loss: 0.0585 - accuracy: 0.9810 - val_loss: 1.7461 - val_accuracy: 0.8156
Epoch 7/20
2274/2274 [==============================] - 1870s 822ms/step - loss: 0.0509 - accuracy: 0.9839 - val_loss: 1.5950 - val_accuracy: 0.8156
Epoch 8/20
2274/2274 [==============================] - 1871s 823ms/step - loss: 0.0481 - accuracy: 0.9847 - val_loss: 2.3330 - val_accuracy: 0.8333
Epoch 9/20
2274/2274 [==============================] - 1870s 823ms/step - loss: 0.0429 - accuracy: 0.9865 - val_loss: 2.4125 - val_accuracy: 0.8089
Epoch 10/20
2274/2274 [==============================] - 1874s 824ms/step - loss: 0.0411 - accuracy: 0.9875 - val_loss: 2.1231 - val_accuracy: 0.8156
Epoch 11/20
2274/2274 [==============================] - 1871s 823ms/step - loss: 0.0384 - accuracy: 0.9886 - val_loss: 2.3754 - val_accuracy: 0.8222
Epoch 12/20
2274/2274 [==============================] - 1871s 823ms/step - loss: 0.0347 - accuracy: 0.9896 - val_loss: 2.6235 - val_accuracy: 0.8022
Epoch 13/20
2274/2274 [==============================] - 1872s 823ms/step - loss: 0.0347 - accuracy: 0.9899 - val_loss: 2.1818 - val_accuracy: 0.8200
Epoch 14/20
2274/2274 [==============================] - 1871s 823ms/step - loss: 0.0354 - accuracy: 0.9899 - val_loss: 2.5202 - val_accuracy: 0.8111
Epoch 15/20
2274/2274 [==============================] - 1870s 822ms/step - loss: 0.0303 - accuracy: 0.9914 - val_loss: 2.5542 - val_accuracy: 0.8200
Epoch 16/20
2274/2274 [==============================] - 1870s 822ms/step - loss: 0.0337 - accuracy: 0.9906 - val_loss: 2.9740 - val_accuracy: 0.8200
Epoch 17/20
2274/2274 [==============================] - 1873s 824ms/step - loss: 0.0306 - accuracy: 0.9915 - val_loss: 2.3661 - val_accuracy: 0.8333
Epoch 18/20
2274/2274 [==============================] - 1871s 823ms/step - loss: 0.0340 - accuracy: 0.9909 - val_loss: 2.0238 - val_accuracy: 0.7778
Epoch 19/20
2274/2274 [==============================] - 1871s 823ms/step - loss: 0.0270 - accuracy: 0.9925 - val_loss: 2.5959 - val_accuracy: 0.7711
Epoch 20/20
2274/2274 [==============================] - 1871s 823ms/step - loss: 0.0316 - accuracy: 0.9918 - val_loss: 2.2649 - val_accuracy: 0.8267
"""































