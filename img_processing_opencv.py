# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 20:21:47 2019

@author: Lenovo
"""
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from skimage import io, filters
import cv2
from sklearn.decomposition import PCA

def main():
    print(cv2.__version__)
    
if __name__=='__main__':
    main()

a = 0
b = -1
l = []
m = []
split = []
sum = 0
print('b')



for path, subdirs, files in os.walk('C:/Users/Lenovo/Desktop/data/English/Img/GoodImg/Bmp/'):
    for filename in files:
        f = path + '/' + filename
        img = cv2.imread(f)
        img = cv2.resize(img,(100,100))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        img = filters.sobel(gray_image)
#        fig = plt.figure()
#        ax1 = fig.add_subplot(1,2,1)
#        ax1.imshow(gray_image,cmap='gray')
#        
#        ax2 = fig.add_subplot(1,2,2)
#        ax2.imshow(img,cmap='gray')
#        plt.show()
        l.append(gray_image.reshape(1,-1)[0])
        m.append(a)
        b += 1
        
    print(a)
    split.append(b)
    a += 1
    
l = np.array(l)
m = np.array(m)


print('m',m)
print('s',split)

a1 = 0
l1 = []
m1 = []
print('b')

for path, subdirs, files in os.walk('C:/Users/Lenovo/Desktop/data/English/Img/BadImag/Bmp/'):
    for filename in files:
        f = path + '/' + filename
        img = cv2.imread(f)
        img = cv2.resize(img,(100,100))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        img = filters.sobel(gray_image)1
#        fig = plt.figure()
#        ax1 = fig.add_subplot(1,2,1)
#        ax1.imshow(gray_image,cmap='gray')
#        
#        ax2 = fig.add_subplot(1,2,2)
#        ax2.imshow(img,cmap='gray')
#        plt.show()
        l1.append(gray_image.reshape(1,-1)[0])
        m1.append(a1)
        
    print(a1)
    a1 += 1    

m1 = np.array(m1)
l1 = np.array(l1)




        






# k = []
#for i in range(a[0].shape[0]):
#    b = {}
#    for j in range(5):
#        if  a[j][i] not in b:
#            b[a[j][i]]=0
#        else:
#            b[a[j][i]]+=1
#    print(b)
#    max=sorted(b.values(),reverse=True)[0]
#    print(max)
#    for key in b.keys():
#        if b[key]==max:
#            k.append(key)
#            break
    
    


