# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 21:58:18 2018

@author: Lenovo
"""


import pandas as pd
import numpy as np

k=5;
g=str(k)
print(g,type(g))
a=np.array([5,8,1,4]).reshape(4,1)
print((a*a).sum())
print(a.max())
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
 index=['Ohio', 'Colorado', 'Utah', 'New York'],
 columns=['one', 'two', 'three', 'four'])

frame = pd.DataFrame(data)
#frame.drop(['one'],axis=1,inplace=True)

print(frame.iloc[:3,[3,0,1]])

df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
  columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
  columns=list('abcde'))

df1=df1.add(df2,fill_value=0)
f= lambda x: x.max()+x.min()

print(df1)
print(df1.cumsum(axis=1))
print(df1.skew(axis=1))

print(f(a))

