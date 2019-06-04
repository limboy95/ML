#!/usr/bin/env python
# coding: utf-8

# In[18]:


#------ Import and to 2D
import numpy
import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPooling1D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

written_train =numpy.load("written_train(1).npy")
spoken_train =numpy.load("spoken_train(1).npy")
match_train = numpy.load("match_train(1).npy")
feature_amount = spoken_train[0].shape[1]
print("shape written:", written_train.shape)
print("shape spoken", spoken_train.shape)
print("shape spoken indiv:", spoken_train[4].shape)
print(feature_amount)
print(match_train.shape)
writtenrow, writtencol = written_train.shape


# In[19]:



maxs = []
for i in range(0, len(spoken_train)):
    maxs.append(spoken_train[i].shape[0])
maxlen_spoken_train = max(maxs)

from keras.preprocessing.sequence import pad_sequences

spoken_train_3d= pad_sequences(spoken_train, maxlen= max(maxs))
print(spoken_train_3d.shape)

spoken_train_2d =spoken_train_3d.reshape(writtenrow, maxlen_spoken_train*feature_amount)

spoken_train_2d 



# In[20]:


#Standardscaler


SS_scaler = StandardScaler()
spoken_train_SSscaler = SS_scaler.fit_transform(spoken_train_2d)
written_train_SSscaler = SS_scaler.fit_transform(written_train)


#When using standard scaler
X_values = numpy.hstack([spoken_train_SSscaler, written_train_SSscaler])
Xvalrow, Xvalcol = X_values.shape


# In[21]:


#split for extra check
#X_values, X_test, match_train, y_val = train_test_split(X_values, match_train, test_size=1/3, random_state=999)


# In[22]:


#oversample

y_train = match_train
X_train = X_values

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==0)))
print("Before OverSampling, counts of label '2': {} ".format(sum(y_train==1)))

sm = SMOTE(random_state=4)
X_res, y_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of X: {}'.format(X_res.shape))
print('After OverSampling, the shape of y: {} \n'.format(y_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_res==0)))
print("After OverSampling, counts of label '2': {}".format(sum(y_res==1)))


X= numpy.array(X_res)
y= numpy.array(y_res)

X_train=X
y_train=y
print(y_train.shape)

a, b = X_train.shape


# In[23]:


#-------MLP


model = Sequential()
model.add(Dense(640, input_dim=Xvalcol, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(320, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

hist = model.fit(X_train, y_train,
          epochs=20,
          batch_size=208,
          validation_split=0.2)


# In[24]:


#------Test data
written_test = numpy.load("written_test(1).npy")
spoken_test =  numpy.load("spoken_test(1).npy")

writtentestrow, writtentestcol = written_test.shape


# In[25]:



maxs = []
for i in range(0, len(spoken_test)):
    maxs.append(spoken_test[i].shape[0])
maxlen_spoken_test = max(maxs)

from keras.preprocessing.sequence import pad_sequences

spoken_test_3d= pad_sequences(spoken_test, maxlen= max(maxs))
print(spoken_test_3d.shape)

spoken_test_2d =spoken_test_3d.reshape(writtentestrow, maxlen_spoken_test*feature_amount)

spoken_test_2d


# In[26]:


#Standardscaler


SS_scaler = StandardScaler()
spoken_test_SSscaler = SS_scaler.fit_transform(spoken_test_2d)
written_test_SSscaler = SS_scaler.fit_transform(written_test)


#When using standard scaler
X_test = numpy.hstack([spoken_test_SSscaler, written_test_SSscaler])


# In[27]:


y_pred = model.predict_classes(X_test, verbose=1) 
print(y_pred)
print(1 in y_pred)
y_pred = y_pred.reshape(writtentestrow,)
print(y_pred)


# In[28]:


numpy.save("results", y_pred)

