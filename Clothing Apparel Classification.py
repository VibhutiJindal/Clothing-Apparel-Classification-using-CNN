#!/usr/bin/env python
# coding: utf-8

# ## CLOTHING APPAREL CLASSIFICATON

# In[1]:


#importing necessary libraries
import numpy as np
from keras.datasets import fashion_mnist


# ### LOADING THE DATA

# In[2]:


#loading fashion MNIST dataset
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()


# In[3]:


print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test.shape)
print('y_test:', y_test.shape)


# ### VIEWING SAMPLE IMAGES

# In[4]:


#importing visualization library
import matplotlib.pyplot as plt
import math
# Plotting 5 images, Subplot arugments represent nrows, ncols and index and  Color map is set to grey since image dataset is grayscale
images_to_display = 25
image_cells = math.ceil(math.sqrt(images_to_display))
plt.figure(figsize=(10,10))
for i in range(images_to_display):
    plt.subplot(image_cells, image_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.xlabel(y_train[i])
plt.show()


# ### RESHAPING THE DATA

# In[5]:


# Save image parameters to the constants 
(IMAGES, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train.shape
IMAGE_CHANNELS = 1
print('IMAGES: ', IMAGES);
print('IMAGE_WIDTH:', IMAGE_WIDTH);
print('IMAGE_HEIGHT:', IMAGE_HEIGHT);
print('IMAGE_CHANNELS:', IMAGE_CHANNELS);


# In[6]:


x_train= x_train.reshape(
    x_train.shape[0],
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS
)

x_test = x_test.reshape(
    x_test.shape[0],
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS
)


# In[7]:


print('x_train_with_chanels:', x_train.shape)
print('x_test_with_chanels:', x_test.shape)


# In[8]:


# Changing image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# In[9]:


# Normalizing the data by changing the image pixel range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255


# In[10]:


#  checking one row from the 0th image to see color chanel values after normalization.
x_train[0][18]


# ### CREATING THE MODEL

# In[11]:


#importing necessary keras specific libraries
from keras.utils import np_utils
import keras
from keras.models import  Sequential
from keras.layers import Dense, Dropout ,Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


# In[12]:


#performing one hot encoding
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
# Calculate the number of classes 
num_classes = y_test.shape[1]


# In[13]:


#creating cnn model
model=Sequential()
model.add(Conv2D(input_shape=(28,28,1),filters=32,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes,activation='softmax'))


# In[14]:


#compiling the model
import tensorflow as tf
model.compile(loss = 'categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

print(model.summary())


# ### TRANING THE MODEL

# In[16]:


# Setting Training Parameters like batch_size, epochs
epochs = 10
model_fitting = model.fit(x_train, y_train,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))


# In[17]:


plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(model_fitting.history['loss'], label='training set')
plt.legend()


# In[18]:


plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(model_fitting.history['accuracy'], label='training set')

plt.legend()


# ### EVALUATING ACCURACY AND LOSS

# In[19]:


score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[20]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# ### MAKING PREDICTIONS

# In[21]:


import pandas as pd
predictions = model.predict([x_test])


# In[22]:


print('predictions:', predictions.shape)


# In[23]:


# Predictions in form of one-hot vectors (arrays of probabilities).
pd.DataFrame(predictions)


# In[24]:


#  extract predictions with highest probabilites 
prediction_s = np.argmax(predictions, axis=1)
pd.DataFrame(prediction_s)


# In[35]:


print(prediction_s[0])
if prediction_s[0] == 0:
  print("T-shirt/top")
elif prediction_s[0] == 1:
  print("Trouser") 
elif prediction_s[0] == 2:
  print("Pullover") 
elif prediction_s[0] == 3:
  print("Dress") 
elif prediction_s[0] == 4:
  print("Coat") 
elif prediction_s[0] == 5:
  print("Sandal")
elif prediction_s[0] == 6:
  print("Shirt") 
elif prediction_s[0] == 7:
  print("Sneaker") 
elif prediction_s[0] == 8:
  print("Bag") 
else:
  print( "Ankle boot")


# In[26]:


plt.imshow(x_test[0].reshape((IMAGE_WIDTH, IMAGE_HEIGHT)), cmap=plt.get_cmap('gray'))
plt.show()


# In[27]:


print(y_test[0])


# In[31]:


y_test_modified=np.argmax(y_test, axis=1)


# In[34]:


import seaborn as sn
confusion_matrix = tf.math.confusion_matrix(y_test_modified, prediction_s)
plt.figure(figsize=(9,9))
sn.heatmap(
    confusion_matrix,
    annot=True,
    linewidths=.5,
    fmt="d",
    square=True,
)
plt.show()


# In[ ]:




