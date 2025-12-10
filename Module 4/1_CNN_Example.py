
import pandas as pd
import numpy as np
import shutil
import os

import tensorflow as tf


class MyCLRuleMonitor(tf.keras.callbacks.Callback):
  def __init__(self, CL):
    super(MyCLRuleMonitor).__init__()
    self.CL = CL

  def on_epoch_end(self, epoch, logs=None):
    trainScore = logs["accuracy"]
    testScore = logs["val_accuracy"]

    if testScore > trainScore and testScore >= self.CL:
      self.model.stop_training = True

# Extract the Zip File ---- Step required in colab if zip file uploaded
zip_file_path = 'cats_and_dogs.zip'
extract_dir = 'cats_and_dogs'

# Create the directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Unpack the archive using Python's standard library
try:
    shutil.unpack_archive(zip_file_path, extract_dir)
    print(f"Successfully unpacked {zip_file_path} to {extract_dir}")
except Exception as e:
    print(f"An error occurred: {e}")

# Preprocessing --- Goal is to make the data compatible for CNN
# Tensorflow by default offers direct class to achieve preprocessing

'Steps for model creation:'
'1. Model architecting'
'2. Model compile'
'3. Model training'
'4. Prediction / Output'

# ImageGenerators
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

# Pass my image:

trainImageData = train_generator.flow_from_directory('cats_and_dogs/cats_and_dogs/train', # Path to training data
                                                     batch_size = 20,       # How many images to pass per tick
                                                     class_mode = 'binary', # binary -- Binary Classification | categorical -- Multiclass classification
                                                     target_size = (64,64)) # 

testImageData = test_generator.flow_from_directory('cats_and_dogs/cats_and_dogs/validation',
                                                   batch_size = 20,
                                                   class_mode = 'binary',
                                                   target_size = (64,64))

print(trainImageData.image_shape)

# Architect the NN:

# Creating the model:

model = tf.keras.Sequential()

# We will use two convolutional layers here:
# Convolutional layer = convolve + pooling (Optional)

# 1st convolutional layer:
        # Convolve:
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape = trainImageData.image_shape, padding= 'same'))
        # Padding
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

# 2nd Convolutional layer:
        # Convolve:
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
        # Padding
model.add(tf.keras.layers.MaxPooling2D(pool_size= (2,2)))

# Flatten:
model.add(tf.keras.layers.Flatten())

# FC layer: Fully Connected Layer | ANN:
model.add(tf.keras.layers.Dense(units = 512, activation='relu'))
model.add(tf.keras.layers.Dense(units = 128, activation='relu'))
model.add(tf.keras.layers.Dense(units = 256, activation='relu'))

# if class mode binary-- sigmoid | if class mode categorical -- softmax
model.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

print(model.summary())

# Compile the model:

model.compile(optimizer = 'adam',loss= 'binary_crossentropy', metrics =['accuracy'])

# Train the model:
# steps_per_epochs ===== applicable for train data
# validation_steps ===== applicable for testing data 
model.fit(trainImageData, 
          validation_data = testImageData, 
          epochs = 200, 
          steps_per_epoch = (len(trainImageData.filenames)//trainImageData.batch_size ),
          validation_steps= (len(testImageData.filenames)//testImageData.batch_size),
          callbacks= [MyCLRuleMonitor(0.7)])