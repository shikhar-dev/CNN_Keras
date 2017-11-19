# Importing Depedencies
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# Fixing Seed for reporductibility
seed = 7
np.random.seed(seed)

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshaping for use in CNN as [samples][height][weight][channels] channels = 1 for grey scale
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalizing intensity values from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# Convert Integers to ONE HOT ENCODED vector [samples][10]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Number of classes to classify
classes = y_train.shape[1]


# Defining Model
def baseline_model():
    # Using Sequential models
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), data_format='channels_last', input_shape=(28,28,1), activation='relu', strides=1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(classes,activation='softmax'))

    # Compiling Model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Building model
model = baseline_model()

# Training the model
model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),batch_size=200,verbose=2)

# Evaluating
score = model.evaluate(X_test,y_test,verbose=0)

# Printing error
print("CNN Error: %.2f%%" % (100-score[1]*100))

# Output for First test case
# Visulizing First test case:
plt.subplot(1,1,1)
plt.imshow(X_test[0],cmap=plt.get_cmap('grey'))
print 'Output :'
print model.predict(X_test[0])