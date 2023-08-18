import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.losses import categorical_crossentropy
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
# img_width = 227
# img_heigth = 227
# batch_size = 16
# TRAINING_DIR = "foldwlabel/newtrain/images"
# train_datagen = ImageDataGenerator(rescale=1 / 226.0)
# train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size=batch_size, class_mode="categorical", target_size=(img_heigth, img_width))
# VALIDATION_DIR = "foldwlabel/newtest/images"
# validation_datagen = ImageDataGenerator(rescale=1 / 226.0)
# validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, batch_size=batch_size,class_mode="categorical", target_size=(img_heigth, img_width))
# image_shape = (227,227,3)
# np.random.seed(29)
# model = Sequential()
# model.add(Conv2D(filters=96,input_shape=image_shape,kernel_size=(11,11),strides=(4,4),padding="valid"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="valid"))
# model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding="valid"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="valid"))
# model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid"))
# model.add(Activation("relu"))
# model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid"))
# model.add(Activation("relu"))
# model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="valid"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="valid"))
# model.add(Flatten())
# model.add(Dense(4096,input_shape=(227*227*3,)))
# model.add(Activation("relu"))
# model.add(Dropout(0.4))
# model.add(Dense(4096))
# model.add(Activation("relu"))
# model.add(Dropout(0.3))
# model.add(Dense(29))
# model.add(Activation("softmax"))
# model.summary()
# lr = 0.000001
# epoc = 50
# opt = keras.optimizers.SGD(learning_rate=lr)
# model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# history = model.fit_generator(train_generator, epochs=epoc, verbose=1, validation_data=validation_generator)
# _, acc = model.evaluate(validation_generator, steps=len(validation_generator), verbose=1)
# print('> %.3f' % (acc * 100.0))







img_width = 256
img_heigth = 256
batch_size = 16

TRAINING_DIR = "foldwlabel/newtrain/images"
train_datagen = ImageDataGenerator(rescale=1 / 255.0)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size=batch_size, class_mode="categorical",
                                                    target_size=(img_heigth, img_width))
VALIDATION_DIR = "foldwlabel/newtest/images"
validation_datagen = ImageDataGenerator(rescale=1 / 255.0)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, batch_size=batch_size,
                                                              class_mode="categorical",
                                                              target_size=(img_heigth, img_width))

model = Sequential([
    Conv2D(16, (3, 3), activation="relu", input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation="relu"),
    Conv2D(32, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    Conv2D(128, (3, 3), activation="relu"),
    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation="relu"),
    Conv2D(256, (3, 3), activation="relu"),
    Conv2D(256, (3, 3), activation="relu"),
    Conv2D(256, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(1024, activation="relu"),
    Dense(512, activation="relu"),
    Dense(256,activation="relu"),
    Dense(128, activation="relu"),
    Dense(29, activation="softmax"),
])
model.summary()
lr = 0.001
epoc = 100
opt = keras.optimizers.SGD(learning_rate=lr)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit_generator(train_generator, epochs=epoc, verbose=1, validation_data=validation_generator )
_, acc = model.evaluate(validation_generator, steps=len(validation_generator), verbose=1)
print('> %.3f' % (acc * 100.0))



import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.losses import categorical_crossentropy
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
# img_width = 227
# img_heigth = 227
# batch_size = 16
# TRAINING_DIR = "fold1/train/images"
# train_datagen = ImageDataGenerator(rescale=1 / 226.0)
# train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size=batch_size, class_mode="categorical", target_size=(img_heigth, img_width))
# VALIDATION_DIR = "fold1/test/images"
# validation_datagen = ImageDataGenerator(rescale=1 / 226.0)
# validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, batch_size=batch_size,class_mode="categorical", target_size=(img_heigth, img_width))
# image_shape = (227,227,3)
# model = Sequential()
# model.add(Conv2D(filters=96,input_shape=image_shape,kernel_size=(11,11),strides=(4,4),padding="valid"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="valid"))
# model.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding="valid"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="valid"))
# model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid"))
# model.add(Activation("relu"))
# model.add(Conv2D(filters=384,kernel_size=(3,3),strides=(1,1),padding="valid"))
# model.add(Activation("relu"))
# model.add(Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding="valid"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="valid"))
# model.add(Flatten())
# model.add(Dense(4096,input_shape=(227*227*3,)))
# model.add(Activation("relu"))
# model.add(Dropout(0.4))
# model.add(Dense(4096))
# model.add(Activation("relu"))
# model.add(Dropout(0.4))
# model.add(Dense(30))
# model.add(Activation("softmax"))
# model.summary()
# lr = 0.00001
# epoc = 50
# opt = keras.optimizers.SGD(learning_rate=lr)
# model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# history = model.fit_generator(train_generator, epochs=epoc, verbose=1, validation_data=validation_generator)
# _, acc = model.evaluate(validation_generator, steps=len(validation_generator), verbose=1)
#print('> %.3f' % (acc * 100.0))






img_width = 256

img_heigth = 256
batch_size = 16

TRAINING_DIR = "fold3/train/images"
train_datagen = ImageDataGenerator(rescale=1 / 255.0)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR, batch_size=batch_size, class_mode="categorical",
                                                    target_size=(img_heigth, img_width))
VALIDATION_DIR = "fold3/test/images"
validation_datagen = ImageDataGenerator(rescale=1 / 255.0)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, batch_size=batch_size,
                                                              class_mode="categorical",
                                                              target_size=(img_heigth, img_width))

model = Sequential([
    Conv2D(16, (3, 3), activation="relu", input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation="relu"),
    Conv2D(32, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    Conv2D(128, (3, 3), activation="relu"),
    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation="relu"),
    Conv2D(256, (3, 3), activation="relu"),
    Conv2D(256, (3, 3), activation="relu"),
    Conv2D(256, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(1024, activation="relu"),
    Dense(512, activation="relu"),
    Dense(256,activation="relu"),
    Dense(128, activation="relu"),
    Dense(30, activation="softmax"),
])
model.summary()
lr = 0.00001
epoc = 100
opt = keras.optimizers.SGD(learning_rate=lr)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit_generator(train_generator, epochs=epoc, verbose=1, validation_data=validation_generator )
_, acc = model.evaluate(validation_generator, steps=len(validation_generator), verbose=1)
print('> %.3f' % (acc * 100.0))




