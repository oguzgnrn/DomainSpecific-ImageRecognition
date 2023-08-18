import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from numpy import genfromtxt
import keras.backend as K
from keras.layers import Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, LeakyReLU, PReLU
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from time import time

size = 224
bs = 32
train_folder = 'db3/fold1/train/images'
test_folder = 'db3/fold1/test/images'
train_data_gen = ImageDataGenerator()
train_data = train_data_gen.flow_from_directory(train_folder, class_mode='categorical', target_size=(size, size),
                                                color_mode='rgb', batch_size=bs)
test_data_gen = ImageDataGenerator()
test_data = test_data_gen.flow_from_directory(test_folder, class_mode='categorical', target_size=(size, size),
                                              shuffle=False)

shape = train_data.image_shape
print("Eğitim görüntülerinin boyutu: ", shape)
k = train_data.num_classes
print("Toplam sınıf sayısı: ", k)
train_samples = train_data.samples
print("Toplam eğitim görüntüsü sayısı: ", train_samples)
test_samples = test_data.samples
print("Toplam test görüntüsü sayısı: ", test_samples)


def googlenet(input_shape, n_classes):
    def inception_block(x, f):
        t1 = Conv2D(f[0], 1, activation='relu')(x)

        t2 = Conv2D(f[1], 1, activation='relu')(x)
        t2 = Conv2D(f[2], 3, padding='same', activation='relu')(t2)

        t3 = Conv2D(f[3], 1, activation='relu')(x)
        t3 = Conv2D(f[4], 5, padding='same', activation='relu')(t3)

        t4 = MaxPool2D(3, 1, padding='same')(x)
        t4 = Conv2D(f[5], 1, activation='relu')(t4)

        output = Concatenate()([t1, t2, t3, t4])
        return output

    input = Input(input_shape)

    x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = Conv2D(64, 1, activation='relu')(x)
    x = Conv2D(192, 3, padding='same', activation='relu')(x)
    x = MaxPool2D(3, strides=2)(x)

    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    x = AvgPool2D(7, strides=1)(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model


model = googlenet(shape, a)
model.summary()
optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001)  # Adamax optimizerı
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
ep = 50
spe = train_samples / bs
ts = test_samples / bs
r = model.fit(train_data, validation_data=test_data, steps_per_epoch=spe, validation_steps=ts, epochs=ep)

model.evaluate(test_data)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

pred = model.predict(test_data).argmax(axis=1)
labels = list(train_data.class_indices.keys())
np.savetxt("g1.csv", pred)
my_data = genfromtxt("g1.csv")
np.savetxt("g1_test.csv", test_data.classes)
my_data_test = genfromtxt("g1_test.csv")
cm = confusion_matrix(test_data.classes, my_data)
plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels, cmap="Greens")  # cmap="BuPu"
plt.title('Confusion Matrix')
plt.show()

print(classification_report(test_data.classes, pred))
