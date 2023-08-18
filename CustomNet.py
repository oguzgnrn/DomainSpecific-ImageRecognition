from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import PIL
from glob import glob
import tensorflow as tf
import timeit
import pathlib
import enum
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


epoch = 100
main_pathe='/content/drive/MyDrive/ProjeNo-3215117/Prip_DataSet/'
main_path = "C:/Users/gnrno/PycharmProjects/pythonBase/"
fold_name=['fold1','fold2','fold3','fold4','fold5']
learning_rate=[0.00001]
IMAGE_SIZE = [224, 224]
confusion_labels=[]

for lr in learning_rate:
  for fn in fold_name:
    train_path = main_path + fn +'/train/images'
    valid_path = main_path + fn +'/test/images'
    for name in os.listdir(train_path):
      confusion_labels.append(name)
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(224, 224),
                                                     batch_size=16,
                                                     class_mode='categorical')
    test_set = test_datagen.flow_from_directory(valid_path,
                                                target_size=(224, 224),
                                                batch_size=16,
                                                class_mode='categorical')
    inception = Sequential([
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
        Dense(30, activation="softmax")])

    x = Flatten()(inception.output)
    folders = glob(main_path + fn+'/train/images/*')
    prediction = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=inception.input, outputs=prediction)
    model.summary()
    opt = tf.keras.optimizers.Adamax(learning_rate = lr)
    ooptimizer = "Adamax"
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

    start = timeit.default_timer()
    r = model.fit_generator(
      training_set,
      shuffle=False,
      validation_data=test_set,
      epochs=epoch,
      steps_per_epoch=len(training_set),
      validation_steps=len(test_set)
    )

    def get_path(fn,epoch,optmzr,lr):
        path2=pathlib.PurePath(fn,str(epoch),optmzr,str(lr))
        return (os.path.join('',path2))

    model.save(main_path +'ResNetResults/_resnet50_'+str(epoch)+'_'+str(ooptimizer) +'_'+str(lr)+'_'+fn+'')


    _,acc = model.evaluate(test_set, steps=len(test_set), verbose=1)
    accuracy=round((acc*100),2)
    print('> %.3f' % (acc * 100.0))
    stop = timeit.default_timer()
    time=int((stop-start)/60)
    print("Time:%d dk " % (time))
    y_pred=model.predict(training_set)
    print(y_pred.shape)


    def get_confusion_matrix(model, test_array):
      all_predictions = np.array([])
      all_labels = np.array([])
      for i in range(len(test_array)):
        training_set, test_set = test_array[i]
        predictions = model.predict(training_set)
        predictions = np.argmax(predictions, axis = 1)
        all_predictions = np.concatenate([all_predictions, predictions])
        labels = np.argmax(test_set, axis = 1)
        all_labels = np.concatenate([all_labels, labels])
      return tf.math.confusion_matrix(all_predictions, all_labels)
    conf=get_confusion_matrix(model,test_set)
    plt.clf()
    fig=plt.figure(figsize=(13,7))
    title=get_path(fn,epoch,ooptimizer,lr)
    sns.heatmap(conf,annot=True,cmap='Blues',annot_kws={'fontsize':13},xticklabels=confusion_labels, yticklabels=confusion_labels,)
    plt.ylabel('Gercek Değer')
    plt.xlabel('Tahmini Değer')
    plt.title(title)
    plt.savefig(main_path +'ResNetResults/_confusion_matrix_'+str(epoch)+'_'+str(ooptimizer) +'_'+str(lr)+'_'+fn+'_'+str(time)+'dk_'+str(accuracy)+'%.png',format="png",dpi=300,bbox_inches='tight')

    plt.show()
    plt.close(fig)

