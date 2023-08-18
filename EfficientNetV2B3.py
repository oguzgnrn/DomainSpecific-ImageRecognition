from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet import ResNet, ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.resnet_v2 import ResNet50V2
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
from keras.applications.efficientnet_v2 import EfficientNetV2B3
from keras.layers import Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy

main_path = "C:/Users/gnrno/PycharmProjects/pythonBase/"
fold_name=['fold1','fold2','fold3','fold4','fold5',]
learning_rate= [0.000001]
IMAGE_SIZE = [224, 224]
confusion_labels=[]
epoch = 100
for lr in learning_rate:
  for fn in fold_name:
    train_path = main_path + fn +'/train/images'
    valid_path = main_path + fn +'/test/images'
    for name in os.listdir(train_path):
      confusion_labels.append(name)
    inception = tf.keras.applications.EfficientNetV2B3(
       include_top=False,
       weights="imagenet",
       input_shape=(224,224,3),
       classes=30,
       classifier_activation="softmax")
    inception = EfficientNetV2B3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False,
                                 include_preprocessing=False)


    for layer in inception.layers:
      layer.trainable = False
    x = Flatten()(inception.output)
    folders = glob(main_path + fn+'/train/images/*')
    prediction = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=inception.input, outputs=prediction)
    # Store the fully connected layers

    model.summary()
    # optz=optimizer.first.value
    opt = tf.keras.optimizers.Adamax(learning_rate = lr)
    ooptimizer = "Adamax"
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    train_datagen = ImageDataGenerator(rescale = 1./255)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode='categorical',
                                                 )
    test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode='categorical',

                                            )
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

    model.save(main_path +'PosetsizEfficientNetV2B3Results/_EfficientNetV2B3_'+str(epoch)+'_'+str(ooptimizer) +'_'+str(lr)+'_'+fn+'')


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
    plt.savefig(main_path +'PosetsizEfficientNetV2B3Results/_confusion_matrix_EfficientNetV2B3_'+str(epoch)+'_'+str(ooptimizer) +'_'+str(lr)+'_'+fn+'_'+str(time)+'dk_'+str(accuracy)+'%.png',format="png",dpi=300,bbox_inches='tight')

    #plt.show()
# plt.close(fig)
