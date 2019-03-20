import os
import random
import numpy as np
import glob
import cv2 as cv
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPool2D, UpSampling2D, Dropout, Concatenate, LeakyReLU, Convolution2D, \
    BatchNormalization, Conv2DTranspose
from keras.optimizers import *
from keras import losses
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger
from keras_preprocessing.image import array_to_img


def myGenerator(X_path, Y_path, epoch, w, h, c, rand=0, ):
    while True:
        x = np.ndarray((epoch, w, h, c), dtype=np.uint8)
        y = np.ndarray((epoch, w, h, 1), dtype=np.uint8)
        label = np.ndarray((w, h, 1), dtype=np.uint8)
        imgg = np.ndarray((w, h, 1), dtype=np.uint8)
        imgs = glob.glob(X_path + "\\*.jpg")
        num = int(len(imgs) / epoch)

        if rand == 1:
            random.shuffle(imgs)

        for i in range(num):
            for k in range(epoch):
                n = k * epoch
                iname = imgs[n + i]
                label_name = Y_path + '\\' + iname[iname.rindex('\\') + 1:].split('.')[0] + ".jpg"
                imgg[:, :, 0] = cv.resize(cv.imread(iname, 0), (w, h))
                label[:, :, 0] = cv.resize(cv.imread(label_name, 0), (w, h))
                x[k] = imgg
                y[k] = label

            yield (x, y)


def get_mynet(row=0, col=0):
    inputs = Input((row, col, 1))
    conv1 = Conv2D(64, (9, 9), strides=1, padding='same', activation='relu')(inputs)
#    conv1 = Conv2D(64, (9, 1), strides=1, padding='same', activation='relu')(conv1)
    conv2 = Conv2D(32, 1, strides=1, padding='same', activation='relu')(conv1)
    conv3 = Conv2D(1, 5, strides=1, padding='same', activation=None)(conv2)

    model = Model(inputs=inputs, outputs=conv3)
    model.compile(optimizer=Adam(lr=0.0001), loss=losses.mean_squared_error, metrics=['accuracy'])
    return model


def train(row=0, col=0):
    model = get_mynet(row, col)
    print('got net')

    model_checkpoint = ModelCheckpoint("D:\\python_project\\test\\canshu.hdf5", monitor='val_acc', mode='auto',
                                       verbose=1, save_best_only=True)
    eStop = EarlyStopping(monitor='loss', patience=4, verbose=1, mode='auto')
    Logging = CSVLogger("D:\\python_project\\test\\log.csv", separator=',', append=False)
    print('Fitting model')
    ximage = myGenerator("D:\\python_project\\src\\train", "D:\\python_project\\src\\train_label", 1, row, col, 1,
                         rand=1)
    testimg = myGenerator("D:\\python_project\\src\\test", "D:\\python_project\\src\\test_label", 1, row, col, 1,
                          rand=1)
    model.fit_generator(ximage, steps_per_epoch=328, epochs=200, verbose=1, validation_data=testimg,
                        validation_steps=109, callbacks=[model_checkpoint, eStop, Logging])


def myGenerator_predict(X_path, epoch, w, h, c, rand=0, ):
    while True:
        x = np.ndarray((epoch, w, h, c), dtype=np.uint8)
        imgg = np.ndarray((w, h, 1), dtype=np.uint8)
        imgs = glob.glob(X_path + "\\*.jpg")
        num = int(len(imgs) / epoch)

        if rand == 1:
            random.shuffle(imgs)

        for i in range(num):
            for k in range(epoch):
                n = k * epoch
                iname = imgs[n + i]
                imgg[:, :, 0] = cv.resize(cv.imread(iname, 0), (w, h))
                x[k] = imgg

            yield x


def predict():
    model = load_model("D:\\python_project\\test\\canshu-0.95144.hdf5")
    ximagee = myGenerator_predict("D:\\python_project\\src\\test", 1, 1280, 1280, 1,
                                  rand=1)
    y = model.predict_generator(ximagee, steps=109, verbose=1)
    print('done')
    i = 0
    for img in y:
        img1 = array_to_img(img)
        img1.save("D:\\python_project\\xiaoguo\\%d.jpg" % i)
        i = i + 1
    print('save done')


if __name__ == '__main__':
    train(1280, 1280)
#    predict()
