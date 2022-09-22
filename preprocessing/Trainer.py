import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np


class Trainer:
    df = None
    train_flow, test_flow = None, None

    def __init__(self, Path: str):
        self.__loadDataset(Path)

    def __loadDataset(self, Path: str):
        self.df = pd.DataFrame()
        self.df = pd.read_csv(Path)
        print(self.df.head())
        print(self.df.describe())

    def createModel(self):
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for index, row in self.df.iterrows():
            k = row['pixels'].split(" ")
            if row['Usage'] == 'Training':
                X_train.append(np.array(k))
                y_train.append(row['emotion'])
            elif row['Usage'] == 'PublicTest':
                X_test.append(np.array(k))
                y_test.append(row['emotion'])
        X_train = np.array(X_train, dtype='uint8')
        y_train = np.array(y_train, dtype='uint8')
        X_test = np.array(X_test, dtype='uint8')
        y_test = np.array(y_test, dtype='uint8')
        X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
        X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

        y_train = to_categorical(y_train, num_classes=7)
        y_test = to_categorical(y_test, num_classes=7)

        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=10,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='nearest')

        testgen = ImageDataGenerator(
            rescale=1. / 255
        )
        datagen.fit(X_train)
        batch_size = 64
        self.train_flow = datagen.flow(X_train, y_train, batch_size=batch_size)
        self.test_flow = testgen.flow(X_test, y_test, batch_size=batch_size)
        model = self.modelGenerator()
        opt = Adam(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        num_epochs = 100
        history = model.fit_generator(self.train_flow,
                                      steps_per_epoch=len(X_train) / batch_size,
                                      epochs=num_epochs,
                                      verbose=2,
                                      validation_data=self.test_flow,
                                      validation_steps=len(X_test) / batch_size)
        return model

    def modelGenerator(self, input_shape=(48, 48, 1)):
        # first input model
        visible = Input(shape=input_shape, name='input')
        num_classes = 7
        # the 1-st block
        conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_1')(visible)
        conv1_1 = BatchNormalization()(conv1_1)
        conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_2')(conv1_1)
        conv1_2 = BatchNormalization()(conv1_2)
        pool1_1 = MaxPooling2D(pool_size=(2, 2), name='pool1_1')(conv1_2)
        drop1_1 = Dropout(0.3, name='drop1_1')(pool1_1)

        # the 2-nd block
        conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_1')(drop1_1)
        conv2_1 = BatchNormalization()(conv2_1)
        conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_2')(conv2_1)
        conv2_2 = BatchNormalization()(conv2_2)
        conv2_3 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_3')(conv2_2)
        conv2_2 = BatchNormalization()(conv2_3)
        pool2_1 = MaxPooling2D(pool_size=(2, 2), name='pool2_1')(conv2_3)
        drop2_1 = Dropout(0.3, name='drop2_1')(pool2_1)

        # the 3-rd block
        conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_1')(drop2_1)
        conv3_1 = BatchNormalization()(conv3_1)
        conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_2')(conv3_1)
        conv3_2 = BatchNormalization()(conv3_2)
        conv3_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_3')(conv3_2)
        conv3_3 = BatchNormalization()(conv3_3)
        conv3_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_4')(conv3_3)
        conv3_4 = BatchNormalization()(conv3_4)
        pool3_1 = MaxPooling2D(pool_size=(2, 2), name='pool3_1')(conv3_4)
        drop3_1 = Dropout(0.3, name='drop3_1')(pool3_1)

        # the 4-th block
        conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_1')(drop3_1)
        conv4_1 = BatchNormalization()(conv4_1)
        conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_2')(conv4_1)
        conv4_2 = BatchNormalization()(conv4_2)
        conv4_3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_3')(conv4_2)
        conv4_3 = BatchNormalization()(conv4_3)
        conv4_4 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_4')(conv4_3)
        conv4_4 = BatchNormalization()(conv4_4)
        pool4_1 = MaxPooling2D(pool_size=(2, 2), name='pool4_1')(conv4_4)
        drop4_1 = Dropout(0.3, name='drop4_1')(pool4_1)

        # the 5-th block
        conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_1')(drop4_1)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_2')(conv5_1)
        conv5_2 = BatchNormalization()(conv5_2)
        conv5_3 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_3')(conv5_2)
        conv5_3 = BatchNormalization()(conv5_3)
        conv5_4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_4')(conv5_3)
        conv5_3 = BatchNormalization()(conv5_3)
        pool5_1 = MaxPooling2D(pool_size=(2, 2), name='pool5_1')(conv5_4)
        drop5_1 = Dropout(0.3, name='drop5_1')(pool5_1)

        # Flatten and output
        flatten = Flatten(name='flatten')(drop5_1)
        ouput = Dense(num_classes, activation='softmax', name='output')(flatten)

        # create model
        model = Model(inputs=visible, outputs=ouput)
        # summary layers
        print(model.summary())

        return model

    def saveModel(self, model):
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")
        print("Saved model to disk")


# if __name__ == '__main__':
#     t = Trainer("../dataset/fer2013.csv")
#     model = t.createModel()
#     t.saveModel(model)
