import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

def build_model(num_classes):
    #CNN structure
    model = Sequential()

    #第一卷积层
    model.add(Conv2D(64, 5, 5, input_shape=(48, 48, 1)))
    #relu
    model.add(Activation('relu'))
    #池化层
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    #第二卷积层
    model.add(Conv2D(64, 3, 3, activation='relu'))
    # relu
    model.add(Activation('relu'))
    # 池化层
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    #第三卷积层
    model.add(Conv2D(128, 3, 3, activation='relu'))
    #relu
    model.add(Activation('relu'))
    # 池化层
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    #降维，从卷积层到全连接层的过渡。
    model.add(Flatten())

    # 全连接层
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))      #防过拟合
    # 全连接层
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))      #防过拟合
    # 全连接层
    model.add(Dense(num_classes, activation='softmax'))

    return model
