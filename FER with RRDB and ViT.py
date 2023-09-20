import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import resnet
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, model_from_json
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from readDataset import *
import tensorflow as tf
from layer import GroupNormalization
from keras.losses import categorical_crossentropy
from transformer import vision_transformer_block

def define_model(input_shape=(100, 100, 1), classes=7):

    inputLayer, outLayer, scaleLayer = resnet.ResNet50(input_shape=input_shape, include_top=False,create_encoder=True)

    # Facial Attention Module
    x = Conv2D(32, (3, 3), padding='same')(outLayer)
    x = GroupNormalization()(x)
    x = Activation(gelu)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x1 = x = Conv2D(64, (3, 3), padding='same')(x)
    x = GroupNormalization()(x)
    x = Activation(gelu)(x)
    x = Add()([x, x1])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = GroupNormalization()(x)
    x = Activation(gelu)(x)
    FAM = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Facial Convolution Module
    # 1st Facial Convolution Block
    x1 = x = Conv2D(32, (3, 3), padding='same')(outLayer)
    x = GroupNormalization()(x)
    x = Activation(gelu)(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = GroupNormalization()(x)
    x = Activation(gelu)(x)
    x = Add()([x, x1])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 2st Facial Convolution Block
    x1 = x = Conv2D(64, (3, 3), padding='same')(x)
    x = GroupNormalization()(x)
    x = Activation(gelu)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = GroupNormalization()(x)
    x = Activation(gelu)(x)
    x = Add()([x, x1])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = GroupNormalization()(x)
    x = Activation(gelu)(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = GroupNormalization()(x)
    FCM = Activation(gelu)(x)
    x = Multiply()([FAM, FCM])
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    GAM = Dense(7, activation='relu')(x)

    # Residual in Residual Dense Block
    RRDBlock = outLayer  # RRD_1
    RDBlock = RDBlocks(outLayer, name='RDBlock_1', count=3, g=32)
    RDBlock = RDBlocks(RDBlock, name='RDBlock_2', count=3, g=32)
    RDBlock = RDBlocks(RDBlock, name='RDBlock_3', count=3, g=32)
    RDBlock = RDBlocks(RDBlock, name='RDBlock_4', count=3, g=32)

    x = Add()([RRDBlock, RDBlock])  # RRD_1
    x = vision_transformer_block(x)
    # x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    LAM = Dense(7, activation='relu')(x)
    x = Multiply()([LAM, GAM])
    x = Dense(7, activation='softmax')(x)
    return Model(inputs=inputLayer, outputs=x)


# 主程式
def run_model():
    fer_classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    # x_train, x_test, y_train, y_test, x_val, y_val = readFERplus()
    x_train, x_test, y_train, y_test = readRAFDB()
    datagen = data_augmentation(x_train)

    epochs = 400
    batch_size = 2

    # Training model from scratch

    black = define_model(input_shape=x_train[0].shape, classes=len(fer_classes))
    black.summary()
    black.compile(optimizer=Adam(learning_rate = 0.0001), loss=[categorical_crossentropy],loss_weights=[1.0], metrics=['accuracy'])
    history = black.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, steps_per_epoch=len(x_train) // batch_size, validation_data=(x_test, y_test), verbose=2)
    test_loss, test_acc = black.evaluate(x_test, y_test, batch_size=batch_size)

    plot_acc_loss(history)
    save_model_and_weights(black, test_acc)

# 資料增強
def data_augmentation(x_train):
    shift = 0.1
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        height_shift_range=shift,
        width_shift_range=shift)
    datagen.fit(x_train)
    return datagen

# 儲存權重
def save_model_and_weights(model, test_acc):
    # Serialize and save model to JSON
    test_acc = int(test_acc * 10000)
    model_json = model.to_json()
    with open('Saved-Models\\model' + str(test_acc) + '.json', 'w') as json_file:
        json_file.write(model_json)
    # Serialize and save weights to JSON
    model.save_weights('Saved-Models\\model' + str(test_acc) + '.h5')
    print('Model and weights are saved in separate files.')

# 激活函數Gelu
def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

# 畫結果圖
def plot_acc_loss(history):
    # Plot accuracy graph
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1.0])
    plt.legend(loc='upper left')
    plt.show()

    # Plot loss graph
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim([0, 3.5])
    plt.legend(loc='upper right')
    plt.show()

# loss
def Focalloss(y_true, y_pred):
    alpha = 0.25
    gamma = 2.0
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    Focalloss = - y_true[...] * K.log(y_pred)
    Focalloss = alpha * K.pow(1 - y_pred, gamma) * Focalloss
    Focalloss = K.mean(K.sum(Focalloss, axis=-1))
    return Focalloss

# Spatial_attention
def spatial_attention(x):
    se_output = x
    maxpool_spatial = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(se_output)
    avgpool_spatial = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(se_output)
    max_avg_pool_spatial = Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    SA = Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid',
                kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)
    x = multiply([se_output, SA])
    return x

# Channel_attention
def channel_attention(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def RDBlocks(x, name, count=6, g=32):
    ## 6 layers of RDB block
    ## this thing need to be in a damn loop for more customisability
    li = [x]
    pas = Conv2D(filters=g, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=gelu,
                 name=name + '_conv1')(x)

    for i in range(2, count + 1):
        li.append(pas)
        out = Concatenate(axis=3)(li)  # conctenated output
        pas = Conv2D(filters=g, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=gelu,
                     name=name + '_conv' + str(i))(out)

    # feature extractor from the dense net
    li.append(pas)
    out = Concatenate(axis=3)(li)
    feat = Conv2D(filters=2048, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=gelu,
                  name=name + '_Local_Conv')(out)
    feat = Add()([feat, x])
    return feat

if __name__ == '__main__':
    run_model()
