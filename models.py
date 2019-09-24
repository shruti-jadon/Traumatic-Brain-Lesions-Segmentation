from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from glob import glob
from keras import regularizers


def get_unet(pretrained_weights=None):
    print('Begining Unet Small')
    weight = 32
    nb_filter = [weight, weight * 2, weight * 4, weight * 8, weight * 16]
    inputs = Input(shape=(256,256,1))
    conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5),conv4], axis=3)
    conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6),conv3], axis=3)
    conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7),conv2], axis=3)
    conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8),conv1], axis=3)
    conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def get_unetw(x_train,pretrained_weights=None):
    print('Begining UNet Wide')

    weight=38
    nb_filter = [weight,weight*2,weight*4,weight*8,weight*16]

    #inputs = Input((img_rows, img_cols, 1))
    inputs = Input(shape=x_train.shape[1:])
    conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def get_unetpp(num_class=1, pretrained_weights=None,  deep_supervision=False):
    print('Begining UNet ++')
    nb_filter = [32, 64, 128, 256, 512]
    img_rows = 256
    img_cols = 256
    color_type = 1
    bn_axis = 3

    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    # img_input = Input(shape=x_train.shape[1:])

    conv1_1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        img_input)
    conv1_1 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        pool1)
    conv2_1 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12',
                            padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_2)
    conv1_2 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_2)

    conv3_1 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        pool2)
    conv3_1 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        conv3_1)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22',
                            padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_2)
    conv2_2 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_2)

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13',
                            padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13',
                          axis=bn_axis)
    conv1_3 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_3)
    conv1_3 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_3)

    conv4_1 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
        pool3)
    conv4_1 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
        conv4_1)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32',
                            padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        conv3_2)
    conv3_2 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        conv3_2)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23',
                            padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23',
                          axis=bn_axis)
    conv2_3 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_3)
    conv2_3 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_3)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14',
                            padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14',
                          axis=bn_axis)
    conv1_4 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_4)
    conv1_4 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_4)

    #     conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])
    conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(
        pool4)
    conv5_1 = Conv2D(nb_filter[4], (3, 3), activation='relu', padding='same')(
        conv5_1)

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42',
                            padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
        conv4_2)
    conv4_2 = Conv2D(nb_filter[3], (3, 3), activation='relu', padding='same')(
        conv4_2)

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33',
                            padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33',
                          axis=bn_axis)
    conv3_3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        conv3_3)
    conv3_3 = Conv2D(nb_filter[2], (3, 3), activation='relu', padding='same')(
        conv3_3)

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24',
                            padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24',
                          axis=bn_axis)
    conv2_4 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_4)
    conv2_4 = Conv2D(nb_filter[1], (3, 3), activation='relu', padding='same')(
        conv2_4)

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15',
                            padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4],
                          name='merge15', axis=bn_axis)
    conv1_5 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_5)
    conv1_5 = Conv2D(nb_filter[0], (3, 3), activation='relu', padding='same')(
        conv1_5)

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid',
                              name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(
        conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid',
                              name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(
        conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid',
                              name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(
        conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid',
                              name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(
        conv1_5)

    model = Model(input=img_input, output=[nestnet_output_4])

    if (pretrained_weights):
        print ("loaded weights")
        model.load_weights(pretrained_weights)
    return model


def get_unet3D(x_train):
    in_layer = Input(shape=x_train.shape[1:])
    bn = BatchNormalization()(in_layer)
    cn1 = Conv3D(8,
                 kernel_size=(1, 5, 5),
                 padding='same',
                 activation='relu')(bn)
    cn2 = Conv3D(8,
                 kernel_size=(3, 3, 3),
                 padding='same',
                 activation='linear')(cn1)
    bn2 = Activation('relu')(BatchNormalization()(cn2))

    dn1 = MaxPooling3D((2, 2, 2))(bn2)
    cn3 = Conv3D(16,
                 kernel_size=(3, 3, 3),
                 padding='same',
                 activation='linear')(dn1)
    bn3 = Activation('relu')(BatchNormalization()(cn3))

    dn2 = MaxPooling3D((1, 2, 2))(bn3)
    cn4 = Conv3D(32,
                 kernel_size=(3, 3, 3),
                 padding='same',
                 activation='linear')(dn2)
    bn4 = Activation('relu')(BatchNormalization()(cn4))

    up1 = Deconvolution3D(16,
                          kernel_size=(3, 3, 3),
                          strides=(1, 2, 2),
                          padding='same')(bn4)

    cat1 = concatenate([up1, bn3])

    up2 = Deconvolution3D(8,
                          kernel_size=(3, 3, 3),
                          strides=(2, 2, 2),
                          padding='same')(cat1)

    pre_out = concatenate([up2, bn2])

    pre_out = Conv3D(1,
                     kernel_size=(1, 1, 1),
                     padding='same',
                     activation='sigmoid')(pre_out)

    pre_out = Cropping3D((1, 2, 2))(pre_out)  # avoid skewing boundaries
    out = ZeroPadding3D((1, 2, 2))(pre_out)
    model = Model(inputs = [in_layer], outputs = [out])
    return model

