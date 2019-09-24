import os
import numpy as np # linear algebra
import nibabel as nib
from nilearn import plotting
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
import keras.backend as K
import argparse
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from loss_functions import *
from models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.utils import multi_gpu_model
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def import_files():
    BASE_IMG_PATH = './sample_data/'
    all_images = glob(os.path.join(BASE_IMG_PATH, '3d_images', 'IMG_*'))
    print(len(all_images), ' matching files found:', all_images[1])
    train_paths, test_paths = train_test_split(all_images, random_state=2018,
                                               test_size=0.4)
    print(len(train_paths), 'training size')
    print(len(test_paths), 'testing size')
    return train_paths, test_paths


DS_FACT = 2  # downscale
SEED = 201


def check_arr_nan(in_img, in_mask):
    indices = []
    for i in range(0, len(in_mask)):
        if (not np.any(in_mask[i])):
            continue
        else:
            indices.append(i)
    return len(indices), np.array(indices)


def mask_preprocess(train_mask):
    indices = np.where(train_mask > 0)
    train_mask[indices] = 1.0
    return train_mask


def read_all_slices(in_paths, path_image, rescale=False, balanced=True,
                    intr=True, ext=False, hem=False):
    """ writing sample code, write different logic, just to flatten out all
    image slices, this logic is messing"""
    cur_vol = np.concatenate(
        [np.transpose(nib.load(c_path).get_data())[:, ::DS_FACT, ::DS_FACT] for
         c_path in in_paths], 0)
    cur_vol_mask = np.concatenate(
        [np.transpose(nib.load(c_path).get_data())[:, ::DS_FACT, ::DS_FACT] for
         c_path in path_image], 0)

    if balanced:
        num_no_bleed, s_id = check_arr_nan(cur_vol, cur_vol_mask)
        num_bleed = range(0, len(cur_vol))
        num_bleed = np.setdiff1d(num_bleed, s_id)
        indices = np.random.choice(num_bleed.shape[0], num_no_bleed,
                                   replace=True)
        s_id = np.concatenate([s_id, num_bleed[indices]])
        cur_vol, cur_vol_mask = cur_vol[s_id], cur_vol_mask[s_id]
    cur_vol_mask = mask_preprocess(cur_vol_mask)
    if intr:
        index=np.where(cur_vol>479)
        cur_vol[index]=-1024.0
        index=np.where(cur_vol<-39)
        cur_vol[index]=-1024.0
    if ext:
        index = np.where(cur_vol > 2285)
        cur_vol[index] = -1024.0
        index = np.where(cur_vol < -916)
        cur_vol[index] = -1024.0
    if hem:
        index = np.where(cur_vol > 1360)
        cur_vol[index] = -1024.0
        index = np.where(cur_vol < -630)
        cur_vol[index] = -1024.0
    if rescale:
        cur_vol = (cur_vol.astype(np.float32) - np.mean(
            cur_vol.astype(np.float32))) / np.std(cur_vol.astype(np.float32))
        return cur_vol, cur_vol_mask
    else:
        return cur_vol, cur_vol_mask


def read_both(in_paths):
    print ("BEGIN READING IMAGES")
    path_image = list(map(lambda x: x.replace('IMG_', 'MASK_'), in_paths))
    return read_all_slices(in_paths, path_image, rescale=False)


def plot_loaded_files(train_vol,train_mask):
    slices = train_vol[-1000, :, :]
    masks = train_mask[-1000, :, :]
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(slices, cmap="gray", origin="lower")
    axes[1].imshow(masks, cmap="gray", origin="lower")
    slices = train_vol[100, :, :]
    masks = train_mask[100, :, :]
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(slices, cmap="gray", origin="lower")
    axes[1].imshow(masks, cmap="gray", origin="lower")


def data_to_numpy():
    train_paths,test_paths = import_files()
    train_vol, train_mask = read_both(train_paths)
    print('train', train_vol.shape, 'mask', train_mask.shape)
    test_vol, test_mask = read_both(test_paths)
    return train_vol, train_mask, test_vol, test_mask


def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(shear_range=0.001, fill_mode='nearest').flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(shear_range=0.001, fill_mode='nearest').flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


def run_me():
    train_vol, train_mask, test_vol, test_mask = data_to_numpy()
    x_data, y_data = train_vol, train_mask
    x_data = x_data[:, :, :, np.newaxis]
    y_data = y_data[:, :, :, np.newaxis]
    test_vol = test_vol[:, :, :, np.newaxis]
    test_mask = test_mask[:, :, :, np.newaxis]
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data,test_size=0.2)
    return x_train, x_val, y_train, y_val, test_vol, test_mask


def plot_loss(history,name):
    plt.plot(history.history['loss'], color='b', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='validation_loss')
    plt.show()
    plt.title (name)
    plt.savefig(name+'loss_function', bbox_inches="tight")


def plot_dc(history,name):
    plt.plot(history.history['dice_coef'], color='b')
    plt.plot(history.history['val_dice_coef'], color='r')
    plt.show()
    plt.title (name)
    plt.savefig(name+'dice_coefficient', bbox_inches="tight")


def main(case):
    x_train, x_val, y_train, y_val, test_vol, test_mask = run_me()
    model = get_unetpp(pretrained_weights='./weights/weight.h5')
    if case:
        try:
          model = multi_gpu_model(model)
        except:
           pass
        model.compile(optimizer=Adam(2e-4), loss=focal_tversky, metrics=[dice_coef, sensitivity, specificity])
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_dice_coef',factor=0.5, patience=30, verbose=1, mode='auto', min_delta=0.001, cooldown=0, min_lr=0.000000001)
        epochs = 300
        batch_size = 16
        steps_per_epoch = 50
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=70)
        weight_saver = ModelCheckpoint('./weights/training.h5',monitor='val_dice_coef', save_best_only=True, save_weights_only=True, mode = 'max')
        history = model.fit_generator(
            my_generator(x_train, y_train, batch_size=batch_size),
            epochs=epochs, validation_data=(x_val, y_val),
            verbose=2, steps_per_epoch=steps_per_epoch
            , callbacks=[weight_saver, learning_rate_reduction,es])

        plot_loss(history, 'INT_Focal_Tversky_Unetpp')
        plot_dc(history, 'INT_Focal_TverskyL_Unetpp')
    else:
        dice_coefficient_test = []
        sensitivity_test =[]
        specificity_test = []
        y_test = model.predict(test_vol)
        test_mask=test_mask.astype(np.float32)
        for i in range(0, len(test_mask)):
            dice_coefficient_test.append(K.eval(dice_coef(test_mask[i], y_test[i])))
            sensitivity_test.append(K.eval(sensitivity(test_mask[i],y_test[i])))
            specificity_test.append(K.eval(specificity(test_mask[i],y_test[i])))
        print ("average of Dice Coefficient, Sensitivity, Specificity", dice_coefficient_test.mean(), sensitivity_test.mean(), specificity_test.mean())


def run():
    parser = argparse.ArgumentParser(description='Arguments for training or testing')
    parser.add_argument("--train", dest="train", help="train model or not", type=str, required=False)
    args = parser.parse_args()
    if args.train:
        main(True)
    else:
        main(False)


if __name__ == '__main__':
    run()




