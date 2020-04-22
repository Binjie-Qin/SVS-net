
import time
import numpy as np
import random
import scipy as sp
import scipy.interpolate
import scipy.ndimage
import scipy.ndimage.interpolation
import random
import h5py
import pylab as py
import matplotlib.pyplot as plt
from skimage import transform
import random
import cv2
plt.switch_backend('agg')
INTENSITY_FACTOR = 0.2
VECTOR_FIELD_SIGMA = 5.  # in pixel
ROTATION_FACTOR = 10  # degree
TRANSLATION_FACTOR = 0.2  # proportion of the image size
SHEAR_FACTOR = 2 * np.pi / 180  # in radian
ZOOM_FACTOR = 0.1


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def random_channel_shift(x1,x2,x3,x4, intensity, channel_index=0):
    x1 = np.rollaxis(x1, channel_index, 0)
    x2 = np.rollaxis(x2, channel_index, 0)
    x3 = np.rollaxis(x3, channel_index, 0)
    x4 = np.rollaxis(x4, channel_index, 0)
    min_x1, max_x1 = np.min(x1), np.max(x1)
    min_x2, max_x2 = np.min(x2), np.max(x2)
    min_x3, max_x3 = np.min(x3), np.max(x3)
    min_x4, max_x4 = np.min(x4), np.max(x4)
    shift = np.random.uniform(-intensity, intensity)  # TODO add a choice if we want the same shift for all channels
    channel_images1 = [np.clip(x_channel + shift, min_x1, max_x1)
                      for x_channel in x1]
    channel_images2 = [np.clip(x_channel + shift, min_x2, max_x2)
                      for x_channel in x2]
    channel_images3 = [np.clip(x_channel + shift, min_x3, max_x3)
                      for x_channel in x3]
    channel_images4 = [np.clip(x_channel + shift, min_x4, max_x4)
                      for x_channel in x4]

    x1 = np.stack(channel_images1, axis=0)
    x1 = np.rollaxis(x1, 0, channel_index + 1)
    x2 = np.stack(channel_images2, axis=0)
    x2 = np.rollaxis(x2, 0, channel_index + 1)
    x3 = np.stack(channel_images3, axis=0)
    x3 = np.rollaxis(x3, 0, channel_index + 1)
    x4 = np.stack(channel_images4, axis=0)
    x4 = np.rollaxis(x4, 0, channel_index + 1)

    return x1,x2,x3,x4

def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [sp.ndimage.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                                final_offset, order=0, mode=fill_mode, cval=cval) for
                      x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix



def train_generator(_X, _Y,_batchSize, iter_times, _keepPctOriginal=0.5, _intensity=INTENSITY_FACTOR, _hflip=True, _vflip=True):
    n_data=_X.shape[0]
    shapeX = _X.shape
    shapeY = _Y.shape
    currentBatch=0
    while 1:
        index=np.random.permutation(n_data)
        X=_X[index,:,:,:,:]
        Y=_Y[index,:,:,:]
        X=np.transpose(X,(0,2,1,3,4))
        for i in range(iter_times):
            if currentBatch == 0:
                x = np.empty((_batchSize, 1, shapeX[2], shapeX[3], shapeX[4]), dtype=np.float32)
                y = np.empty((_batchSize, 1, shapeY[2], shapeY[3]), dtype=np.float32)
            index_list = random.randint(0, n_data - 1)
            img_x = np.empty((shapeX[2], 1, shapeX[3], shapeX[4]), dtype=np.float32)
            img_x1 = X[index_list][0]
            img_x2 = X[index_list][1]
            img_x3 = X[index_list][2]
            img_x4 = X[index_list][3]
            img_y = Y[index_list]
            if random.random() > _keepPctOriginal:
                if _intensity != 0:
                    img_x1, img_x2, img_x3, img_x4 = random_channel_shift(img_x1, img_x2, img_x3, img_x4, _intensity)
                if _hflip == True and random.random() > 0.5:
                    img_x1 = flip_axis(img_x1, 1)
                    img_x2 = flip_axis(img_x2, 1)
                    img_x3 = flip_axis(img_x3, 1)
                    img_x4 = flip_axis(img_x4, 1)

                    img_y = flip_axis(img_y, 1)

                if _vflip == True and random.random() > 0.5:
                    img_x1 = flip_axis(img_x1, 2)
                    img_x2 = flip_axis(img_x2, 2)
                    img_x3 = flip_axis(img_x3, 2)
                    img_x4 = flip_axis(img_x4, 2)
                    img_y = flip_axis(img_y, 2)

                if random.random() > 0.5:
                    angle = np.random.randint(-10, 10)
                    img_x1 = np.reshape(transform.rotate(np.reshape(img_x1, [512, 512]), angle), [1, 512, 512])
                    img_x2 = np.reshape(transform.rotate(np.reshape(img_x2, [512, 512]), angle), [1, 512, 512])
                    img_x3 = np.reshape(transform.rotate(np.reshape(img_x3, [512, 512]), angle), [1, 512, 512])
                    img_x4 = np.reshape(transform.rotate(np.reshape(img_x4, [512, 512]), angle), [1, 512, 512])
                    img_y = np.reshape(transform.rotate(np.reshape(img_y, [512, 512]), angle), [1, 512, 512])

                if random.random() > 0.5:
                    crop_size = [400, 400]
                    w_s = random.randint(0, 512 - crop_size[1])
                    h_s = random.randint(0, 512 - crop_size[0])
                    img1_ = np.reshape(img_x1, [512, 512])[h_s:h_s + crop_size[0], w_s:w_s + crop_size[1]]
                    img2_ = np.reshape(img_x2, [512, 512])[h_s:h_s + crop_size[0], w_s:w_s + crop_size[1]]
                    img3_ = np.reshape(img_x3, [512, 512])[h_s:h_s + crop_size[0], w_s:w_s + crop_size[1]]
                    img4_ = np.reshape(img_x4, [512, 512])[h_s:h_s + crop_size[0], w_s:w_s + crop_size[1]]
                    imgy_ = np.reshape(img_y, [512, 512])[h_s:h_s + crop_size[0], w_s:w_s + crop_size[1]]
                   # print(img1_.shape)

                    img_x1 = transform.resize(img1_, (512, 512))
                    img_x2 = transform.resize(img2_, (512, 512))
                    img_x3 = transform.resize(img3_, (512, 512))
                    img_x4 = transform.resize(img4_, (512, 512))
                    img_y = transform.resize(imgy_, (512, 512))

                    img_x1 = np.reshape(img_x1, [1, 512, 512])
                    img_x2 = np.reshape(img_x2, [1, 512, 512])
                    img_x3 = np.reshape(img_x3, [1, 512, 512])
                    img_x4 = np.reshape(img_x4, [1, 512, 512])
                    img_y = np.reshape(img_y, [1, 512, 512])
            if random.random() > 0.5:
                zoom_factor = 0.2
                z_x, z_y = np.random.uniform(1 - zoom_factor, 1 + zoom_factor, 2)
                t_x = np.random.uniform(-0.2, 0.2) * 512
                t_y = np.random.uniform(-0.2, 0.2) * 512

                M = np.float32([[z_x, 0, t_x], [0, z_y, t_y]])

                img1_ = np.reshape(img_x1, [512, 512])
                img2_ = np.reshape(img_x2, [512, 512])
                img3_ = np.reshape(img_x3, [512, 512])
                img4_ = np.reshape(img_x4, [512, 512])
                imgy_ = np.reshape(img_y, [512, 512])
                dst1 = cv2.warpAffine(img1_, M, (512, 512))
                dst2 = cv2.warpAffine(img2_, M, (512, 512))
                dst3 = cv2.warpAffine(img3_, M, (512, 512))
                dst4 = cv2.warpAffine(img4_, M, (512, 512))
                dsty = cv2.warpAffine(imgy_, M, (512, 512))

                img_x1 = np.reshape(dst1, [1, 512, 512])
                img_x2 = np.reshape(dst2, [1, 512, 512])
                img_x3 = np.reshape(dst3, [1, 512, 512])
                img_x4 = np.reshape(dst4, [1, 512, 512])
                img_y = np.reshape(dsty, [1, 512, 512])
            if random.random() > 0.5:
                _shear = 2 * np.pi / 180
                shear = np.random.uniform(-_shear, _shear)
                shear_matrix = np.array([[np.cos(shear), 0, 0],
                                         [-np.sin(shear), 1, 0]])

                img1_ = np.reshape(img_x1, [512, 512])
                img2_ = np.reshape(img_x2, [512, 512])
                img3_ = np.reshape(img_x3, [512, 512])
                img4_ = np.reshape(img_x4, [512, 512])
                imgy_ = np.reshape(img_y, [512, 512])
                dst1 = cv2.warpAffine(img1_, shear_matrix, (512, 512))
                dst2 = cv2.warpAffine(img2_, shear_matrix, (512, 512))
                dst3 = cv2.warpAffine(img3_, shear_matrix, (512, 512))
                dst4 = cv2.warpAffine(img4_, shear_matrix, (512, 512))
                dsty = cv2.warpAffine(imgy_, shear_matrix, (512, 512))

                img_x1 = np.reshape(dst1, [1, 512, 512])
                img_x2 = np.reshape(dst2, [1, 512, 512])
                img_x3 = np.reshape(dst3, [1, 512, 512])
                img_x4 = np.reshape(dst4, [1, 512, 512])
                img_y = np.reshape(dsty, [1, 512, 512])
            img_x[0] = img_x1[...]
            img_x[1] = img_x2[...]
            img_x[2] = img_x3[...]
            img_x[3] = img_x4[...]
            img_x = np.transpose(img_x, (1, 0, 2, 3))
            x[currentBatch][...] = img_x[...]
            y[currentBatch][...] = img_y[...]
            currentBatch += 1

            if currentBatch==_batchSize:
                currentBatch=0
                yield (x,y)
            elif i ==iter_times-1:
                yield (x[:currentBatch], y[:currentBatch])
                currentBatch = 0

def validation_generator(_X, _Y,_batchSize):
    n_data = _X.shape[0]
    shapeX = _X.shape
    shapeY = _Y.shape
    currentBatch = 0
    index = np.random.permutation(n_data)
    X = _X[index, :, :, :,:]
    Y = _Y[index, :, :, :]
    while 1:
        for i in range(n_data):
            if currentBatch == 0:
                x = np.empty((_batchSize, 1, shapeX[2], shapeX[3],shapeX[4]), dtype=np.float32)
                y = np.empty((_batchSize, 1, shapeY[2], shapeY[3]), dtype=np.float32)
            index_list = random.randint(0, n_data-1)
            img_x = X[index_list]
            img_y = Y[index_list]

            x[currentBatch][...] = img_x[...]
            y[currentBatch][...] = img_y[...]
            currentBatch += 1
            if currentBatch == _batchSize:
                currentBatch = 0
                yield (x, y)
            elif i==n_data-1:
                yield (x[:currentBatch], y[:currentBatch])
                currentBatch = 0

