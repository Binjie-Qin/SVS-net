###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,Add,Convolution2D,merge,Conv3D, MaxPooling3D,Multiply
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
import h5py
import sys
sys.path.insert(0, './lib/')
#from help_functions import *
from keras.layers import BatchNormalization,SpatialDropout3D,Reshape,GlobalMaxPooling3D,GlobalAveragePooling2D
#function to obtain data for training/testing (validation)
#from extract_patches import get_data_training
from keras.layers.core import  Dropout, Activation
from keras import backend as K
import tensorflow as tf
print(K.backend())
from data_feed import *
from keras import optimizers
#from pre_processing import my_PreProc
import math
import sys
sys.setrecursionlimit(4000)
#Define the neural network
def focal_loss(gamma=2,alpha=0.75):
    def focal_loss_fixed(y_true,y_pred):
        pt_1=tf.where(tf.equal(y_true,1),y_pred,tf.ones_like(y_pred))
        pt_0=tf.where(tf.equal(y_true,0),y_pred,tf.zeros_like(y_pred))
        return -K.sum(alpha*K.pow(1.-pt_1,gamma)*K.log(pt_1))-K.sum((1-alpha)*K.pow(pt_0,gamma)*K.log(1.-pt_0))
    return focal_loss_fixed
def block_2_conv(input,num_filter):
    conv1=Conv2D(num_filter,(3,3),strides=(1,1),padding='same',data_format='channels_first')(input)
    conv1_bn=BatchNormalization(axis=1)(conv1)
    conv1_relu=Activation('relu')(conv1_bn)
    conv2=Conv2D(num_filter,(3,3),strides=(1,1),padding='same',data_format='channels_first')(conv1_relu)
    conv2_bn=BatchNormalization(axis=1)(conv2)
    conv2_add=Add()([input,conv2_bn])
    conv2_relu=Activation('relu')(conv2_add)
    return conv2_relu

def block_2_conv3D(input,num_filter):
    conv1 = Conv3D(num_filter, (3, 3,3), strides=(1, 1,1), padding='same', data_format='channels_first')(input)
    conv1_bn = BatchNormalization(axis=1)(conv1)
    conv1_relu = Activation('relu')(conv1_bn)
    conv2 = Conv3D(num_filter, (3, 3,3), strides=(1, 1,1), padding='same', data_format='channels_first')(conv1_relu)
    conv2_bn = BatchNormalization(axis=1)(conv2)
    conv2_add = Add()([input, conv2_bn])
    conv2_relu = Activation('relu')(conv2_add)
    return conv2_relu
def attention_block(input,iter,depth):
    global_pool=GlobalMaxPooling3D(data_format='channels_first')(input)
    global_pool1=Reshape((depth,1,1,1))(global_pool)
    conv_1x1=Conv3D(depth,(1,1,1),padding='same',data_format='channels_first')(global_pool1)
    relu_out=Activation('relu')(conv_1x1)
    conv_2x1=Conv3D(depth,(1,1,1),strides=(1,1,1),padding='same',data_format='channels_first')(relu_out)
    sigmoid_out=Activation('sigmoid')(conv_2x1)
    concat1=sigmoid_out
    #print("***********1")
    #print(concat1.shape)
    for i in range(4-1):
        concat1=concatenate([concat1,sigmoid_out],axis=2)
    concat2=concat1
    for j in range(iter-1):
        concat2=concatenate([concat2,concat1],axis=3)
    concat3=concat2
    for k in range(iter-1):
        concat3=concatenate([concat3,concat2],axis=4)
    #print("************2")
    #print(concat3.shape)
    out=Multiply()([input,concat3])
    return out
def saliency_map_attention_block(input,depth):
    conv_1x1=Conv3D(depth,(1,1,1),padding='same',data_format='channels_first')(input)
    relu_out=Activation('relu')(conv_1x1)
    conv_2x1=Conv3D(depth,(1,1,1),padding='same',data_format='channels_first')(relu_out)
    sigmoid_out=Activation('sigmoid')(conv_2x1)
    out1=Multiply()([input,sigmoid_out])
    out=Add()([input,out1])
    return out
def channel_attnetion_block(low_input,high_input,depth,size):
    input=concatenate([low_input,high_input],axis=1)
    global_pool=GlobalAveragePooling2D(data_format='channels_first')(input)
    global_pool1 = Reshape((2*depth, 1, 1))(global_pool)
    conv_1x1 = Conv2D(depth, (1, 1), padding='same', data_format='channels_first')(global_pool1)
    relu_out = Activation('relu')(conv_1x1)
    conv_2x1 = Conv2D(depth, (1, 1), strides=(1, 1), padding='same', data_format='channels_first')(relu_out)
    sigmoid_out = Activation('sigmoid')(conv_2x1)
    concat1 = sigmoid_out
    for i in range(size-1):
        concat1=concatenate([concat1,sigmoid_out],axis=2)
    concat2=concat1
    for j in range(size-1):
        concat2=concatenate([concat2,concat1],axis=3)
    out1 = Multiply()([low_input, concat2])
    out2=Add()([out1,high_input])

    return out2





# F1 score: harmonic mean of precision and sensitivity DICE = 2*TP/(2*TP + FN + FP)
def DiceCoef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f*y_pred_f)
	return (2.*intersection)/(K.sum(y_true_f) + K.sum(y_pred_f) + 0.00001)

def DiceCoefLoss(y_true, y_pred):
	return -DiceCoef(y_true, y_pred)



def  get_unet3D_new_4_fram_2(n_ch,frame,patch_height,patch_width):
    inputs = Input(shape=(n_ch, frame,patch_height, patch_width))
    conv0 = Conv3D(8, (1, 1,1), padding='same')(inputs)

    conv1 = block_2_conv3D(conv0, 8)
    ## channel attention
    #out1=attention_block(conv1,512,8)

    ###特征输出
    conv1_3d_2d = Conv3D(8, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv1)
    #conv1_3d_2d = Conv3D(8, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(out1)
    conv1_trans_2d = Reshape((8, 512, 512))(conv1_3d_2d)

    conv1_1 = Conv3D(16, (2, 2,2), strides=(1,2,2),padding='same', data_format='channels_first')(conv1)
    conv1_1 = BatchNormalization(axis=1)(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)



    conv2 = block_2_conv3D(conv1_1, 16)
    ## channel attention
    #out2 = attention_block(conv2, 256, 16)

    ###特征输出
    conv2_3d_2d = Conv3D(16, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv2)
    #conv2_3d_2d = Conv3D(16, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(out2)
    conv2_trans_2d = Reshape((16, 256, 256))(conv2_3d_2d)

    conv2_1 = Conv3D(32, (2, 2,2), strides=(1,2,2), padding='same',data_format='channels_first')(conv2)
    conv2_1 = BatchNormalization(axis=1)(conv2_1)
    conv2_1 = Activation('relu')(conv2_1)



    conv3 = block_2_conv3D(conv2_1, 32)
    ## channel attention
    #out3 = attention_block(conv3, 128, 32)

    ###特征输出
    conv3_3d_2d = Conv3D(32, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv3)
    #conv3_3d_2d = Conv3D(32, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(out3)
    conv3_trans_2d = Reshape((32, 128, 128))(conv3_3d_2d)

    conv3_1 = Conv3D(64, (2, 2,2), strides=(1,2,2),padding='same', data_format='channels_first')(conv3)
    conv3_1 = BatchNormalization(axis=1)(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)



    conv4 = block_2_conv3D(conv3_1, 64)
    ##saliency_map
    out4_1=saliency_map_attention_block(conv4,64)
    ## channel attention
    #out4 = attention_block(conv4, 64, 64)
    out4 = attention_block(out4_1, 64, 64)

    ###特征输出
    #conv4_3d_2d = Conv3D(64, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv4)
    conv4_3d_2d = Conv3D(64, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(out4)
    conv4_trans_2d = Reshape((64, 64, 64))(conv4_3d_2d)

    conv4_1 = Conv3D(128, (2, 2,2), strides=(1,2,2), padding='same',data_format='channels_first')(conv4)
    conv4_1 = BatchNormalization(axis=1)(conv4_1)
    conv4_1 = Activation('relu')(conv4_1)



    conv5 = block_2_conv3D(conv4_1, 128)
    ## channel attention
    out5 = attention_block(conv5, 32, 128)

    ###特征输出
    #conv5_3d_2d = Conv3D(128, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv5)
    conv5_3d_2d = Conv3D(128, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(out5)
    conv5_trans_2d = Reshape((128, 32, 32))(conv5_3d_2d)

    conv5_dropout = SpatialDropout3D(0.5,data_format='channels_first')(conv5)
    conv5_1 = Conv3D(256, (2, 2,2), strides=(1,2,2), padding='same',data_format='channels_first')(conv5_dropout)
    conv5_1 = BatchNormalization(axis=1)(conv5_1)
    conv5_1 = Activation('relu')(conv5_1)




    conv6 = block_2_conv3D(conv5_1, 256)
    ## channel attention
    out6 = attention_block(conv6, 16, 256)
    ###特征输出
    #conv6_3d_2d=Conv3D(256,(4,1,1),strides=(1,1,1),data_format='channels_first')(conv6)
    conv6_3d_2d = Conv3D(256, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(out6)
    conv6_trans_2d=Reshape((256,16,16))(conv6_3d_2d)

    conv6_dropout = SpatialDropout3D(0.5,data_format='channels_first')(conv6)
    conv6_1 = Conv3D(512, (2, 2,2), strides=(1,2,2), padding='same',data_format='channels_first')(conv6_dropout)
    conv6_1 = BatchNormalization(axis=1)(conv6_1)
    conv6_1 = Activation('relu')(conv6_1)

    conv3d_2d=Conv3D(512,(4,1,1),strides=(1,1,1),data_format='channels_first')(conv6_1)
    #print(conv3d_2d.shape)
    conv_trans_2d=Reshape((512,8,8))(conv3d_2d)


    up1 = UpSampling2D(size=(2, 2))(conv_trans_2d)
    up1_1 = Conv2D(256, (2, 2), strides=1, padding='same', data_format='channels_first')(up1)
    up1_1 = BatchNormalization(axis=1)(up1_1)
    up1_1 = Activation('relu')(up1_1)
    up1_2 = concatenate([ conv6_trans_2d , up1_1], axis=1)
    up1_3 = block_2_conv(up1_2, 512)

    up2 = UpSampling2D(size=(2, 2))(up1_3)
    up2_1 = Conv2D(128, (2, 2), strides=1, padding='same', data_format='channels_first')(up2)
    up2_1 = BatchNormalization(axis=1)(up2_1)
    up2_1 = Activation('relu')(up2_1)
    up2_2 = concatenate([conv5_trans_2d, up2_1], axis=1)
    up2_3 = block_2_conv(up2_2, 256)

    up3 = UpSampling2D(size=(2, 2))(up2_3)
    up3_1 = Conv2D(64, (2, 2), strides=1, padding='same', data_format='channels_first')(up3)
    up3_1 = BatchNormalization(axis=1)(up3_1)
    up3_1 = Activation('relu')(up3_1)
    up3_2 = concatenate([conv4_trans_2d, up3_1], axis=1)
    up3_3 = block_2_conv(up3_2, 128)

    up4 = UpSampling2D(size=(2, 2))(up3_3)
    up4_1 = Conv2D(32, (2, 2), strides=1, padding='same', data_format='channels_first')(up4)
    up4_1 = BatchNormalization(axis=1)(up4_1)
    up4_1 = Activation('relu')(up4_1)
    up4_2 = concatenate([conv3_trans_2d, up4_1], axis=1)
    up4_3 = block_2_conv(up4_2, 64)

    up5 = UpSampling2D(size=(2, 2))(up4_3)
    up5_1 = Conv2D(16, (2, 2), strides=1, padding='same', data_format='channels_first')(up5)
    up5_1 = BatchNormalization(axis=1)(up5_1)
    up5_1 = Activation('relu')(up5_1)
    up5_2 = concatenate([conv2_trans_2d, up5_1], axis=1)
    up5_3 = block_2_conv(up5_2, 32)

    up6 = UpSampling2D(size=(2, 2))(up5_3)
    up6_1 = Conv2D(8, (2, 2), strides=1, padding='same', data_format='channels_first')(up6)
    up6_1 = BatchNormalization(axis=1)(up6_1)
    up6_1 = Activation('relu')(up6_1)
    up6_2 = concatenate([conv1_trans_2d, up6_1], axis=1)
    up6_3 = block_2_conv(up6_2, 16)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up6_3)

    model = Model(inputs=inputs, outputs=outputs)
    #model.compile(optimizer='sgd', loss=DiceCoefLoss, metrics=[DiceCoef])
    return model



def  get_unet3D_new_4_fram_2_new(n_ch,frame,patch_height,patch_width):
    inputs = Input(shape=(n_ch, frame,patch_height, patch_width))
    conv0 = Conv3D(8, (1, 1,1), padding='same')(inputs)

    conv1 = block_2_conv3D(conv0, 8)
    ## channel attention
    #out1=attention_block(conv1,512,8)

    ###特征输出
    conv1_3d_2d = Conv3D(8, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv1)
    #conv1_3d_2d = Conv3D(8, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(out1)
    conv1_trans_2d = Reshape((8, 512, 512))(conv1_3d_2d)

    conv1_1 = Conv3D(16, (2, 2,2), strides=(1,2,2),padding='same', data_format='channels_first')(conv1)
    conv1_1 = BatchNormalization(axis=1)(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)



    conv2 = block_2_conv3D(conv1_1, 16)
    ## channel attention
    #out2 = attention_block(conv2, 256, 16)

    ###特征输出
    conv2_3d_2d = Conv3D(16, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv2)
    #conv2_3d_2d = Conv3D(16, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(out2)
    conv2_trans_2d = Reshape((16, 256, 256))(conv2_3d_2d)

    conv2_1 = Conv3D(32, (2, 2,2), strides=(1,2,2), padding='same',data_format='channels_first')(conv2)
    conv2_1 = BatchNormalization(axis=1)(conv2_1)
    conv2_1 = Activation('relu')(conv2_1)



    conv3 = block_2_conv3D(conv2_1, 32)
    ## channel attention
    #out3 = attention_block(conv3, 128, 32)

    ###特征输出
    conv3_3d_2d = Conv3D(32, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv3)
    #conv3_3d_2d = Conv3D(32, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(out3)
    conv3_trans_2d = Reshape((32, 128, 128))(conv3_3d_2d)

    conv3_1 = Conv3D(64, (2, 2,2), strides=(1,2,2),padding='same', data_format='channels_first')(conv3)
    conv3_1 = BatchNormalization(axis=1)(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)



    conv4 = block_2_conv3D(conv3_1, 64)
    ##saliency_map
    #out4_1=saliency_map_attention_block(conv4,64)
    ## channel attention
    #out4 = attention_block(conv4, 64, 64)
    #out4 = attention_block(out4_1, 64, 64)

    ###特征输出
    #conv4_3d_2d = Conv3D(64, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv4)
    conv4_3d_2d = Conv3D(64, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv4)
    conv4_trans_2d = Reshape((64, 64, 64))(conv4_3d_2d)

    conv4_1 = Conv3D(128, (2, 2,2), strides=(1,2,2), padding='same',data_format='channels_first')(conv4)
    conv4_1 = BatchNormalization(axis=1)(conv4_1)
    conv4_1 = Activation('relu')(conv4_1)



    conv5 = block_2_conv3D(conv4_1, 128)
    ## channel attention
    #out5 = attention_block(conv5, 32, 128)

    ###特征输出
    #conv5_3d_2d = Conv3D(128, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv5)
    conv5_3d_2d = Conv3D(128, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(conv5)
    conv5_trans_2d = Reshape((128, 32, 32))(conv5_3d_2d)

    conv5_dropout = SpatialDropout3D(0.5,data_format='channels_first')(conv5)
    conv5_1 = Conv3D(256, (2, 2,2), strides=(1,2,2), padding='same',data_format='channels_first')(conv5_dropout)
    conv5_1 = BatchNormalization(axis=1)(conv5_1)
    conv5_1 = Activation('relu')(conv5_1)




    conv6 = block_2_conv3D(conv5_1, 256)
    ## channel attention
    out6 = attention_block(conv6, 16, 256)
    ###特征输出
    #conv6_3d_2d=Conv3D(256,(4,1,1),strides=(1,1,1),data_format='channels_first')(conv6)
    conv6_3d_2d = Conv3D(256, (4, 1, 1), strides=(1, 1, 1), data_format='channels_first')(out6)
    conv6_trans_2d=Reshape((256,16,16))(conv6_3d_2d)

    conv6_dropout = SpatialDropout3D(0.5,data_format='channels_first')(conv6)
    conv6_1 = Conv3D(512, (2, 2,2), strides=(1,2,2), padding='same',data_format='channels_first')(conv6_dropout)
    conv6_1 = BatchNormalization(axis=1)(conv6_1)
    conv6_1 = Activation('relu')(conv6_1)

    conv3d_2d=Conv3D(512,(4,1,1),strides=(1,1,1),data_format='channels_first')(conv6_1)
    #print(conv3d_2d.shape)
    conv_trans_2d=Reshape((512,8,8))(conv3d_2d)


    up1 = UpSampling2D(size=(2, 2))(conv_trans_2d)
    up1_1 = Conv2D(256, (2, 2), strides=1, padding='same', data_format='channels_first')(up1)
    up1_1 = BatchNormalization(axis=1)(up1_1)
    up1_1 = Activation('relu')(up1_1)
    up1_2=channel_attnetion_block(conv6_trans_2d,up1_1,256,16)

    #up1_2 = concatenate([ conv6_trans_2d , up1_1], axis=1)
    up1_3 = block_2_conv(up1_2, 256)

    up2 = UpSampling2D(size=(2, 2))(up1_3)
    up2_1 = Conv2D(128, (2, 2), strides=1, padding='same', data_format='channels_first')(up2)
    up2_1 = BatchNormalization(axis=1)(up2_1)
    up2_1 = Activation('relu')(up2_1)
    up2_2=channel_attnetion_block(conv5_trans_2d,up2_1,128,32)
    #up2_2 = concatenate([conv5_trans_2d, up2_1], axis=1)
    up2_3 = block_2_conv(up2_2, 128)

    up3 = UpSampling2D(size=(2, 2))(up2_3)
    up3_1 = Conv2D(64, (2, 2), strides=1, padding='same', data_format='channels_first')(up3)
    up3_1 = BatchNormalization(axis=1)(up3_1)
    up3_1 = Activation('relu')(up3_1)
    up3_2=channel_attnetion_block(conv4_trans_2d,up3_1,64,64)
    #up3_2 = concatenate([conv4_trans_2d, up3_1], axis=1)
    up3_3 = block_2_conv(up3_2, 64)

    up4 = UpSampling2D(size=(2, 2))(up3_3)
    up4_1 = Conv2D(32, (2, 2), strides=1, padding='same', data_format='channels_first')(up4)
    up4_1 = BatchNormalization(axis=1)(up4_1)
    up4_1 = Activation('relu')(up4_1)
    up4_2=channel_attnetion_block(conv3_trans_2d,up4_1,32,128)
    #up4_2 = concatenate([conv3_trans_2d, up4_1], axis=1)
    up4_3 = block_2_conv(up4_2, 32)

    up5 = UpSampling2D(size=(2, 2))(up4_3)
    up5_1 = Conv2D(16, (2, 2), strides=1, padding='same', data_format='channels_first')(up5)
    up5_1 = BatchNormalization(axis=1)(up5_1)
    up5_1 = Activation('relu')(up5_1)
    up5_2=channel_attnetion_block(conv2_trans_2d,up5_1,16,256)
   # up5_2 = concatenate([conv2_trans_2d, up5_1], axis=1)
    up5_3 = block_2_conv(up5_2, 16)

    up6 = UpSampling2D(size=(2, 2))(up5_3)
    up6_1 = Conv2D(8, (2, 2), strides=1, padding='same', data_format='channels_first')(up6)
    up6_1 = BatchNormalization(axis=1)(up6_1)
    up6_1 = Activation('relu')(up6_1)
    up6_2=channel_attnetion_block(conv1_trans_2d,up6_1,8,512)
    #up6_2 = concatenate([conv1_trans_2d, up6_1], axis=1)
    up6_3 = block_2_conv(up6_2, 8)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up6_3)

    model = Model(inputs=inputs, outputs=outputs)
    #model.compile(optimizer='sgd', loss=DiceCoefLoss, metrics=[DiceCoef])
    return model










#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
_batchSize = int(config.get('training settings', 'batch_size'))




n_ch=1
frame=4
patch_height=512
patch_width=512
model = get_unet3D_new_4_fram_2_new(n_ch, frame,patch_height, patch_width)  #the U-net model
## data parallel
#parallel_model=multi_gpu_model(model,gpus=2)
parallel_model=model
sgd= optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
parallel_model.compile(optimizer=sgd, loss=DiceCoefLoss, metrics=[DiceCoef])
#parallel_model.compile(optimizer='sgd', loss=[focal_loss(gamma=2,alpha=0.25)], metrics=[DiceCoef])


print ("Check: final output of the network:")
print (parallel_model.output_shape)
#plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)




new_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original')
new_train_imgs_groundTruth=path_data + config.get('data paths', 'train_groundTruth')
train_data_ori= h5py.File(new_train_imgs_original,'r')
train_data_gt=h5py.File(new_train_imgs_groundTruth,'r')

train_imgs_original= np.array(train_data_ori['image'])
train_groundTruth=np.array(train_data_gt['image'])

train_imgs = train_imgs_original/np.max(train_imgs_original)
train_masks = train_groundTruth/np.max(train_groundTruth)

#check masks are within 0-1
#assert(np.min(train_masks)==0 and np.max(train_masks)==1)
print("imgs max value:")
print(np.max(train_imgs))
print("imgs min value")
print(np.min(train_imgs))
print("label max value")
print(np.max(train_masks))
print("label min value")
print(np.min(train_masks))
print ("\ntrain images/masks shape:")
print (train_imgs.shape)
print ("train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs)))
print ("train masks are within 0-1\n")
#============  Training ==================================
checkpoint_test = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', monitor='val_loss', save_best_only=True,save_weights_only=True) #save at each epoch if the validation decreased
checkpoint = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment + "bestTrainWeight" + ".h5", monitor='loss', save_best_only=True, save_weights_only=True)

def step_decay(epoch):
     lrate = 0.01 #the initial learning rate (by default in keras)
     if epoch%200==0:
         lrate=lrate*0.1
     return lrate
     

lrate_drop = LearningRateScheduler(step_decay)

keepPctOriginal = 0.5
hflip = True
vflip = True
iter_times=250
num=train_imgs_original.shape[0]
np.random.seed(0)
index=list(np.random.permutation(num))
_X_train=train_imgs[index][0:174]
_Y_train=train_masks[index][0:174]
print(_X_train.shape)
print(_Y_train.shape)
_X_vali=train_imgs[index][174:219]
_Y_vali=train_masks[index][174:219]
print(_X_vali.shape)
print(_Y_vali.shape)


def ImgGenerator():
    for image in train_generator(_X_train, _Y_train,_batchSize, iter_times, _keepPctOriginal=0.5,
                                  _intensity=INTENSITY_FACTOR, _hflip=True, _vflip=True):
          yield image
def valiGenerator():
    for image in validation_generator(_X_vali, _Y_vali,_batchSize):
        yield image

stepsPerEpoch = math.ceil((num-40) / _batchSize)
validationSteps = math.ceil(40 / _batchSize)
history = parallel_model.fit_generator(ImgGenerator(), verbose=2, workers=1,
                                                 validation_data=valiGenerator(),
                                                 steps_per_epoch=stepsPerEpoch, epochs=N_epochs,
                                                 validation_steps=validationSteps,
                                                 callbacks=[lrate_drop,checkpoint,checkpoint_test])
model.summary()
#========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)


