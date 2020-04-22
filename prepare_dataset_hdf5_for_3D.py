#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE_new database
#
#============================================================

import os
import h5py
from PIL import Image
import numpy as np
import os
import pylab as py
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)
    f.close()

##3------------Path of the images 3D--------------------------------------------------------------
original_imgs_test = "/home/a/ori_retinal_vessel_pretraining/final_experiments/new_test_data/##3D/"

Nimgs_test = 37
channels = 1
height = 512
width = 512
frame=4
dataset_path = "./3D_new_datasets_training_testing/"
def get_datasets(imgs_dir):
    imgs_test = np.empty((Nimgs_test,frame, height, width))
    for item in range(Nimgs_test):
        ori_img_name1=imgs_dir+'ori_'+str(item+1)+'_1.png'
        ori_img_name2 = imgs_dir + 'ori_' + str(item + 1) + '_2.png'
        ori_img_name3 = imgs_dir + 'ori_' + str(item + 1) + '_3.png'
        ori_img_name4 = imgs_dir + 'ori_' + str(item + 1) + '_4.png'
        img1=Image.open(ori_img_name1)
        img2 = Image.open(ori_img_name2)
        img3 = Image.open(ori_img_name3)
        img4 = Image.open(ori_img_name4)

        img1_=np.array(img1)/np.max(np.array(img1))
        img2_=np.array(img2)/np.max(np.array(img2))
        img3_=np.array(img3)/np.max(np.array(img3))
        img4_=np.array(img4)/np.max(np.array(img4))
        imgs_test[item,0,:,:] = img1_
        imgs_test[item, 1, :, :] = img2_
        imgs_test[item, 2, :, :] = img3_
        imgs_test[item, 3, :, :] = img4_
        #imgs_test=imgs_test/np.max(imgs_test)
        print("imgs1 max: " + str(np.max(img1_)))
        print("imgs1 min: " + str(np.min(img1_)))
        print("imgs2 max: " + str(np.max(img2_)))
        print("imgs2 min: " + str(np.min(img2_)))
        print("imgs3 max: " + str(np.max(img3_)))
        print("imgs3 min: " + str(np.min(img3_)))
        print("imgs4 max: " + str(np.max(img4_)))
        print("imgs4 min: " + str(np.min(img4_)))




    imgs1 = np.reshape(imgs_test, (Nimgs_test, 1,frame, height, width))

    return imgs1
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the testing datasets
imgs_1 = get_datasets(original_imgs_test)
print ("saving test datasets")
print (imgs_1.shape)
print(np.max(imgs_1))
print(np.min(imgs_1))

write_hdf5(imgs_1,dataset_path + "dataset_imgs_train.hdf5")











