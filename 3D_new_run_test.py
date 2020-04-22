###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import  time 
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy.io as sio
import configparser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')
import sys
sys.setrecursionlimit(4000)


import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
# help_functions.py
#from help_functions import *
# extract_patches.py
#from extract_patches import recompone
#from extract_patches import recompone_overlap
#from extract_patches import paint_border
#from extract_patches import kill_border
#from extract_patches import pred_only_FOV
#from extract_patches import get_data_testing
#from extract_patches import get_data_testing_overlap
# pre_processing.py
#from pre_processing import my_PreProc
import h5py

#group a set of images row per columns
def group_images(data,per_row):
    assert data.shape[0]%per_row==0
    assert (data.shape[1]==1 or data.shape[1]==3)
    data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe,data[k]),axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1,len(all_stripe)):
        totimg = np.concatenate((totimg,all_stripe[i]),axis=0)
    return totimg


#visualize image (as PIL image, NOT as matplotlib!)
def visualize(data,filename):
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    return img





#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('./configuration.txt')
#===========================================
#working directory
path_data = config.get('data paths', 'path_local')

#loading test images 
test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_data_ori = h5py.File(test_imgs_original,'r')


test_imgs_orig=np.array(test_data_ori['image'])
print("***************")
print(np.max(test_imgs_orig))
test_imgs_orig/=np.max(test_imgs_orig)
test_img_orig=test_imgs_orig[10,0,2,:,:]
print ("ori_images size :")
print (test_img_orig.shape)
print('max ori:')
print(np.max(test_img_orig))
print('min ori:')
print(np.min(test_img_orig))

full_img_height = test_imgs_orig.shape[3]
full_img_width = test_imgs_orig.shape[4]



#model name directory
name_experiment = config.get('experiment name', 'name')
path_experiment = './model/'
N_visual = int(config.get('testing settings', 'N_group_visual'))




#================ Run the prediction of the images ==================================
best_last = config.get('testing settings', 'best_last')
#Load the saved model
model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
#Calculate the predictions
start_time=time.time()
predictions = model.predict(test_imgs_orig, batch_size=1, verbose=2)
run_time=time.time()-start_time
print('run time:')
print(run_time)
print ("predicted images size :")
print (predictions[0].shape)
print('max pred:')
print(np.max(predictions[0]))
print('min pred:')
print(np.min(predictions[0]))







#orig_imgs = test_imgs_orig[:,:,0:full_img_height,0:full_img_width]
orig_imgs = test_imgs_orig[:,0,2,0:full_img_height,0:full_img_width]
n_data=orig_imgs.shape[0]
orig_imgs=np.reshape(orig_imgs,(n_data,1,full_img_height,full_img_width))
pred_imgs = predictions[:,:,0:full_img_height,0:full_img_width]
save_path='./exp_test_result/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
print ('preds_shape:' +str(pred_imgs.shape))
pred_save=np.array(pred_imgs)
sio.savemat(save_path+'preds.mat',{'preds':pred_save})
print ("Orig imgs shape: " +str(orig_imgs.shape))
print ("pred imgs shape: " +str(pred_imgs.shape))

# visualize(group_images(orig_imgs,N_visual),save_path+"all_originals")#.show()
# visualize(group_images(pred_imgs,N_visual),save_path+"all_predictions")#.show()
# ##visualize results comparing mask and prediction:
# assert (orig_imgs.shape[0]==pred_imgs.shape[0])
# N_predicted = orig_imgs.shape[0]
# group = N_visual
# assert (N_predicted%group==0)
# for i in range(int(N_predicted/group)):
    # orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
    # #masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
    # pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
    # total_img = np.concatenate((orig_stripe,pred_stripe),axis=0)
    # visualize(total_img,save_path+name_experiment +"_Original_Prediction"+str(i))#.show()

