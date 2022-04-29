Citation:
Dongdong Hao, Song Ding, Linwei Qiu, Yisong Lv, Baowei Fei, Yueqi Zhu, Binjie Qin, Sequential vessel segmentation via deep channel attention network, Neural Networks, preprint, 2020.

Abstract: Accurately segmenting contrast-filled vessels from X-ray coronary angiography (XCA) image sequence is an essential step for the diagnosis and therapy of coronary artery disease. However, developing automatic vessel segmentation is particularly challenging due to the overlapping structures, low contrast and the presence of complex and dynamic background artifacts in XCA images. This paper develops a novel encoder-decoder deep network architecture which exploits the several contextual frames of 2D+t sequential images in a sliding window centered at current frame to segment 2D vessel masks from the current frame. The architecture is equipped with temporal-spatial feature extraction in encoder stage, feature fusion in skip connection layers and channel attention mechanism in decoder stage. In the encoder stage, a series of 3D convolutional layers are employed to hierarchically extract temporal-spatial features. Skip connection layers subsequently fuse the temporal-spatial feature maps and deliver them to the corresponding decoder stages. To efficiently discriminate vessel features from the complex and noisy backgrounds in the XCA images, the decoder stage effectively utilizes channel attention blocks to refine the intermediate feature maps from skip connection layers for subsequently decoding the refined features in 2D ways to produce the segmented vessel masks. Furthermore, Dice loss function is implemented to train the proposed deep network in order to tackle the class imbalance problem in the XCA data due to the wide distribution of complex background artifacts. Extensive experiments by comparing our method with other state-of-the-art algorithms demonstrate the proposed method's superior performance over other methods in terms of the quantitative metrics and visual validation. 





1. Environment：
keras=2.1.6
tensorflow=1.2.1
python=3.6.0

2. Document description：
./model   /* The model structures and weight files obtained after training 
./src       /*  The source code required for training
./data     /* training dataset, validation dataset, will be updating soon......
run_training.py and 3D_new_run_test.py      /* Starting code for training and testing
prepare_dataset_hdf5_for_3D.py  /* the source code for generating training dataset and test data in hdf5 format

3. source code for training：
python3 ./src/vessel_NN_training.py

4.  source code for testing：
python3 ./3D_new_run_test.py  

