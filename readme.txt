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


Citation:
Dongdong Hao, Song Ding, Linwei Qiu, Yisong Lv, Baowei Fei, Yueqi Zhu, Binjie Qin, Sequential vessel segmentation via deep channel attention network, Neural Networks, preprint, 2020.
