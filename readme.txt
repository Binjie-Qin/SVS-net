1.依赖环境：
keras=2.1.6
tensorflow=1.2.1
python=3.6.0

2.各文件说明：
./model/* 为训练得到的模型结构和权重文件
./src/* 为训练所需代码
./data/* 为训练集和测试集数据 后续整理逐渐完善
run_training.py 和3D_new_run_test.py 为训练和测试启动代码
prepare_dataset_hdf5_for_3D.py 为生成hdf5格式的训练集和测试集代码

.训练方法：
python3 ./src/vessel_NN_training.py

.测试用法：
python3 ./3D_new_run_test.py  


