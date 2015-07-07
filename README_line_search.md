# modified convert_cifar_dat
Files: ls_convert_cifar_data.cpp
Add following functionality:
 - shuffle data set
 - number of samples which you want to convert from data set into db

# In Memory Data Layer: read db into DRAM as vector <string> 
Files: inmemory_data_layer.cpp, data_layers.hpp.
Data layer which reads all db into vector, so you can:
 - shuffle db, 
 - get/ set current position of cursor 

#integrated batch normlaization layer from Dmytro Mishkin (ducha-aiki)
#see https://github.com/BVLC/caffe/pull/1965 
Files: bn_layer.cpp, bn_layer.cu

# cifar-10 models
Modified cifar10-quick model. Added:
  - relu and dropout layers between ip layers
  - batch normalization before conv. layers 
  
 ==== TODO ===================================
 1. fix ls_convert_cifar_data.cpp
   modify line  train_db->Open(output_folder + "/cifar10_test_" + db_type +"_"+ num_samples, db::NEW);
 2. modify script ls_create_cifar10.sh:
   DBLENGTH=1000\
 3. fix ls_cifar_quick_solver.prototxt
   snapshot: 100000
   
  
 ===== BUILD ================================
 1. copy Makefile.config  and edit it:
    - BLAS := open
    - BLAS_LIB := /opt/OpenBLAS/lib/ 
 2. make all -j
    
 ==== CIFAR-10 example ======================
 1. get data:
    $cd data/cifar-10
    $./get_cifar10.sh
 2. build regular CIFAR-10 train and test lmdb 
    $./examples/cifar10/create_cifar10.sh 
 3. build db for line-search
    $./examples/cifar10/ls_create_cifar10.sh 
 4. train baseline 
    $./examples/cifar10/train_quick.sh 2>&1 | tee ./examples/cifar10/logs/baseline.log 
 5  train: with ALRC and AMC
    $./examples/cifar10/ls_train_quick.sh 2>&1 | tee ./examples/cifar10/logs/ls_alrc_amc.log
    plot graph 
    $./plot_accuracy.py ./examples/cifar10/logs/ls_alrc_amc.log 


