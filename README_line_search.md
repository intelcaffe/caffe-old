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


