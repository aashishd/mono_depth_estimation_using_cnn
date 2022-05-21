# DeepSense_DL_Project
Deep Learning Project

# Training Command 
python train_model.py --config configs/config_alexnet.yaml

Model and parameters defined in config file will be used for training pipeline. 

# Data pipeline
Download Nyuv2 data from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html just the small labeled pairs of aligned RGB and depth images (1449). You will get a mat file. Run Nyuv2_utils.py in the main branch with specific directory.
My file arrangment like this![image](https://user-images.githubusercontent.com/103913831/164450753-4bdd6d0b-2bb2-4e55-a8ee-1204c9fdf7bd.png)

I just create a test for loader named data_loader_test
