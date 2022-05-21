# Depth Estimation using monocular Images
This is a pytorch implementation of [Deeper Depth Predic- tion with Fully Convolutional Residual Networks](https://arxiv.org/abs/1606.00373) .
The implementation is inspired by this paper, but a lot of other experiments are performed with some new architectures and different loss functions.

## Additional contributions to the paper
1. The paper only mention two networks mainly VGG and Resnet but other architectures for multiple Deep Neural networks such as the encoder back- bones notably VGG-16, VGG-19, Resnet-50, EfficientNet, Alexnet were experimented in this implementation, 
2. Used CityScape dataset for disparity estimation which is a slightly different
problem and we tried to justify how it can be solved us- ing the same models and we get good results on validation data for the same. 
3. Designed some very unique and advanced data augmentations and implemented which were really helpful in fast training the models. 
4. Also experimented with transfer learning from indoor to outdoor dataset

## Training Command 
python train_model.py --config configs/config_alexnet.yaml

Model and parameters defined in config file will be used for training pipeline.

## Evaluation Command 
python evaluate_model.py 
    
    Arguments 

    --model-class
    --weights-path
    --output-path
    --batch-size
    --dataset
