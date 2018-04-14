# Proof of Concept of CNN Architectures in Keras

Implementation in Keras 

## 1. Aggregated Residual Networks
- Implemented as POC for CIFAR10 and CIFAR100 datasets i.e. not tuned to provide max accuracy  
- Based on:  
Aggregated Residual Transformations for Deep Neural Networks by Xie et al.  
https://arxiv.org/abs/1611.05431    

Aggregated Residual Networks use the split-transform-merge strategy to define a simple and highly modularized network architecture. The network consists of repeating blocks of same topology followed by transformations and aggregation. In this approach, the authors define a new hyper-parameter called as cardinality which specifies the number of branchings that can occur from the original convolution layer. To each of the branches, transformations (bottleneck convolutions) are performed and all the transformations are merged. To ensure direct gradient flow, a residual connection is added to the merged output of the transformations. It is important to note here that the downsampling is performed by a transformation which involves a convolution of stride (2,2).

### Advantages
* Split-transform-merge strategy proves to be better than going deeper or wider.  
* Comparatively lesser number of hyper parameters to set.  

### Disadvantages
* Split transform merge strategies employ an increased number of convolutions  
* Performance weaker than DenseNet  

## 2. Dense Network  
- Implemented as POC for CIFAR10 and CIFAR100 datasets i.e. not tuned to provide max accuracy  
- Based on:  
Densely Connected Convolutional Networks by Huang et al.  
https://arxiv.org/abs/1608.06993  

In DenseNet, the authors address the vanishing gradient problem in deep CNNs through the novel approach of channel-wise stacking/concatenation of the output of the convolution layers which are of a fixed depth called growth rate, to each of the subsequent convolution layers (each of size equal to the growth rate). At any point, each layer will have the outputs of all the previous layers, hence providing a direct gradient flow to all earlier layers. The dense connections allow the network to be efficiently deep. Between the dense connections are transition layers whose main purpose is to compress and downsample the previous output to the subsequent layers.

### Advantages
* Strong gradient flow due to depth-stacked previous layer outputs
* Parameter and computational efficiency
* Maintains low complexity features

### Disadvantages
* The number of trainable parameters are sensitive to the small perturbations in the hyper
parameters, hence lacking flexibility to extend to the network size to apply to other datasets
* Growth rate is constant throughout the network.
* Static compression factor used

## 3. Spore Network

SporeNet comprises of hybrid composite alternating sub-blocks of dense layers and aggregrated residual layers. The aim of SporeNet architecture is to deploy the ”Divide and Conquer”  strategy. The stacked layers from the output of the dense block are branched depth-wise using the split-transform-merge strategy. We hypothesize that by performing the transformations on the depth-split branches of the output of the dense layers, we are able to operate and learn more on the distributions of the previous layers and by performing the residual connection on the post-transformation concatenation, we are able to preserve the direct gradient flow. Here we perform the downsampling through the dense transition layer which resides between the dense sub-block and the aggregate residual sub-block.

### Features
* Low level features learning by operating on cardinal blocks of dense layers while preserving direct gradient flow through the residual connection
* Dynamic growth rate
* Threshold based compression during transition
* High flexibility i.e. more relevant hyper parameters to tune for going deeper and wider

## 4. Cherry Network  
  
CherryNet comprises of ”cherry” layers. The ”cherry” layers contain dense layer embeddings inside the branches of the aggregate residual layers. The aim of CherryNet architecture is to extract low complexity features of the branches yielded by the split operation of the aggregate residual layer. Using this approach, we would span both deeper (using dense accumulations) and wider (using cardinal branching). Here we perform the downsampling through the dense transition layer which resides between cherry blocks.

### Features

* Allows larger filter depths initially which can be further subjected to deeper learning by using dense layers, while maintaining strong backward connections for direct gradient flow
* Dynamic growth rate
* Threshold based compression during transition
* High flexibility i.e. more relevant hyper parameters to tune for going deeper and wider at the same time
