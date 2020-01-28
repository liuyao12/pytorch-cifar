# ResNet with a "twist"

* As far as I'm aware, a simple and novel architecture of ConvNets (Convolutional Neural Networks) that is readily applicable to any existing ResNet backbone. PyTorch implementation on CIFAR10.

* The key idea would be hard to come by or justify without viewing ResNet as a partial differential equation (like the heat equation). Traditionally, the standard toolkit for machine learning only includes bits of multi-variable calculus, linear algebra, and statistics, and not so much PDE. This partly explains why ResNet comes on the scene relatively late (2015), and why this enhanced version of ResNet has not been "reinvented" by the DL community.

* Code based off of https://github.com/kuangliu/pytorch-cifar, and the [official PyTorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) 

* Questions and comments shall be greatly appreciated [@liuyao12](https://twitter.com/liuyao12) or liuyao@gmail.com

A quick summary of ConvNets from a Partial Differential Equations (PDE) point of view. For details, see my [notebook](https://observablehq.com/@liuyao12/neural-networks-and-partial-differential-equations) on observable.

neural network | "heat" equation
:----:|:-------:
input layer | initial condition
feed forward | solving the equation
hidden layers | solution at intermediate times
output layer | solution at final time
convolution with 3×3 kernel | differential operator of order ≤ 2
weights | coefficients
boundary handling (padding) | boundary condition
multiple channels/filters/feature_maps | system of (coupled) PDEs
e.g. 16×16×3×3 kernel | 16×16 matrix of differential operators
16×16×1×1 kernel | 16×16 matrix of constants
groups=2 (in Conv2d) | matrix is block diagonal (direct sum of 2 blocks)


Basically, classical ConvNets (ResNets) are **linear PDEs with constant coefficients**, and here I'm simply making it **variable coefficients**, with the variables being polynomials of degree ≤ 1, which should theoretically enable the neural net to learn more ways to deform than diffusion and translation (e.g., rotation and scaling).

# Implementation in PyTorch
See the [notebook](https://github.com/liuyao12/pytorch-cifar/blob/master/cifar10_with_PDE.ipynb).

* 94.32% ResNet34 with twist, in 50 epochs
* 94.72% ResNet50 with twist, in 120 epochs
