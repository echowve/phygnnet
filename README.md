# PhyGNNet: Solving spatiotemporal PDEs with Physics-informed Graph Neural Network

This repo is the official implementation of ["PhyGNNet: Solving spatiotemporal PDEs with Physics-informed Graph Neural Network"](https://doi.org/10.1145/3590003.3590029) by Longxiang Jiang, Liyuan Wang, Xinkun Chu, Yonghao Xiao, and Hao Zhang $^{*}$.

## Abstract
Partial differential equations (PDEs) are a common means of describing physical processes. Solving PDEs can obtain simulated results of physical evolution. Currently, the mainstream neural network method is to minimize the loss of PDEs thus constraining neural networks to fit the solution mappings. By the implementation of differentiation, the methods can be divided into PINN methods based on automatic differentiation and other methods based on discrete differentiation. PINN methods rely on automatic backpropagation, and the computation step is time-consuming, for iterative training, the complexity of the neural network and the number of collocation points are limited to a small condition, thus abating accuracy. The discrete differentiation is more efficient in computation, following the regular computational domain assumption. However, in practice, the assumption does not necessarily hold. In this paper, we propose a PhyGNNet method to solve PDEs based on graph neural network and discrete differentiation on irregular domain. Meanwhile, to verify the validity of the method, we solve Burgers equation and conduct a numerical comparison with PINN. The results show that the proposed method performs better both in fit ability and time extrapolation than PINN.

## Approach

<div align="center">
  <img width="100%" src="assets/framework.png">
</div>

## Example

We provide example for solving burgers equation, just create a conda environment with python==3.8
```
conda create -n meshpde python==3.8 && conda activate meshpde
```
then, install the required package with

```
pip install -r requirements.txt
```
and start the training process with

```
python train.py
```
When train finished, to evaluate the trained model and visualize solution results, just run 
```
python test.py
```
and,the results images will be saved in the `images` folder.


## Citation

If you find this repository useful, please consider giving ⭐ or citing:

```
@misc{jiang2023phygnnet,
      title={PhyGNNet: Solving spatiotemporal PDEs with Physics-informed Graph Neural Network}, 
      author={Longxiang Jiang and Liyuan Wang and Xinkun Chu and Yonghao Xiao and Hao Zhang},
      year={2023},
      eprint={2208.04319},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```


