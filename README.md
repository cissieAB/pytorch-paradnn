# Benchmarking the parameterized FC/CNN with PyTorch

This work is a reproduction of [ParaDnn](https://github.com/Emma926/paradnn)
(originally implemented with TensorFlow), which is a benchmark set of hyper-parameterized
DNNs. We rewrite the benchmarks with Pytorch.
The bibtex information of the original paper is as below.

```bibtex
@inproceedings{wang2020systematic,
  title={A Systematic Methodology for Analysis of Deep Learning Hardware and Software Platforms},
  author={Wang, Yu Emma and Wei, Gu-Yeon and Brooks, David},
  booktitle={The 3rd Conference on Machine Learning and Systems (MLSys)},
  year={2020}
}
```

Currently, we include parametrized fully-connected (FC) and convolutional neural network
(CNN) models only.

## Model designs
- [Fully-connected neural networks (FC)](./docs/fc.md)
- [Convolutional neural networks (CNN)](./docs/cnn.md)

## Platforms
We test our the benchmark on two JLab [ifarm](https://scicomp.jlab.org/scicomp/home) GPUs: T4 and A100 PCIe.
The hardware configurations are listed as below.

## TODOs
- [ ] Platform configurations
- [ ] A100's tf32 data type
