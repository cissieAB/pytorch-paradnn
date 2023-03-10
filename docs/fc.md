## Fully connected (FC) neural networks

```bash
Benchmarking the PyTorch FC training performance.

optional arguments:
  -h, --help            show this help message and exit
  -b, --benchmark       benchmark the performance with various hyper-parameters
  --use_gpu {0,1}       1: use the first GPU; 0: use CPU.
  --input_type {f16,bf16,f32,f64}
                        data type of the input X tensors. By default is 'f32'.
  --layers LAYERS       number of hidden layers
  --nodes NODES         the dimension of the hidden layers
  --batch_size BATCH_SIZE
                        batch size
  --input_size INPUT_SIZE
                        the dimension of the input layer
  --output_size OUTPUT_SIZE
                        the dimension of the output layer
```

### Design

- 5 configurable FC model hyperparameters: input_size $D_{in}$, output_size $D_{out}$,
  batch_size $bs$, layers $H$, nodes $D_{H}$.
  - layers: number of hidden layers.
  - nodes: the (identical) size of the hidden layers.
  - Each hidden layer and the input layer is followed by ReLU activation function. The
    output layer does not have any activation.
- Optimizer: Adam (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).
  Use the referred ParaDnn parameters, though they differ from the PyTorch/TF Keras defaults.
- Loss function: `torch.nn.CrossEntropyLoss()`
- Input $X$ tensor is generated from a uniform distribution on the interval [0, 1) based on
  the input tensor type. Output $y$ tensor is randomly generated of torch.int64.
  The input tensor type supports the below 4 options
  ([see official PyTorch tensor type](https://pytorch.org/docs/stable/tensors.html#data-types))


| Bash data type options  |    f16     |      bf16      |     f32     |     f64      |
|:------------------------|:----------:|:--------------:|:-----------:|:------------:|
| PyTorch tensor datatype | torch.half | torch.bfloat16 | torch.float | torch.double |

- Total number of parameters ( $\Phi$ ) in the FC network:
  $\Phi = D_{H} * D_{H} * (H - 1) + D_{H} * (D_{in} + D{out})$
- Throughput is calculated by: $ 6 * \Phi * bs * steps / time$


## References

- [PyTorch profiler tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Utilizing GPU CUDA tensor cores (Automatic Mixed Precision, AMP) with PyTorch](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

