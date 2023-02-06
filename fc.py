"""
Fully-connected NN models.

mailto: xmei@jlab.org
"""
import socket
import argparse
import gc
import time
from collections import OrderedDict

import torch
import torch.nn as nn

CSV_HEADER_STR = "device,input_type,layers,nodes,batch_size,input_size,output_size,#params,duration,tflops"


# 30 warmup steps and 100 steps when timing
class BenchmarkSteps:
    WARMUP = 300
    ITER = 100


class BenchmarkParamConfigs:
    """
    Hyperparameters are taken from Table 2 in https://yuemmawang.github.io/publications/wang-mlsys2020.pdf.
    """

    class Layers:
        MIN = 4
        MAX = 128

    class Nodes:
        MIN = 32
        MAX = 8192

    class BatchSize:
        MIN = 64
        MAX = 16384

    class InputSize:
        MIN = 2000
        EX_MAX = 10000
        INC = 2000

    class OutputSize:
        MIN = 200
        EX_MAX = 1200
        INC = 200

    class AdamParams:
        EPS = 1e-8  # 1e-7 for TensorFlow


class HyperParams:
    """
    Build the NN model hyperparameters.
    Default values are set upon initialization.
    """

    def __init__(self):
        # configurable to cli inputs
        self.layers = BenchmarkParamConfigs.Layers.MIN
        self.nodes = BenchmarkParamConfigs.Nodes.MIN
        self.batch_size = BenchmarkParamConfigs.BatchSize.MIN
        self.input_size = BenchmarkParamConfigs.InputSize.MIN
        self.output_size = BenchmarkParamConfigs.OutputSize.MIN
        self.input_type = 'f32'
        self.use_gpu = False

        self.warmup_steps = BenchmarkSteps.WARMUP
        self.steps = BenchmarkSteps.ITER

    def get_num_params(self):
        return self.nodes ** 2 * (self.layers - 1) + self.nodes * (self.input_size + self.output_size)

    def get_ops(self):
        """
        Get the estimated FLOPs of the network.
        """
        return self.steps * self.batch_size * \
               self.get_num_params() * 6  # FWD OPs: 2x(#NN params); BKP Ops: ~ 2* FWD OPs

    def update_params_from_cli(self, cli_args):
        """Update hyperparameters from cli inputs"""
        if cli_args.layers:
            self.layers = cli_args.layers
        if cli_args.nodes:
            self.nodes = cli_args.nodes
        if cli_args.batch_size:
            self.batch_size = cli_args.batch_size
        if cli_args.input_size:
            self.input_size = cli_args.input_size
        if cli_args.output_size:
            self.output_size = cli_args.output_size
        if cli_args.input_type:
            self.input_type = cli_args.input_type
        self.update_device_from_cli(cli_args)

    def update_device_from_cli(self, cli_args):
        if cli_args.use_gpu and cli_args.use_gpu == 1:
            self.use_gpu = True

    def set_param_bs(self, bs):
        self.batch_size = bs

    def set_param_input_size(self, size):
        self.input_size = size

    def set_param_output_size(self, size):
        self.output_size = size

    def set_param_input_type(self, data_type):
        self.input_type = data_type

    def log_param_info(self):
        print("Hyper-parameter configuration: ")
        print(f"  input_size: {self.input_size}")
        print(f"  output_size: {self.output_size}")
        print(f"  layers: {self.layers}")
        print(f"  nodes: {self.nodes}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  device: {get_device_name(self.use_gpu)}")
        print(f"  input_tensor_type: {self.input_type}")
        print(f"  #params: {self.get_num_params()}")
        print("\n")


class FC(nn.Module):
    """
    Define the FC network structure.

    The input layer and the hidden layers are followed by the ReLU activation function, while
    the output layer does not have activation.
    """

    def __init__(self, input_size, hidden_size, output_size, depth, act=torch.nn.ReLU):
        super(FC, self).__init__()

        layers = [('input', torch.nn.Linear(input_size, hidden_size)), ('input_activation', act())]  # input layer
        for i in range(depth):  # hidden layers
            layers.append(('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size)))
            layers.append(('activation_%d' % i, act()))
        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))  # output layer

        layer_dict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layer_dict)

    def forward(self, x):
        out = self.layers(x)
        return out


class Net:

    def __init__(self, params):
        self.params = params

        self.device = torch.device("cuda") if (torch.cuda.is_available() and self.params.use_gpu) \
            else torch.device("cpu")

        self.model = self._get_model(self.params.input_type)

        self.optimizer = torch.optim.Adam(self.model.parameters(), eps=BenchmarkParamConfigs.AdamParams.EPS)
        self.loss = nn.CrossEntropyLoss()

    def _get_model(self, input_tensor_type):
        model = FC(self.params.input_size, self.params.nodes, self.params.output_size, self.params.layers
                   ).to(self.device)
        # Pytorch tensor types: https://pytorch.org/docs/stable/tensors.html
        if input_tensor_type == 'f64':
            model = model.double()
        elif input_tensor_type == 'f16':
            model = model.half()
        elif input_tensor_type == 'bf16':
            model = model.bfloat16()
        return model

    def get_inputs(self, input_tensor_type):
        """
        Get the input tensors based on desired data types. They live on device.
        """
        # generate random inputs from a uniform distribution on the interval [0, 1)
        if input_tensor_type == 'f32':
            return torch.rand((self.params.batch_size, self.params.input_size), device=self.device)
        elif input_tensor_type == 'f64':
            return torch.rand((self.params.batch_size, self.params.input_size), dtype=torch.double, device=self.device)
        elif input_tensor_type == 'f16':
            return torch.rand((self.params.batch_size, self.params.input_size), dtype=torch.half, device=self.device)
        elif input_tensor_type == 'bf16':
            return torch.rand((self.params.batch_size, self.params.input_size), dtype=torch.bfloat16,
                              device=self.device)
        # TODO: add tf 32 for A100

    def get_outputs(self):
        """
        Get the output tensor (living on device) of int64/long.
        """
        return torch.randint(high=self.params.output_size, size=(self.params.batch_size,), device=self.device)

    def train(self, num_steps, timing_flag):
        # Currently, AutocastCPU only support Bfloat16 as the autocast_cpu_dtype
        for i in range(num_steps):
            pred = self.model(self.get_inputs(self.params.input_type))
            cur_loss = self.loss(pred, self.get_outputs())
            # print(f"step={i + 1}, timing={timing_flag}, "
            #       f"pred.dtype={pred.dtype}, loss.dtype={cur_loss.dtype}, loss={cur_loss.item()}")

            self.optimizer.zero_grad()
            cur_loss.backward()
            self.optimizer.step()

    def train_amp(self):
        # TODO
        pass

    def benchmark(self):
        """
        The main benchmarking process.
        """
        # warm-up steps
        self.train(self.params.warmup_steps, False)

        start = self.start_timer()
        self.train(self.params.steps, True)
        stop = self.end_timer()

        duration = stop - start  # in seconds
        tflops = float(self.params.get_ops()) / 1e12 / duration

        return duration, tflops

    def start_timer(self):
        gc.collect()
        if str(self.device) == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        return time.time()

    def end_timer(self):
        if str(self.device) == "cuda":
            torch.cuda.synchronize()
            # comment out before benchmark
            # might be smaller than 4 * num_params
            # print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
        return time.time()


def get_args(parser):
    # tutorial here: https://docs.python.org/3/howto/argparse.html
    parser.add_argument("-b", "--benchmark", action="store_true",
                        help="benchmark the FC performance with various hyper-parameters")

    parser.add_argument("--use_gpu", type=int, choices=[0, 1], help="1: use the first GPU; 0: use CPU.")
    parser.add_argument("--input_type", choices=['f16', 'bf16', 'f32', 'f64'], default='f32',
                        help="data type of the input X tensors. The default value is 'f32'.")

    parser.add_argument("--layers", type=int, help="number of hidden layers")
    parser.add_argument("--nodes", type=int, help="the dimension of the hidden layers")

    # params loops in the benchmark wrapper
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--input_size", type=int, help="the dimension of the input layer")
    parser.add_argument("--output_size", type=int, help="the dimension of the output layer")

    return parser.parse_args()


def get_device_name(use_gpu):
    if torch.cuda.is_available() and use_gpu:
        return torch.cuda.get_device_name()
    else:
        return socket.gethostname() + "-cpu"


def benchmark_wrapper(hyperparams):
    """
    Benchmark on the batch_size, input_size, output_size domain.

    Other domain will be passed by the bash script.
    """
    print(CSV_HEADER_STR)

    device_str = get_device_name(hyperparams.use_gpu)

    bs = BenchmarkParamConfigs.BatchSize.MIN
    while bs <= BenchmarkParamConfigs.BatchSize.MAX:
        for input_size in range(BenchmarkParamConfigs.InputSize.MIN,
                                BenchmarkParamConfigs.InputSize.EX_MAX,
                                BenchmarkParamConfigs.InputSize.INC):
            for output_size in range(BenchmarkParamConfigs.OutputSize.MIN,
                                     BenchmarkParamConfigs.OutputSize.EX_MAX,
                                     BenchmarkParamConfigs.OutputSize.INC):
                hyperparams.set_param_bs(bs)
                hyperparams.set_param_input_size(input_size)
                hyperparams.set_param_output_size(output_size)

                net = Net(hyperparams)
                duration, tflops = net.benchmark()
                # "device, input_type, layers, nodes, batch_size, input_size, output_size, #params, duration, tflops"
                print(f"{device_str}, {hyperparams.input_type}, {hyperparams.layers}, {hyperparams.nodes}, "
                      f"{hyperparams.batch_size}, {hyperparams.input_size}, {hyperparams.output_size}, "
                      f"{hyperparams.get_num_params()}, {duration}, {tflops}")
        bs *= 2


if __name__ == "__main__":
    params = HyperParams()

    parser = argparse.ArgumentParser(description="Benchmarking the PyTorch FC training performance.")
    args = get_args(parser)
    params.update_params_from_cli(args)

    if args.benchmark:
        benchmark_wrapper(params)
    else:  # single run mode
        print(f"Torch version {torch.__version__}\n")
        params.log_param_info()
        net = Net(params)
        duration, tflops = net.benchmark()
        print(f"Duration: {duration} seconds, TFLOPS: {tflops}\n\n")
