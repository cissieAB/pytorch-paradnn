
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import FCParamConfigs

INPUT_SIZE_MIN = FCParamConfigs.InputSize.MIN
INPUT_SIZE_MAX = FCParamConfigs.InputSize.EX_MAX
INPUT_SIZE_INC = FCParamConfigs.InputSize.INC

OUTPUT_SIZE_MIN = FCParamConfigs.OutputSize.MIN
OUTPUT_SIZE_MAX = FCParamConfigs.OutputSize.EX_MAX
OUTPUT_SIZE_INC = FCParamConfigs.OutputSize.INC

LAYERS_MIN = FCParamConfigs.Layers.MIN
LAYERS_MAX = FCParamConfigs.Layers.MAX

NODES_MIN = FCParamConfigs.Nodes.MIN
NODES_MAX = FCParamConfigs.Nodes.MAX

# creating dataframe
df = pd.read_csv('benchmark_T4.csv')  # TODO: change path into an input
"""
       device  input_type   layers  ...   #params   duration    tflops
0    Tesla T4         f32        4  ...     73472   0.188689  0.014952
1    Tesla T4         f32        4  ...     79872   0.272968  0.011236
...

device          object
input_type      object
layers           int64
nodes            int64
batch_size       int64
input_size       int64
output_size      int64
#params          int64
duration       float64
tflops         float64
"""

# print(df.dtypes)
# print(df.shape)
# df.info(verbose=True)


def plot_performance_vs_bs_by_layers_n_nodes_n_input_n_output(layers, nodes):

    num_row_plots = (INPUT_SIZE_MAX + INPUT_SIZE_INC - INPUT_SIZE_MIN) // INPUT_SIZE_INC
    num_col_plots = (OUTPUT_SIZE_MAX + OUTPUT_SIZE_INC - OUTPUT_SIZE_MIN) // OUTPUT_SIZE_INC

    fig, ax = plt.subplots(num_row_plots, num_col_plots, figsize=(18, 15), sharey=True)
    fig.text(0.1, 0.92, f'TFLOPS vs batch_size (layers={layers}, nodes={nodes})')
    for input_size in range(INPUT_SIZE_MIN, INPUT_SIZE_MAX + INPUT_SIZE_INC, INPUT_SIZE_INC):
        for output_size in range(OUTPUT_SIZE_MIN, OUTPUT_SIZE_MAX + OUTPUT_SIZE_INC, OUTPUT_SIZE_INC):
            title_str = f'input_{input_size}_output_{output_size}'
            target_df = df.loc[
                (df['layers'] == layers) & (df['nodes'] == nodes)
                & (df['input_size'] == input_size)
                & (df['output_size'] == output_size)
                # & df['input_type'].str.contains('f32')
                ]
            # print(target_df)

            target_tb = target_df.pivot_table(index='batch_size', columns=['input_type'], values='tflops',
                                              aggfunc='mean')
            # print(target_tb)

            cur_ax = ax[input_size // INPUT_SIZE_INC - 1, output_size // OUTPUT_SIZE_INC - 1]
            target_tb.plot(
                # logx=True,
                style=['+-', '*-'],
                title=title_str, legend=False,
                ax=cur_ax
            )  # https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.plot.html
            cur_ax.set(xlabel=None)
    plt.legend(loc='best')
    # handles, labels = ax[-1, -1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center')
    fig.savefig(f'T4/linear/performance_vs_bs_l_{layers}_n_{nodes}.png')


def plot_performance_vs_bs_n_nodes(fixed_layers, fixed_input, fixed_output, fixed_dtype):
    target_df = df.loc[
        (df['layers'] == fixed_layers)
        & (df['input_size'] == fixed_input)
        & (df['output_size'] == fixed_output)
        & (df['input_type'].str.contains(fixed_dtype))
        ]
    notes_str = f'{fixed_dtype}_l_{fixed_layers}_input_{fixed_input}_output_{fixed_output}'
    print(notes_str)

    target_tb = target_df.pivot_table(index='batch_size', columns=['nodes'], values='tflops',
                                      aggfunc='mean')
    print(target_tb)

    plt.figure()
    plt.clf()
    plt.imshow(target_tb)
    plt.colorbar()
    plt.xticks(range(target_tb.shape[1]), np.log2(target_tb.columns).astype(int))
    plt.xlabel('Log2(#nodes)')
    plt.yticks(range(target_tb.shape[0]), np.log2(target_tb.index).astype(int))
    plt.ylabel('Log2(batch size)')
    plt.title(f'TFLOPS ({notes_str})')
    plt.savefig(f'heatmap_perf_{notes_str}.png')


"""
nodes=32, layers=4-128, f16, f32 all data
"""
num_layers = LAYERS_MIN
while num_layers <= LAYERS_MAX:
    plot_performance_vs_bs_by_layers_n_nodes_n_input_n_output(num_layers, 8)
    num_layers *= 2

"""
l=8, input=8000, output=1000, f16 and f32
"""
# plot_performance_vs_bs_n_nodes(4, 8000, 1000, 'f32')
plot_performance_vs_bs_n_nodes(8, 8000, 1000, 'f32')
plot_performance_vs_bs_n_nodes(4, 8000, 1000, 'f16')  # peak: 32.489109, bs=16384, n=2048
plot_performance_vs_bs_n_nodes(8, 8000, 1000, 'f16')  # peak: 32.576202, bs=16384, n=2048
