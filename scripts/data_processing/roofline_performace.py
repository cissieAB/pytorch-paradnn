import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cycler
import numpy as np


def add_sub_plot(df, target_node, i, ax):
    target_df = df.loc[df['nodes'] == target_node]
    target_tb = target_df.pivot_table(index='batch_size', columns=['layers'], values='tflops', aggfunc='mean')
    # print(target_tb)
    # print(target_tb.shape, target_tb.index, target_tb.columns)

    ax[i].plot(target_tb, '.-')
    ax[i].set_xscale('log', base=2)
    # ax[i].xaxis.tick_top()
    ax[i].set_yscale('log')
    # ax[i].set_xlabel(r'Batch size ($bs$)')
    # legend is in reverse
    if i == 0:
        ax[i].set_ylabel('Performance (TFLOPS)')
        ax[i].legend(target_tb.columns[::-1], title=r"Layers ($l$)", loc="upper left", fontsize=8)
    ax[i].grid(True, which='both', alpha=0.5)
    ax[i].set_title(r"$D_H$=" + str(target_node), x=0.8, y=0.2, pad=-15, size=12)


df = pd.read_csv("../../results/fc_T4_f32.csv")  # <=== change file path

fixed_dtype, fixed_input, fixed_output = "f32", 8000, 200

target_nodes = [32, 256, 512, 1024]

df = df.loc[
    (df['input_type'].str.contains(fixed_dtype)) &
    (df['input_size'] == fixed_input) &
    (df['output_size'] == fixed_output)
]

N = len(target_nodes)

cmap = matplotlib.colormaps['OrRd']
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(1, 0.3, 6)))  # color in reverse to match legend

fig, ax = plt.subplots(1, N, sharey='row', sharex='row', figsize=(10.5, 3))

for i in range(N):
    add_sub_plot(df, target_nodes[i], i, ax)

plt.tight_layout()
# plt.title("T4 f32", loc="right", y=0.5)

plt.savefig(f'roofline_T4_{fixed_dtype}_input_{fixed_input}_output_{fixed_output}.png')
