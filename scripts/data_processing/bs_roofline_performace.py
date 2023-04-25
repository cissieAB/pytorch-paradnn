import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cycler
import numpy as np

flag_text = ["Low utilization", "Medium utilization", "High utilization"]


def add_sub_plot(df, target_node, i, ax):
    target_df = df.loc[df['nodes'] == target_node]
    target_tb = target_df.pivot_table(index='batch_size', columns=['layers'], values='tflops', aggfunc='mean')
    # print(target_tb)
    # print(target_tb.shape, target_tb.index, target_tb.columns)

    ax[i].plot(target_tb, '.-')
    ax[i].set_xscale('log', base=2)
    ax[i].tick_params(axis="x", direction="in", pad=-15)  # inner xtick
    ax[i].set_yscale('log')
    ax[i].set_xlabel(r'Batch size ($bs$)')
    # legend is in reverse
    if i == 0:
        ax[i].set_ylabel('Performance (TFLOPS)')
        ax[i].legend(target_tb.columns[::-1], title=r"Layers ($l$)", loc="upper left", fontsize=8)
    ax[i].grid(True, which='both', alpha=0.5)
    text_str = r"$D_H$=" + str(target_node)
    ax[i].text(x=2048, y=0.01, s=text_str, fontsize="large")
    ax[i].set_title(flag_text[i])


df = pd.read_csv("../../results/fc_A100.csv")

fixed_dtype = "f32"  # <=== change configuration
fixed_input, fixed_output = 8000, 200

# target_nodes = [32, 1024, 8192]  # <=== for A100 FP16
target_nodes = [32, 256, 2048]  # <=== for A100 FP32

df = df.loc[
    (df['input_type'].str.contains(fixed_dtype)) &
    (df['input_size'] == fixed_input) &
    (df['output_size'] == fixed_output)
]

N = len(target_nodes)

cmap = matplotlib.colormaps['OrRd']
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(1, 0.3, 6)))  # color in reverse to match legend

fig, ax = plt.subplots(1, N, sharey='row', sharex='row', figsize=(8.5, 3))

for i in range(N):
    add_sub_plot(df, target_nodes[i], i, ax)

text_str = "FP32" if fixed_dtype == "f32" else "FP16"
ax[-1].text(x=2048, y=0.8, s=text_str, fontsize="xx-large")
plt.tight_layout()

plt.savefig(f'roofline_A100_{fixed_dtype}_input_{fixed_input}_output_{fixed_output}.png')
