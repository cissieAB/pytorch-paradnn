import pandas as pd
import matplotlib.pyplot as plt
import sys, argparse


def get_target_line_from_file(df, data_type):
    target_id = df.loc[df['input_type'] == data_type]['tflops'].idxmax()
    # print(target_id, df.iloc[target_id])
    return df.iloc[target_id].to_dict()


def get_dtypes_from_df(df):
   """Return the data types in the df in a list"""
   return df['input_type'].unique()


def print_dict(dic):
    """A helper function to print target line to csv format"""
    value_list = list(dic.values())
    for v in value_list[:-2]:
        print(v, end=",")
    print(value_list[-1])


def process_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_csv", required=True,
                    help="Name of the input csv file")
    args = vars(ap.parse_args())
    return args


def get_title_str(dic):
    device_str = 'T4' if dic["device"] == "Tesla T4" else "A100"
    return f'{device_str}_{dic["input_type"]}_l_{dic["layers"]}_input_{dic["input_size"]}_output_{dic["output_size"]}'


def heatmap_x_node_y_bs(df, fixed_layers, fixed_input, fixed_output, fixed_dtype, title_str=''):
    """
    Plot the heatmap performance with given layers, input, output, datatype. Title string is optional.
    """
    target_df = df.loc[
        (df['layers'] == fixed_layers)
        & (df['input_size'] == fixed_input)
        & (df['output_size'] == fixed_output)
        & (df['input_type'].str.contains(fixed_dtype))
        ]

    if not title_str:
        title_str = f'{fixed_dtype}_l_{fixed_layers}_input_{fixed_input}_output_{fixed_output}'

    target_tb = target_df.pivot_table(index='batch_size', columns=['nodes'], values='tflops',
                                      aggfunc='mean')
    # print(target_tb)

    plt.figure()
    plt.clf()
    # cmap reference: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    plt.imshow(target_tb, cmap='OrRd')
    plt.colorbar()

    # plt.xticks(range(target_tb.shape[1]), np.log2(target_tb.columns).astype(int))
    plt.xticks(range(target_tb.shape[1]), target_tb.columns.astype(int))
    plt.xlabel('Log2(#nodes)')
    plt.yticks(range(target_tb.shape[0]), target_tb.index.astype(int))
    plt.ylabel('Log2(batch size)')
    plt.title(f'TFLOPS ({title_str})')
    plt.savefig(f'heatmap_perf_{title_str}.png')


def main(argv):
    args = process_args(argv)

    df = pd.read_csv(args['input_csv'])
    # df.info(verbose=True)

    data_types = get_dtypes_from_df(df)

    for dt in data_types:
        # print(dt)
        target_dict = get_target_line_from_file(df, dt)
        print_dict(target_dict)
        title = get_title_str(target_dict)
        heatmap_x_node_y_bs(df,
                           target_dict['layers'], target_dict['input_size'],
                           target_dict['output_size'], target_dict['input_type'],
                           title)


if __name__ == "__main__":
    main(sys.argv[1:])
