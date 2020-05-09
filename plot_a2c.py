import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Function is callable from command line or from another file
# Model_names is the only required command line argument. List as many models as you want
# show dictates if we want to show the figure (we might want to save-only) (Default True)
# save dictates if we want to save the figure (we might want to show-only) (Default False)
# xkcd toggles xkcd mode because why not (default False)
# save_dir is the directory to save to (overwrite to make save be True)
# Columns is which columns of csv to display. Can be strings, or "mean" or "reward" or indices.
# single_plot decides whether to put all plots on one graph or not. I did some weird things with padding. Default False
# For example you can compare the means across two different models, but if the training times don't line up it gets weird.
# Example calls
# python plot.py --model_names This_Model_1 This_Model_2 --columns reward mean --single_plot True --xkcd True
# python plot.py --model_names This_Model_1 This_Model_2 --columns reward mean --single_plot True --save True --show False
# python plot.py --model_names This_Model_1 This_Model_2 --columns 1 0 --single_plot True --save True --show False
# python plot.py --model_names This_Model_1 --columns 1 --save True --show False

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--xkcd', type=bool, default=False)
    parser.add_argument('--single_plot', type=bool, default=False)
    parser.add_argument('--show', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--model_names', nargs='*', required=True)
    parser.add_argument('--columns', nargs='*')

    args = parser.parse_args()
    columns = args.columns if args.columns else ['reward'] # default reward
    for index in range(len(columns)):
        try:
            columns[index] = int(columns[index]) # convert column indices to ints
        except ValueError:
            pass
    save_dir = args.save_dir
    save = args.save or args.save_dir != "" # auto-set to true if save_dir is set
    show = args.show
    xkcd = args.xkcd
    model_names = args.model_names
    single_plot = args.single_plot
    print(single_plot)

    plot(model_names, columns=columns, save=save, show=show, save_dir=save_dir, xkcd=xkcd, single_plot=single_plot)

def plot(model_names, columns=['mean'], save=False, show=True, save_dir = "./", xkcd=False, single_plot=False):
    longest_X = np.array([])
    ys = {}
    xs = {}
    cols = []
    values = ["eplenmean","eprewmean","explained_variance","fps","nupdates","policy_entropy","total_timesteps","value_loss"]
    for column in columns:
        if isinstance(column, int) and column >= 0 and column <= len(values):
            col = values[column]
        elif column == "mean":
            col = values[0]
        elif column == "reward":
            col = values[1]
        elif column in dataset.columns.values:
            col = column
        else:
            raise Exception('invalid column!')
        cols.append(col)
    if xkcd:
        plt.xkcd()
    for model_name in model_names:
        dataset = pd.read_csv("models/" + model_name + "/progress.csv")
        # get all headers in csv (as strings)

        # get the labels. Column 0 is mean, Column 1 is Reward, Column -1 is times
        for col in cols:
            X = np.array(dataset[values[-2]], dtype='float32')
            y = np.array(dataset[col], dtype='float32')
            xs[(model_name, col)] = X
            ys[(model_name, col)] = y
            if len(X) > len(longest_X):
                longest_X = X

    if single_plot:
        for col in cols:
            for model_name in model_names:
                y = ys[(model_name, col)]
                padded_y = np.ones(len(longest_X)) * np.min(y)
                print(y)
                padded_y[:len(y)] = y
                print(padded_y)
                plt.plot(longest_X, padded_y, label=model_name)

            plt.legend(loc='best')
            plt.ylabel(col)
            plt.xlabel("Training Iterations")
            plt.title(str(model_names) + ": " + col + " vs. Training Iterations")
            if save:
                file_name = model_name + "_" + str(column)
                directory = save_dir + model_name + "/plots/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(directory + file_name + ".png")
            if show:
                plt.show()

    if not single_plot:
        for model_name in model_names:
            for col in cols:
                X = xs[(model_name, col)]
                y = ys[(model_name, col)]
                plt.ylabel(col)
                plt.xlabel("Training Iterations")
                plt.title(model_name + ": " + col + " vs. Training Iterations")
                plt.plot(X, y)
                if save:
                    file_name = model_name + "_" + str(column)
                    directory = save_dir + model_name + "/plots/"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    plt.savefig(directory + file_name + ".png")
                if show:
                    plt.show()

if __name__ == '__main__':
    main()
