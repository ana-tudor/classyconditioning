import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import types

# This function is used to plot training and testing checkpoints from csvs
# This has two modes. Plotting the same model, and plotting different models. Each has its own command line arguments associated.
# Model_names is the only required argument
# python plot.py --model_names Model_1 # by default --diff_models False
# --model_names This can be several model names that will be plotted
# --save This chooses whether to save the plots (Default False)
# --show This chooses whether to show the plots (Default True)
# --save_dir the directory to save to
# --xkcd Plot in xkcd style
# --single_plot This chooses whether to plot all models on same plot or individually
# --columns This is columns of interest from the csv. Plots vs. Time by default for PPO
# --save_name The name for EVERY Plot generated this run. Suggested to only make one plot at a time
# --plot_name The title for EVERY Plot generated this run. Suggested to only make one plot at a time
# Example: python plot.py --model_names This_Model_1 This_Model_2 --columns 1 0 --single_plot True --save True --show False

# python plot.py --model_names Model_1 --diff_models True
# --model_names This can be several model names that will be plotted
# --save This chooses whether to save the plots (Default False)
# --save_dir the directory to save to
# --show This chooses whether to show the plots (Default True)
# --xkcd Plot in xkcd style
# --single_plot This chooses whether to plot all models on same plot or individually
# --columns list of y-axis column index for each model. For example model1 has reward in column2, model2 has reward in column3, use --columns 2 3
# --time_columns list of x-axis(time) column index for each model
# --save_name The name for EVERY Plot generated this run. Suggested to only make one plot at a time
# --plot_name The title for EVERY Plot generated this run. Suggested to only make one plot at a time
# Example: Plot episode_length_mean for three models in single plot and save to each model folder. Respective column indices listed
# python plot.py --model_names acer_explore1 a2c_test PPO2_initial_test --columns 13 0 0 --time_columns -1 -2 -1 --single_plot True --diff_models True --save True
# Example: Plot episode_reward_mean for three models in single plot and save to each model folder. Respective column indices listed
# python plot.py --model_names acer_explore1 a2c_test PPO2_initial_test --columns 14 1 1 --time_columns -1 -2 -1 --single_plot True --diff_models True --save True

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--xkcd', type=bool, default=False)
    parser.add_argument('--single_plot', type=bool, default=False)
    parser.add_argument('--show', type=bool, default=True)
    parser.add_argument('--diff_models', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default="./plotting/")
    parser.add_argument('--model_names', nargs='*', required=True)
    parser.add_argument('--columns', nargs='*')
    parser.add_argument('--time_columns', nargs='*')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--plot_name', type=str, nargs='*', default=None)
    parser.add_argument('--average_tests', type=bool, default=False)
    parser.add_argument('--avg_x_vals', type=float, nargs='*', default=None)


    args = parser.parse_args()
    columns = args.columns if args.columns else ['reward'] # default reward
    for index in range(len(columns)):
        try:
            columns[index] = int(columns[index]) # convert column indices to ints
        except ValueError:
            pass
    time_columns = args.time_columns if args.time_columns else [-1] # default last column
    for index in range(len(time_columns)):
        try:
            time_columns[index] = int(time_columns[index]) # convert time_columns indices to ints
        except ValueError:
            pass
    save_dir = args.save_dir
    save = args.save or args.save_dir != "" # auto-set to true if save_dir is set
    show = args.show
    xkcd = args.xkcd
    save_name = args.save_name
    plot_name = args.plot_name
    diff_models = args.diff_models
    model_names = args.model_names
    single_plot = args.single_plot
    average_tests = args.average_tests
    avg_x_vals = args.avg_x_vals


    if average_tests and (avg_x_vals is None):
        raise Exception("Please pass in x values for your averaged models")

    if diff_models:
        plot_diff_models(model_names, columns=columns, save=save, show=show,
                        save_dir=save_dir, xkcd=xkcd, single_plot=single_plot,
                        time_columns=time_columns, save_name=save_name, plot_name=plot_name,
                        average_tests=average_tests, average_x_vals=avg_x_vals)
    elif not diff_models:
        plot(model_names, columns=columns, save=save, show=show, save_dir=save_dir,
            xkcd=xkcd, single_plot=single_plot, save_name=save_name, plot_name=plot_name)

def plot(model_names, columns=['reward'], save=False, show=True, save_dir = "./plotting/",
        xkcd=False, single_plot=False, save_name=None, plot_name=None):
    ys = {}
    xs = {}
    cols = []
    values = ["eplenmean", "eprewmean", "fps", "loss/approxkl", "loss/clipfrac",
        "loss/policy_entropy", "loss/policy_loss", "loss/value_loss",
        "misc/explained_variance", "misc/nupdates", "misc/serial_timesteps",
        "misc/time_elapsed", "misc/total_timesteps", "total_timesteps",
        "mean_episode_length", "mean_episode_reward"]
    for column in columns:
        if isinstance(column, int) and column >= 0 and column <= len(values):
            col = values[column]
        elif column == "mean":
            col = values[0]
        elif column == "reward":
            col = values[1]
        elif column not in values:
            raise Exception('invalid column!')
        else:
            col = column
        cols.append(col)

    if xkcd:
        plt.xkcd()
    for model_name in model_names:
        dataset = pd.read_csv("models/" + model_name + "/progress.csv")

        # get all headers in csv (as strings)

        # get the labels. Column 0 is mean, Column 1 is Reward, Column -1 is times
        for col in cols:
            X = np.array(dataset[dataset.columns.values[-1]], dtype='float32')
            y = np.array(dataset[col], dtype='float32')
            xs[(model_name, col)] = X
            ys[(model_name, col)] = y

    if single_plot:
        for col in cols:
            for model_name in model_names:
                y = ys[(model_name, col)]
                x = xs[(model_name, col)]
                plt.plot(x, y, label=model_name)

            plt.legend(loc='best')
            plt.ylabel(col)
            plt.xlabel("Training Iterations")
            if plot_name:
                plt.title(plot_name)
            else:
                title = str(model_names) + ": " + col + " vs. Training Iterations"
                plt.title(title)
            if save:
                if save_name:
                    file_name = save_name
                else:
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
                if plot_name:
                    if len(plot_name) > 1 and not isinstance(plot_name, str):
                        print("run")
                        plot_name = ' '.join(plot_name)
                    plt.title(plot_name)
                else:
                    title = model_name + ": " + col + " vs. Training Iterations"
                    plt.title(title)
                plt.plot(X, y)
                if save:
                    if save_name:
                        file_name = save_name
                    else:
                        file_name = model_name + "_" + str(column)
                    directory = save_dir + model_name + "/plots/"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    plt.savefig(directory + file_name + ".png")
                if show:
                    plt.show()

def plot_diff_models(model_names, columns=[0], save=False, show=True, save_dir = "./plotting/",
                    xkcd=False, single_plot=False, time_columns=[-1], save_name=None, plot_name=None,
                    average_tests=False, average_x_vals=None):
    ys = {}
    xs = {}
    col_names = set()
    assert len(model_names) == len(time_columns), "NOT len(model_names) == len(time_columns)"

    for column in columns:
        assert isinstance(column, int), "NOT isinstance(column, int)"
    cols = columns

    if xkcd:
        plt.xkcd()
    for index in range(len(model_names)):
        model_name = model_names[index]
        time_column = time_columns[index]
        dataset = pd.read_csv("models/" + model_name + "/progress.csv")

        # get all headers in csv (as strings)

        # get the labels. Column 0 is mean, Column 1 is Reward, Column -1 is times for PPO
        # for col in cols:
        col = columns[index]
        X = np.array(dataset[dataset.columns.values[time_column]], dtype='float32')
        y = np.array(dataset[dataset.columns.values[col]], dtype='float32')
        xs[model_name] = X
        ys[model_name] = y
        col_names.add(dataset.columns.values[col])

    if single_plot:
        if average_tests:
            y = []
            for model_name in model_names:
                y.append(np.mean(ys[model_name]))
            x = average_x_vals
            plt.plot(x, y, label=model_name)
        else:
            for model_name in model_names:
                y = ys[model_name]
                x = xs[model_name]
                plt.plot(x, y, label=model_name)

        plt.legend(loc='best')
        plt.ylabel("Columns " + str(col_names))
        plt.xlabel("Training Iterations")
        title = ""
        for model_name in model_names:
            title += model_name + "_"
        title += list(col_names)[0]
        title += "_vs_TrainingIterations"
        if plot_name:
            if len(plot_name) > 1 and not isinstance(plot_name,str):
                print("run")
                plot_name = ' '.join(plot_name)
            plt.title(plot_name)
        else:
            plt.title(title)
        if save:
            if save_name:
                file_name = save_name
            else:
                file_name = title
            for model_name in model_names:
                directory = save_dir + model_name + "/plots/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                plt.savefig(directory + file_name + ".png")
        if show:
            plt.show()

    if not single_plot:
        for model_name in model_names:
            for col_name in col_names:
                X = xs[model_name]
                y = ys[model_name]
                plt.ylabel("Column " + col_name)
                plt.xlabel("Training Iterations")
                title = model_name + "_" + col_name + "_vs_TrainingIterations"
                if plot_name:
                    if len(plot_name) > 1 and not isinstance(plot_name,str):
                        print("run")
                        plot_name = ' '.join(plot_name)
                    plt.title(plot_name)
                else:
                    plt.title(title)
                plt.plot(X, y)
                if save:
                    if save_name:
                        file_name = save_name
                    else:
                        file_name = title
                    directory = save_dir + model_name + "/plots/"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    plt.savefig(directory + file_name + ".png")
                if show:
                    plt.show()

if __name__ == '__main__':
    main()
