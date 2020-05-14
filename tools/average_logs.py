import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Function averages log files of different models
# python average_logs.py --model_names model_1 model_2
# --across_time average across time (Default False)
# --save_name the directory and name of the new csv
# Example: python average_logs.py --model_names 50_epopt_ckpt05 50_epopt_ckpt10 --save_name 50_epopt_avg_ckpts --across_time True

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')

    parser.add_argument('--save_name', type=str, default="./models/averaged")
    parser.add_argument('--model_names', nargs='*', required=True)
    parser.add_argument('--across_time', nargs='*', default=False)

    args = parser.parse_args()

    model_names = args.model_names
    save_dir = args.save_name
    across_time = args.across_time

    if len(model_names) < 2:
        raise Exception("Please provide at least 2 file paths to models to be merged!")
    elif not across_time:
        merge_logs(model_names[0], model_names[1],save_dir)
    else:
        merge_logs_across_time(model_names, save_dir)

def merge_logs(model1, model2, save_dir):
    data1 = pd.read_csv("models/" + model1 + "/progress.csv")
    data2 = pd.read_csv("models/" + model2 + "/progress.csv")

    #Raise exception if data are not same g
    if data1.shape != data2.shape:
        raise Exception("Model logging files must be same size!")
    if data1.columns[0] != data2.columns[0]:
        raise Exception("Model logging files must have same columns")

    merged = data1.copy()

    for i in range(len(data1.columns) - 1):
        col = data1.columns[i]
        merged[col] = np.mean([data1[col], data2[col]], axis=0)

    if save_dir:
        merged.to_csv("models/" + save_dir + "/progress.csv")
    else:
        raise Exception("Could not save, please provide save directory")

def merge_logs_across_time(model_names, save_dir):
    cols = ["test_results", "train_iters"]
    xs = [int(name[-2:]) for name in model_names]

    ys = []
    for model_name in model_names:
        data = pd.read_csv("models/" + model_name + "/progress.csv")
        ys.append(np.mean(data['eprewmean']))

    df = pd.DataFrame(data=np.array([ys, xs]).T, columns=cols)
    if save_dir:
        df.to_csv("models/" + save_dir + "/progress.csv")
    else:
        raise Exception("Could not save, please provide save directory")

if __name__ == '__main__':
    main()
