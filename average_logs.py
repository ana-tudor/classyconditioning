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

    parser.add_argument('--save_name', type=str, default="./models/averaged")
    parser.add_argument('--model_names', nargs='*', required=True)

    args = parser.parse_args()

    model_names = args.model_names
    save_dir = args.save_name

    if len(model_names) != 2:
        raise Exception("Please provide only 2 file paths to models to be merged!")
    else:
        merge_logs(model_names[0], model_names[1],save_dir)



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



if __name__ == '__main__':
    main()
