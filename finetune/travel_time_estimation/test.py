import pickle
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.model_selection import train_test_split
import loratorch as lora
#choose the GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from models_time import ViT, SimpleCNN, SimpleRNN, SimpleLSTM, SimpleGRU
from torch_utils import get_device, numpy_to_data_loader
from torch_utils import model_fit, classification_acc, model_evaluate

def main():
    device = get_device()
    x_train, y_train_orig = pickle.load(open('/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/travel_time_estimation/dataset/train_seek_and_serve_pad_07_2000_2020.pkl', 'rb'))
    print(x_train[-1])
    print(y_train_orig[-1])

    # Create a mask to identify non-padded regions
    mask = np.any(x_train != 0, axis=-1)

    # Calculate the label as the last non-padded time - start time * 5
    last_time = x_train[np.arange(x_train.shape[0]), mask.sum(axis=1) - 1, 2]
    start_time = x_train[:, 0, 2]
    y_train = (last_time - start_time) * 5

    # Reshape the labels to have shape (num_samples, 1)
    y_train = y_train.reshape(-1, 1)

    # Extract the start time from each sequence
    start_times = x_train[:, 0, 2]

    # Tile the start time across the entire sequence length
    tiled_start_times = np.tile(start_times, (x_train.shape[1], 1)).T

    # Replace the last column with the tiled start time only for non-padded regions
    x_train[:, :, 2][mask] = tiled_start_times[mask]

    # Tile the original y_train_orig across the sequence length
    tiled_y_train_orig = np.tile(y_train_orig.reshape(-1, 1), (1, x_train.shape[1])).reshape(x_train.shape[0], x_train.shape[1], 1)

    # Set the tiled y_train_orig to zero for padded regions
    tiled_y_train_orig[~mask] = 0

    # Concatenate the tiled y_train_orig with the x_train array
    x_train_with_y = np.concatenate((x_train, tiled_y_train_orig), axis=-1)

    print(x_train_with_y[-1])
    print(y_train[-1])
    print(x_train_with_y.shape)
    print(y_train.shape)

    # Dump the modified data to the pickle file
    pickle.dump((x_train_with_y, y_train), open('/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/travel_time_estimation/dataset/train_time_estimation_data_07.pkl', 'wb'))

if __name__ == '__main__':
    main()