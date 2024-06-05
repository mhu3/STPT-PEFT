import pickle
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from train_time import ViT  # Assuming your model is defined in trainfrom models_time import ViT, SimpleCNN, SimpleRNN, SimpleLSTM, SimpleGRU
from torch_utils import get_device, numpy_to_data_loader
from torch_utils import model_fit, classification_acc
from torch_utils import numpy_to_data_loader, model_evaluate

from sklearn.model_selection import train_test_split

#choose the GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def mean_absolute_percentage_error(y_pred, y_true, threshold=1):
    """Compute Mean Absolute Percentage Error."""
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    
    # Threshold the true values
    y_true = torch.where(y_true < threshold, torch.tensor(threshold, device=y_true.device), y_true)
    
    mape = torch.mean(torch.abs((y_pred - y_true) / y_true)) * 100
    return mape.item()  # Return the MAPE as a Python float


def evaluate(model_path, threshold=1):

    device = get_device()
    # Load the test data
    x_train, y_train = pickle.load(open('/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/travel_time_estimation/dataset/train_time_estimation_data_07.pkl', 'rb'))

    # # normalize the data
    x_train[:, :, 0] = x_train[:, :, 0] / 92
    x_train[:, :, 1] = x_train[:, :, 1] / 49
    x_train[:, :, 2] = x_train[:, :, 2] / 288

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    
    # Load the trained model
    model = torch.load(model_path)
    model.eval()

    model = model.to(device)
    
    # Create a data loader for the test data
    test_loader = numpy_to_data_loader(x_test, y_test, y_dtype=torch.float32, batch_size=128, shuffle=False)
    
    # Evaluate the model on the test data
    with torch.no_grad():
        y_true = []
        y_pred = []
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)  # Move the input batch to the device
            y_pred_batch = model(x_batch)
            y_true.extend(y_batch.cpu().numpy())  # Move the tensor to CPU before converting to numpy
            y_pred.extend(y_pred_batch.cpu().numpy())  # Move the tensor to CPU before converting to numpy

        print(y_true)
        print(y_pred)

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(torch.tensor(y_pred), torch.tensor(y_true), threshold=threshold)

        print(f"Test Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.4f}")
        
        # Analyze the results
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        print(f"\nError Analysis:")
        print(f"Max Absolute Error: {np.max(abs_error):.4f}")
        print(f"Min Absolute Error: {np.min(abs_error):.4f}")
        print(f"Mean Absolute Error: {np.mean(abs_error):.4f}")
        print(f"Median Absolute Error: {np.median(abs_error):.4f}")
        
        # Analyze instances with large errors
        large_error_threshold = 100  # Adjust this value based on your data
        large_error_indices = np.where(abs_error > large_error_threshold)[0]
        
        print(f"\nInstances with Large Errors (> {large_error_threshold}):")
        for idx in large_error_indices:
            print(f"Index: {idx}, True Value: {y_true[idx]:.4f}, Predicted Value: {y_pred[idx]:.4f}, Error: {error[idx]:.4f}")

if __name__ == '__main__':
    model_path = '/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/travel_time_estimation/results/models/time_large_ViT_LL_lr0.001__non_frozen_verification_pretrain_halfyear_100_20000_30epochs_1.pt'
    
    evaluate(model_path)