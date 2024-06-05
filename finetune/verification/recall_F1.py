import pickle
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.model_selection import train_test_split

from models_siamese import SiameseViT
from torch_utils import get_device, numpy_to_data_loader
from torch_utils import model_fit, model_evaluate

def main():
    device = get_device()

    x_train, y_train = pickle.load(open('dataset/downstream_train_siamese_whole_day_with_status_07_pad_2000_2020_2000.pkl', 'rb'))

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.6, random_state=42, stratify=y_train)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    y_train = y_train[:, np.newaxis]
    y_val = y_val[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    x_val[:, :, 0] = x_val[:, :, 0] / 92
    x_val[:, :, 1] = x_val[:, :, 1] / 49
    x_val[:, :, 2] = x_val[:, :, 2] / 288

    x_test[:, :, 0] = x_test[:, :, 0] / 92
    x_test[:, :, 1] = x_test[:, :, 1] / 49
    x_test[:, :, 2] = x_test[:, :, 2] / 288

    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    val_loader = numpy_to_data_loader(x_val, y_val, y_dtype = torch.float32, batch_size=256, shuffle=False)
    test_loader = numpy_to_data_loader(x_test, y_test, y_dtype = torch.float32, batch_size=256, shuffle=False)

    # model_tag = "CNN_Large"
    model_tag = "large_ViT_last_layer_10%_non_frozen"
    
    pretrain_tag = 'halfyear_100_20000_30epochs_1'
    # pretrain_tag = None

    model = torch.load(f'new/new_models/{model_tag}_verification_pretrain_{pretrain_tag}.pt' if pretrain_tag is not None else f'new/new_models/{model_tag}_model_verification.pt')

    model = model.to(device)

    y_pred = []
    y_true = []
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predictions = torch.round(outputs)  # Round the predictions to 0 or 1
            y_pred.extend(predictions.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Calulate accuracy for binary classification
    acc = np.sum(y_pred == y_true) / len(y_pred)

    # Calculate precision, recall, and F1-score for binary classification
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(model_tag, pretrain_tag)
    
    print(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    
if __name__ == '__main__':
    # set seed
    import numpy as np
    import random
    import torch
    
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Run main
    main()
