import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import loratorch as lora

import numpy as np
from scipy.spatial.distance import cdist

from models_classification import  ViT
from torch_utils import get_device, numpy_to_data_loader
from torch_utils import model_fit, classification_acc, model_evaluate


def main():
    device = get_device()

    # Load pre-processed data
    # train_class the first  4days
    # val_class the last 1day
    x, y = pickle.load(open('dataset/train_class_whole_day_with_status_classification.pkl', 'rb'))

    x_train = x[400:480]
    y_train = y[400:480]
    x_val = x[480:560]
    y_val = y[480:560]
    x_test = x[560:720]
    y_test = y[560:720]

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    # prepare data
    train_loader = numpy_to_data_loader(x_train, y_train, batch_size=16, shuffle=True)
    val_loader = numpy_to_data_loader(x_val, y_val, batch_size=512, shuffle=False)
    test_loader = numpy_to_data_loader(x_test, y_test, batch_size=512, shuffle=False)

    # Train model
    learning_rates = [5e-4, 1e-3, 5e-5, 1e-4]

    for lr in learning_rates:
        print(f"Running with learning rate: {lr}")

        model = ViT(  
                            input_size=x_train.shape[2],
                            num_heads = 16, 
                            num_layers = 8, 
                            dropout=0.1, 
                            use_lora=False, 
                            r=8, 
                            adapter_size=64,
                            use_houlsby=True, 
                            use_weighted_layer_sum=False,
                            ensembler_hidden_size=64,
                            apply_knowledge_ensembler=False,
                            enable_adapterbias=False,
                            num_prompt_tokens=0,
                            num_prefix_tokens=0,
                            enable_prefix_tuning=False,
                            enable_prompt_tuning=False)

        # model = SimpleCNN(input_size=x_train.shape[2])

        mode = 'houlsby'  # Choose from 'lora', 'frozen', 'non_frozen', or 'houlsby', or 'bitfit', or 'prompt'

        if mode == 'lora':
            lora.mark_only_lora_as_trainable(model)
            for name, param in model.named_parameters():
                # if 'layer_weights' in name or 'fc' in name or "input_layer" in name or "pos_embedding" in name or "cls_token" in name or "knowledge_ensembler" in name:
                if 'layer_weights' in name or 'fc' in name or "knowledge_ensembler" in name or "pos_embedding" in name:
                    param.requires_grad = True
        elif mode == 'frozen':
            for name, param in model.named_parameters():
                if 'layer_weights' in name or 'fc' in name or "knowledge_ensembler" in name or "pos_embedding" in name:    
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif mode == 'prompt':
            for name, param in model.named_parameters():
                if 'layer_weights' in name or 'fc' in name or "prompt" in name or "pos_embedding" in name or "knowledge_ensembler" in name: #or "input_layer" in name or "pos_embedding" in name or "cls_token" in name or "prompts" in name or "knowledge_ensembler" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False   
        elif mode == 'prefix':
            for name, param in model.named_parameters():
                if 'layer_weights' in name or 'fc' in name or "prefix" in name or "pos_embedding" in name or "knowledge_ensembler" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False          
        elif mode == 'non_frozen':
                for name, param in model.named_parameters():
                    if "prompt_embedding" not in name and "prefix_embedding" not in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        elif mode == 'houlsby':
            for name, param in model.named_parameters():
                if 'down_project' in name or 'up_project' in name or 'layer_weights' in name or 'fc' in name or "prompt_pos_embedding" in name or "knowledge_ensembler" in name or "pos_embedding" in name: #or "input_layer" in name or "pos_embedding" in name or "cls_token" in name or "knowledge_ensembler" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif mode == 'bitfit':
            for name, param in model.named_parameters():
                # This will enable gradients only for parameters in 'adapter', 'fc', 'bias', and 'layer_weights',
                # but exclude the 'input_layer' biases.
                if ('adapter' in name or 'fc' in name or 'bias' in name or 'layer_weights' in name or "pos_embedding" in name) and 'input_layer.bias' not in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif mode == 'adapterbias':
            for name, param in model.named_parameters():
                # This will enable gradients only for parameters in 'adapter', 'fc', 'bias', and 'layer_weights',
                # but exclude the 'input_layer' biases.
                if 'adapter' in name or 'fc' in name or 'layer_weights' in name or "pos_embedding" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        else:
            raise ValueError("Invalid mode. Choose from 'lora', 'frozen', or 'non_frozen', or 'houlsby', or 'bitfit'.")

        num_gpus = torch.cuda.device_count()
        print("Number of available GPUs: ", num_gpus)

        if num_gpus > 1:
            model = nn.DataParallel(model)

        # Define the tag for the model
        model_tag = f"classification_large_ViT_{lr}_pe"
        # model_tag = f"classification_large_CNN_lr{lr}"

        # Define the tag for the pre-trained model (or set it to None if no pre-trained model is available)
        pretrain_tag = "halfyear_100_20000_30epochs_1"  # Change this value or set it to None
        # pretrain_tag = None

        if pretrain_tag is not None:
            pre_trained_model = torch.load(f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/pretrain/pretrainmodels/large_model_ViT_Siamese_with_status_{pretrain_tag}.pt")
            if isinstance(pre_trained_model, nn.DataParallel):
                model_state_dict = pre_trained_model.module.state_dict()
            else:
                model_state_dict = pre_trained_model.state_dict()
            model_dict = model.module.state_dict()

            pre_trained_model_dict = {}
            for k, v in model_state_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        if 'fc' not in k:
                            pre_trained_model_dict[k] = v
                    else:
                        print(f"Size mismatch for key: {k}")
                        print(f"Pre-trained model shape: {v.shape}")
                        print(f"Current model shape: {model_dict[k].shape}")
                        if 'fc' not in k:
                            if k == 'model.pos_embedding':
                                # Use the pre-trained position embeddings for the original sequence

                                if 'prefix' in k:
                                    pre_trained_model_dict[k] = v[:, :model_dict[k].shape[1] - model.num_prefix_tokens, :]

                                if 'prompt' in k:
                                    pre_trained_model_dict[k] = v[:, :model_dict[k].shape[1] - model.num_prompt_tokens, :]
                            else:
                                raise ValueError(f"Size mismatch between pre-trained model and current model for key: {k}")

            model_dict.update(pre_trained_model_dict)
            model.module.load_state_dict(model_dict)

            borrowed_params = sum(p.numel() for p in pre_trained_model_dict.values())
            print(f"Total parameters borrowed from pre-trained model: {borrowed_params}")

            # print the size of the parameters of every borrowed parameter
            for name, _ in pre_trained_model_dict.items():
                print(f"Weight borrowed from pre-trained model: {name}")

        for name, param in model.named_parameters():
            print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters in model: {parameters}")
        parameters = sum(p.numel() for p in model.parameters())
        print(f"Total parameters in model: {parameters}")

        if num_gpus > 1:
            for name, param in model.module.named_parameters():
                print(f"{name}: {param.numel()} parameters")
        else:
            for name, param in model.named_parameters():
                print(f"{name}: {param.numel()} parameters")

        # model = torch.load(f"/home/mhu3/research/PT_Gridcell_finetune/LLM-siamese/finetune_with_action/classification_wholeday_done/results_4_30_2024/models/{model_tag}_{mode}_model_class_{pretrain_tag}.pt" if pretrain_tag is not None else f"results_nearest/models/{model_tag}_{mode}_model_class.pt")

        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        acc_fn = classification_acc("multi-class")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # optimizer = optim.Adam(model.parameters(), lr=lr)
        EPOCHS = 300

        history = model_fit(
            model,
            loss_fn,
            acc_fn,
            optimizer,
            train_loader = train_loader,
            epochs=EPOCHS,
            val_loader=val_loader,
            save_best_only=True,
            early_stopping=10,
            patience=10,
            save_every_epoch=False,
            save_path=f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/classification/results3_scarcity/models/{model_tag}_{mode}_model_class_{pretrain_tag}.pt" if pretrain_tag is not None else f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/classification/results2_scarcity/models/{model_tag}_{mode}_model_class.pt",
            device=device,
        )

        model = torch.load(f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/classification/results3_scarcity/models/{model_tag}_{mode}_model_class_{pretrain_tag}.pt" if pretrain_tag is not None else f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/classification/results2_scarcity/models/{model_tag}_{mode}_model_class.pt")

        evaluate = model_evaluate(
            model,
            loss_fn,
            acc_fn,
            test_loader,
            device=device,
        )
        print(model_tag, pretrain_tag)

        train_loss = history['train_loss']
        val_loss = history['val_loss']
        train_acc = history['train_acc']
        val_acc = history['val_acc']

        if pretrain_tag is not None:
            loss_path = f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/classification/results3_scarcity/loss/{model_tag}_{mode}_class_{pretrain_tag}.pkl"
        else:
            loss_path = f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/classification/results3_scarcity/loss/{model_tag}_{mode}_class.pkl"

        pickle.dump([train_loss, val_loss, train_acc, val_acc], open(loss_path, "wb"))

        EPOCHS = len(train_loss)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(EPOCHS), train_loss, label='train')
        plt.plot(range(EPOCHS), val_loss, label='val')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(EPOCHS), train_acc, label='train')
        plt.plot(range(EPOCHS), val_acc, label='val')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        if pretrain_tag is not None:
            plt.savefig(f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/classification/results3_scarcity/figures/{model_tag}_{mode}_class_{pretrain_tag}.png")
        else:
            plt.savefig(f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/classification/results3_scarcity/figures/{model_tag}_{mode}_class.png")

if __name__ == '__main__':
    # set seed
    import numpy as np
    import random
    import torch
    
    # seed = np.random.randint(0, 10000)
    # print(f"Seed: {seed}")
    
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Run main
    main()
