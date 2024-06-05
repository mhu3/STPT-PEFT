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

from models_ss import  binaryViT
from torch_utils import get_device, numpy_to_data_loader
from torch_utils import model_fit, classification_acc, model_evaluate

def main():
    device = get_device()

    x_train, y_train = pickle.load(open('/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/seek_serve/dataset/train_seek_and_serve_pad_07_2000_2020.pkl', 'rb'))

    print(x_train.shape, y_train.shape)

    # # #train,val,test split
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.6, random_state=42, stratify=y_train)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    # get a subset of the training data to see the data scarcity, 50% of the data
    x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.9, random_state=42, stratify=y_train)

    y_train = y_train[:, np.newaxis]
    y_val = y_val[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    # normalize the data
    x_train[:, :, 0] = x_train[:, :, 0] / 92
    x_train[:, :, 1] = x_train[:, :, 1] / 49
    x_train[:, :, 2] = x_train[:, :, 2] / 288

    x_val[:, :, 0] = x_val[:, :, 0] / 92
    x_val[:, :, 1] = x_val[:, :, 1] / 49
    x_val[:, :, 2] = x_val[:, :, 2] / 288

    x_test[:, :, 0] = x_test[:, :, 0] / 92
    x_test[:, :, 1] = x_test[:, :, 1] / 49
    x_test[:, :, 2] = x_test[:, :, 2] / 288
    
    # print how many 0 and 1 in train, val, test
    print(np.sum(y_train == 0), np.sum(y_train == 1))
    print(np.sum(y_val == 0), np.sum(y_val == 1))
    print(np.sum(y_test == 0), np.sum(y_test == 1))

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    train_loader = numpy_to_data_loader(x_train, y_train, y_dtype = torch.float32, batch_size=16, shuffle=True)
    val_loader = numpy_to_data_loader(x_val, y_val, y_dtype = torch.float32, batch_size=512, shuffle=False)
    test_loader = numpy_to_data_loader(x_test, y_test, y_dtype = torch.float32, batch_size=512, shuffle=False)

    # Train model
    model = binaryViT(input_size=x_train.shape[2])

    model_tag = 'ViT_large'

    pretrain_tag = 'halfyear_100_20000_30epochs_1'
 
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {parameters}")

    # learning_rates = [5e-4]
    learning_rates = [1e-4, 5e-5]
    batch_sizes = [16]

    for lr in learning_rates:  # Outer loop over learning rates
        print(f"Running with learning rate: {lr}")
        for batch_size in batch_sizes:  # Inner loop over batch sizes
            print(f"Running with batch size: {batch_size}")

            # Data loading happens inside the inner loop to reflect the current batch size
            train_loader = numpy_to_data_loader(x_train, y_train, y_dtype=torch.float32, batch_size=batch_size, shuffle=True)
            val_loader = numpy_to_data_loader(x_val, y_val, y_dtype=torch.float32, batch_size=512, shuffle=False)
            test_loader = numpy_to_data_loader(x_test, y_test, y_dtype=torch.float32, batch_size=512, shuffle=False)

            model = binaryViT(input_size=x_train.shape[2],
                            num_heads = 16, 
                            num_layers = 8, 
                            dropout=0.1, 
                            use_lora=True, 
                            r=8, 
                            adapter_size=64,
                            use_houlsby=False, 
                            use_weighted_layer_sum=False,
                            ensembler_hidden_size=64,
                            apply_knowledge_ensembler=False,
                            enable_adapterbias =False,
                            num_prefix_tokens=0,
                            num_prompt_tokens=0,
                            enable_prefix_tuning=False,
                            enable_prompt_tuning=False)

            # Define the mode for setting parameter trainability
            mode = 'lora'  # Choose from 'lora', 'frozen', 'non_frozen', 'houlsby', or 'bitfit'

            # Apply mode-specific parameter settings
            if mode == 'lora':
                lora.mark_only_lora_as_trainable(model)
                for name, param in model.named_parameters():
                    # if 'layer_weights' in name or 'fc' in name or "input_layer" in name or "pos_embedding" in name or "cls_token" in name or "knowledge_ensembler" in name:
                    if 'layer_weights' in name or 'fc' in name  or "pos_embedding" in name or "knowledge_ensembler" in name:
                        param.requires_grad = True
            elif mode == 'frozen':
                for name, param in model.named_parameters():
                    # if 'layer_weights' in name or 'fc' in name or "input_layer" in name or "pos_embedding" in name or "cls_token" in name or"knowledge_ensembler" in name:
                    if 'layer_weights' in name or 'fc' in name or "pos_embedding" in name or "knowledge_ensembler" in name:    
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif mode == 'prompt':
                for name, param in model.named_parameters():
                    if 'layer_weights' in name or 'fc' in name or "prompt" in name or "pos_embedding" in name or "prompt_pos_embedding" in name or "knowledge_ensembler" in name: #or "input_layer" in name or "pos_embedding" in name or "cls_token" in name or "prompts" in name or "knowledge_ensembler" in name:
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
                # Freeze all except 'layer_weights' and 'fc'
                for name, param in model.named_parameters():
                    if 'down_project' in name or 'up_project' in name or "pos_embedding" in name or 'layer_weights' in name or 'fc' in name or "knowledge_ensembler" in name: #or "input_layer" in name or "pos_embedding" in name or "cls_token" in name or "knowledge_ensembler" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif mode == 'bitfit':
                for name, param in model.named_parameters():
                    # This will enable gradients only for parameters in 'adapter', 'fc', 'bias', and 'layer_weights',
                    # but exclude the 'input_layer' biases.
                    if ('adapter' in name or 'fc' in name or 'bias' in name or "pos_embedding" in name or "knowledge_ensembler" in name or 'layer_weights' in name) and 'input_layer.bias' not in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif mode == 'adapterbias':
                for name, param in model.named_parameters():
                    # This will enable gradients only for parameters in 'adapter', 'fc', 'bias', and 'layer_weights',
                    # but exclude the 'input_layer' biases.
                    if 'adapter' in name or 'fc' in name or 'layer_weights' in name or "knowledge_ensembler" in name or "pos_embedding" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                raise ValueError("Invalid mode. Choose from 'lora', 'frozen', 'non_frozen', 'houlsby', or 'bitfit', or 'prefix'.")
            
            # Check the number of available GPUs
            num_gpus = torch.cuda.device_count()
            print("Number of available GPUs: ", num_gpus)

            if num_gpus > 1:
                # Wrap the model with nn.DataParallel
                model = nn.DataParallel(model)

                print("Model wrapped with nn.DataParallel")

            # Set model tag according to mode
            model_tag_base = f'ss_large_ViT_lr{lr}_batch{batch_size}_scar'
            # model_tag_base = '30%_from_time_ss_large_ViT_concatenate_knowledge_ensembler_nearest_1'
            model_tag = f'{model_tag_base}_{mode}'

            # print the parameters in the model
            for name, param in model.named_parameters():
                print(name, param.size())

            pretrain_tag = 'halfyear_100_20000_30epochs_1'
            # pretrain_tag = None

            parameters = sum(p.numel() for p in model.parameters())
            print(f"Total parameters in model: {parameters}")

            if pretrain_tag is not None:
                pre_trained_model = torch.load(f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/pretrain/pretrainmodels/large_model_ViT_Siamese_with_status_{pretrain_tag}.pt")
                if isinstance(pre_trained_model, nn.DataParallel):
                    model_state_dict = pre_trained_model.module.state_dict()
                else:
                    model_state_dict = pre_trained_model.state_dict()
                
                parameters = sum(p.numel() for p in model_state_dict.values())
                print(f"Total parameters in pre-trained model: {parameters}")
                
                for name in model_state_dict.keys():
                    print(name)
                model_dict = model.module.state_dict()
                for name in model_dict.keys():
                    print(name)

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
                            if 'fc' not in k and 'pos_embedding' not in k and 'input_layer.weight' not in k:
                                raise ValueError(f"Size mismatch between pre-trained model and current model for key: {k}")

                # Initialize positional embedding randomly
                pre_trained_pos_embedding = model_dict['model.pos_embedding']
                pre_trained_pos_embedding = torch.zeros_like(pre_trained_pos_embedding)
                print(pre_trained_pos_embedding.shape)
                pre_trained_model_dict['model.pos_embedding'] = pre_trained_pos_embedding

                # Modify input_layer.weight shape
                input_layer_weight_shape = model_dict['model.input_layer.weight'].shape
                pre_trained_input_layer_weight = model_state_dict['model.input_layer.weight']
                print(input_layer_weight_shape, pre_trained_input_layer_weight.shape)
                if input_layer_weight_shape != pre_trained_input_layer_weight.shape:
                    pre_trained_input_layer_weight = pre_trained_input_layer_weight[:, :input_layer_weight_shape[1]]
                print(pre_trained_input_layer_weight.shape)
                pre_trained_model_dict['model.input_layer.weight'] = pre_trained_input_layer_weight
                
                model_dict.update(pre_trained_model_dict)
                model.module.load_state_dict(model_dict)

                borrowed_params = sum(p.numel() for p in pre_trained_model_dict.values())
                print(f"Total parameters borrowed from pre-trained model: {borrowed_params}")

                for name, _ in pre_trained_model_dict.items():
                    print(f"Weight borrowed from pre-trained model: {name}")

            for name, param in model.named_parameters():
                print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

            # print the number of total trainable parameters
            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters in model: {parameters}")

            # print the number of total trainable parameters except for 'fc'
            parameters = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'fc' not in name)
            print(f"Total trainable parameters in model except for 'fc': {parameters}")


            # model = torch.load(f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/seek_serve/results_scarcity/models/ss_large_ViT_lr0.0005_batch16_bitfit_model_seek_serve_pretrain_halfyear_100_20000_30epochs_1.pt')

            model.to(device)
            loss_fn = nn.BCELoss()
            acc_fn = classification_acc("binary")
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            EPOCHS = 800
            
            history = model_fit(
                model,
                loss_fn,
                acc_fn,
                optimizer,
                train_loader,
                epochs=EPOCHS,
                val_loader=val_loader,
                save_best_only=True,
                early_stopping=10,
                patience = 10,
                save_every_epoch=False,
                save_path=f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/seek_serve/results2_scarcity3/models/{model_tag}_model_seek_serve_pretrain_{pretrain_tag}.pt' if pretrain_tag is not None else f'{model_tag}_model_seek_serve.pt',
                # save_path=f'{model_tag}_model_seek_serve.pt',
                device=device,
            )

            train_loss = history['train_loss']
            val_loss = history['val_loss']
            train_acc = history['train_acc']
            val_acc = history['val_acc']
            
            model = torch.load(f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/seek_serve/results2_scarcity3/models/{model_tag}_model_seek_serve_pretrain_{pretrain_tag}.pt' if pretrain_tag is not None else f'{model_tag}_model_seek_serve.pt')
            # Evaluate model
            evaluate = model_evaluate(
                model,
                loss_fn,
                acc_fn,
                test_loader,
                device=device,
            )
            
            # print model tag , pretrain_tag, and test loss
            print(model_tag, pretrain_tag)

            # Save the loss and accuracy if the tag is not None
            if pretrain_tag is not None:
                loss_path = f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/seek_serve/results2_scarcity3/loss/{model_tag}_seek_serve_{pretrain_tag}_07_2000_2020.pkl"
            else:
                loss_path = f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/seek_serve/results2_scarcity3/loss/{model_tag}_seek_serve_07_2000_2020.pkl"
            # loss_path = "loss/{model_tag}_5_class.pkl"
            
            pickle.dump([train_loss, val_loss, train_acc, val_acc], open(loss_path, "wb"))

            # pickle.dump([train_loss, val_loss, train_acc, val_acc], open('loss/{model_tag}_seek_serve_{pretrain_tag}.pkl', 'wb'))

            EPOCHS = len(train_loss)

            # Draw figure to plot the loss and accuracy
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
                plt.savefig(f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/seek_serve/results2_scarcity3/figures/{model_tag}_seek_serve_{pretrain_tag}_07_2000_2020.png')
            else:
                plt.savefig(f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/seek_serve/results2_scarcity3/figures/{model_tag}_seek_serve_07_2000_2020.png')

if __name__ == '__main__':
    # set seed
    import numpy as np
    import random
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Run main
    main()


    
