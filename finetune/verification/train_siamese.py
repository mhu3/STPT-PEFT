# import pickle
# import pandas as pd
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# from sklearn.model_selection import train_test_split
# import loratorch as lora

# #choose the GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# from models_siamese import SiameseViT
# from torch_utils import get_device, numpy_to_data_loader
# from torch_utils import model_fit, classification_acc, model_evaluate

# def main():
#     device = get_device()

#     # Load pre-processed data
#     # train_class the first  4days
#     # val_class the last 1day
#     x_train, y_train = pickle.load(open('dataset/downstream_train_siamese_whole_day_with_status_07_pad_2000_2020_2000.pkl', 'rb'))
#     # x_val, y_val = pickle.load(open('dataset/downstream_val_siamese_whole_day_with_status_07_pad_2050_2100_1000.pkl', 'rb'))
#     # print(x_train.shape, y_train.shape)

#     # # #train,val,test split
#     # right now the results2_scarcity2
#     x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.6, random_state=42, stratify=y_train)
#     # try a new split rate
#     # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=42, stratify=y_train)
#     x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.4, random_state=42, stratify=y_test)
#     # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

#     # x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.75, random_state=42, stratify=y_train)

#     y_train = y_train[:, np.newaxis]
#     y_val = y_val[:, np.newaxis]
#     y_test = y_test[:, np.newaxis]

#     # Normalize the small training set (and optionally the remaining set)
#     x_train[:, :, 0] = x_train[:, :, 0] / 92
#     x_train[:, :, 1] = x_train[:, :, 1] / 49
#     x_train[:, :, 2] = x_train[:, :, 2] / 288

#     x_val[:, :, 0] = x_val[:, :, 0] / 92
#     x_val[:, :, 1] = x_val[:, :, 1] / 49
#     x_val[:, :, 2] = x_val[:, :, 2] / 288

#     x_test[:, :, 0] = x_test[:, :, 0] / 92
#     x_test[:, :, 1] = x_test[:, :, 1] / 49
#     x_test[:, :, 2] = x_test[:, :, 2] / 288

#     print(x_train.shape, y_train.shape)
#     print(x_val.shape, y_val.shape)
#     print(x_test.shape, y_test.shape)

#     train_loader = numpy_to_data_loader(x_train, y_train, y_dtype = torch.float32, batch_size=16, shuffle=True)
#     val_loader = numpy_to_data_loader(x_val, y_val, y_dtype = torch.float32, batch_size=256, shuffle=False)
#     test_loader = numpy_to_data_loader(x_test, y_test, y_dtype = torch.float32, batch_size=256, shuffle=False)

#     learning_rates = [5e-3]

#     for lr in learning_rates:
#         print(f"Running with learning rate: {lr}")
#         # Train model
#         model = SiameseViT(input_size=x_train.shape[2],
#                         num_heads = 16, 
#                         num_layers = 8, 
#                         dropout=0.1, 
#                         use_lora=True,
#                         r=8, 
#                         adapter_size=64,
#                         use_houlsby=False, 
#                         use_weighted_layer_sum=False,
#                         ensembler_hidden_size=64,
#                         apply_knowledge_ensembler=False,
#                         enable_adapterbias=False,
#                         num_prefix_tokens=0,
#                         num_prompt_tokens=0,
#                         enable_prefix_tuning=False,
#                         enable_prompt_tuning=False)
                        

#         # Define the mode for setting parameter trainability
#         mode = 'lora'  # Choose from 'lora', 'frozen', 'non_frozen', 'houlsby', or 'bitfit'

#         # Apply mode-specific parameter settings
       
#         if mode == 'lora':
#             lora.mark_only_lora_as_trainable(model)
#             for name, param in model.named_parameters():
#                 # if 'layer_weights' in name or 'fc' in name or "input_layer" in name or "pos_embedding" in name or "cls_token" in name or "knowledge_ensembler" in name:
#                 if 'layer_weights' in name or 'fc' in name or "knowledge_ensembler" in name or "pos_embedding" in name:
#                     param.requires_grad = True
#         elif mode == 'frozen':
#             for name, param in model.named_parameters():
#                 if 'layer_weights' in name or 'fc' in name or "knowledge_ensembler" in name or "pos_embedding" in name: 
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False
#         elif mode == 'prompt':
#             for name, param in model.named_parameters():
#                 if 'layer_weights' in name or 'fc' in name or "prompt" in name or "pos_embedding" in name or "knowledge_ensembler" in name: #or "input_layer" in name or "pos_embedding" in name or "cls_token" in name or "prompts" in name or "knowledge_ensembler" in name:
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False   
#         elif mode == 'prefix':
#             for name, param in model.named_parameters():
#                 if 'layer_weights' in name or 'fc' in name or "prefix" in name or "pos_embedding" in name or "knowledge_ensembler" in name:
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False          
#         elif mode == 'non_frozen':
#                 for name, param in model.named_parameters():
#                     if "prompt_embedding" not in name and "prefix_embedding" not in name:
#                         param.requires_grad = True
#                     else:
#                         param.requires_grad = False
#         elif mode == 'houlsby':
#             for name, param in model.named_parameters():
#                 if 'down_project' in name or 'up_project' in name or 'layer_weights' in name or 'fc' in name or "pos_embedding" in name or "knowledge_ensembler" in name: #or "input_layer" in name or "pos_embedding" in name or "cls_token" in name or "knowledge_ensembler" in name:
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False
#         elif mode == 'bitfit':
#             for name, param in model.named_parameters():
#                 if ('adapter' in name or 'fc' in name or 'bias' in name or 'layer_weights' in name or "pos_embedding" in name or "knowledge_ensembler" in name) and 'input_layer.bias' not in name:
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False
#         elif mode == 'adapterbias':
#             for name, param in model.named_parameters():
#                 if 'adapter' in name or 'fc' in name or 'layer_weights' in name or "knowledge_ensembler" in name or "pos_embedding" in name:
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False

#         else:
#             raise ValueError("Invalid mode. Choose from 'lora', 'frozen', or 'non_frozen', or 'houlsby', or 'bitfit'.")
        
#         # Check the number of available GPUs
#         num_gpus = torch.cuda.device_count()
#         print("Number of available GPUs: ", num_gpus)

#         if num_gpus > 1:
#             # Wrap the model with nn.DataParallel
#             model = nn.DataParallel(model)

#         # Set model tag according to mode
#         model_tag_base = f'verification_large_ViT_lr{lr}_pe'
#         model_tag = f'{model_tag_base}_{mode}'

#         # print the parameters in the model
#         for name, param in model.named_parameters():
#             print(name, param.size())

#         # Define the tag for the pre-trained model (or set it to None if no pre-trained model is available)
#         pretrain_tag = "halfyear_100_20000_30epochs_1"  # Change this value or set it to None
#         # pretrain_tag = None
        
#         if pretrain_tag is not None:
#             pre_trained_model = torch.load(f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/pretrain/pretrainmodels/large_model_ViT_Siamese_with_status_{pretrain_tag}.pt")
#             if isinstance(pre_trained_model, nn.DataParallel):
#                 model_state_dict = pre_trained_model.module.state_dict()
#             else:
#                 model_state_dict = pre_trained_model.state_dict()
#             for name in model_state_dict.keys():
#                 print(name)
#             model_dict = model.module.state_dict()
#             for name in model_dict.keys():
#                 print(name)

#             parameters = sum(p.numel() for p in model_state_dict.values())
#             print(f"Total parameters in pre-trained model: {parameters}")

#             # print the parameters in the model
#             for name, param in model_state_dict.items():
#                 print(name, param.size())
            
#             pre_trained_model_dict = {k: v for k, v in model_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
#             # Update the model's state dictionary with the filtered pre-trained model's state dictionary
#             model_dict.update(pre_trained_model_dict)

#             # Load the modified state dictionary into the model
#             model.module.load_state_dict(model_dict)

#             borrowed_params = sum(p.numel() for p in pre_trained_model_dict.values())
#             print(f"Total parameters borrowed from pre-trained model: {borrowed_params}")

#             for name, _ in pre_trained_model_dict.items():
#                 print(f"Weight borrowed from pre-trained model: {name}")

#         for name, param in model.named_parameters():
#             print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

#         # print total number of tranaible parameters in the model
#         parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print(f"Total trainable parameters in model: {parameters}")
#         #print total number of parameters in the model
#         parameters = sum(p.numel() for p in model.parameters())
#         print(f"Total parameters in model: {parameters}")

#         if num_gpus > 1:
#             for name, param in model.module.named_parameters():
#                 print(f"{name}: {param.numel()} parameters")
#         else:
#             for name, param in model.named_parameters():
#                 print(f"{name}: {param.numel()} parameters")

#         for name, param in model.named_parameters():
#             print(f"{name}: {param.numel()} parameters") 

#         model.to(device)
#         loss_fn = nn.BCELoss()
#         acc_fn = classification_acc("binary")
#         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#         EPOCHS = 100
        
#         history = model_fit(
#             model,
#             loss_fn,
#             acc_fn,
#             optimizer,
#             train_loader,
#             epochs=EPOCHS,
#             val_loader=val_loader,
#             save_best_only=True,
#             early_stopping=10,
#             patience = 10,
#             save_every_epoch=False,
#             save_path=f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2/models/{model_tag}_lr{lr}_verification_pretrain_{pretrain_tag}.pt' if pretrain_tag is not None else f'new/new_models/{model_tag}_model_verification.pt',
#             device=device,
#         )

#         train_loss = history['train_loss']
#         val_loss = history['val_loss']
#         train_acc = history['train_acc']
#         val_acc = history['val_acc']
        
#         model = torch.load(f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2/models/{model_tag}_lr{lr}_verification_pretrain_{pretrain_tag}.pt' if pretrain_tag is not None else f'new/new_models/{model_tag}_model_verification.pt')
        
#         # Evaluate model
#         evaluate = model_evaluate(
#             model,
#             loss_fn,
#             acc_fn,
#             test_loader,
#             device=device,
#         )
        
#         # print model tag , pretrain_tag, and test loss
#         print(model_tag, pretrain_tag)

#         # Save the loss and accuracy if the tag is not None
#         if pretrain_tag is not None:
#             loss_path = f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2/loss/{model_tag}_lr{lr}_verification_{pretrain_tag}_07_2050_2100.pkl"
#         else:
#             loss_path = f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2/loss/{model_tag}_lr{lr}_verification_07_2050_2100.pkl"
#         # loss_path = "loss/{model_tag}_5_class.pkl"
        
#         pickle.dump([train_loss, val_loss, train_acc, val_acc], open(loss_path, "wb"))

#         EPOCHS = len(train_loss)

#         # Draw figure to plot the loss and accuracy
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.plot(range(EPOCHS), train_loss, label='train')
#         plt.plot(range(EPOCHS), val_loss, label='val')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.subplot(1, 2, 2)
#         plt.plot(range(EPOCHS), train_acc, label='train')
#         plt.plot(range(EPOCHS), val_acc, label='val')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend()
        
#         if pretrain_tag is not None:
#             plt.savefig(f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2/figures/{model_tag}_lr{lr}_verification_{pretrain_tag}_07_2050_2100.png')
#         else:
#             plt.savefig(f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2/figures/{model_tag}_lr{lr}_verification_07_2050_2100.png')

# if __name__ == '__main__':
#     # set seed
#     import numpy as np
#     import random
#     import torch

#     torch.manual_seed(0)
#     np.random.seed(0)
#     random.seed(0)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # Run main
#     main()



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

from models_siamese import SiameseViT
from torch_utils import get_device, numpy_to_data_loader
from torch_utils import model_fit, classification_acc, model_evaluate

def main():
    device = get_device()

    # Load pre-processed data
    # train_class the first  4days
    # val_class the last 1day
    x_train, y_train = pickle.load(open('dataset/downstream_train_siamese_whole_day_with_status_07_pad_2000_2020_2000.pkl', 'rb'))
    # x_val, y_val = pickle.load(open('dataset/downstream_val_siamese_whole_day_with_status_07_pad_2050_2100_1000.pkl', 'rb'))
    # print(x_train.shape, y_train.shape)

    # # #train,val,test split
    # right now the results2_scarcity2
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.6, random_state=42, stratify=y_train)
    # try a new split rate
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=42, stratify=y_train)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.4, random_state=42, stratify=y_test)
    # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.75, random_state=42, stratify=y_train)

    y_train = y_train[:, np.newaxis]
    y_val = y_val[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    # Normalize the small training set (and optionally the remaining set)
    x_train[:, :, 0] = x_train[:, :, 0] / 92
    x_train[:, :, 1] = x_train[:, :, 1] / 49
    x_train[:, :, 2] = x_train[:, :, 2] / 288

    x_val[:, :, 0] = x_val[:, :, 0] / 92
    x_val[:, :, 1] = x_val[:, :, 1] / 49
    x_val[:, :, 2] = x_val[:, :, 2] / 288

    x_test[:, :, 0] = x_test[:, :, 0] / 92
    x_test[:, :, 1] = x_test[:, :, 1] / 49
    x_test[:, :, 2] = x_test[:, :, 2] / 288

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    train_loader = numpy_to_data_loader(x_train, y_train, y_dtype = torch.float32, batch_size=16, shuffle=True)
    val_loader = numpy_to_data_loader(x_val, y_val, y_dtype = torch.float32, batch_size=256, shuffle=False)
    test_loader = numpy_to_data_loader(x_test, y_test, y_dtype = torch.float32, batch_size=256, shuffle=False)

    learning_rates = [5e-3]

    configs = [
        # {'mode': 'non_linear', 'num_prefix_tokens': 0, 'num_prompt_tokens': 0},
        # {'mode': 'weighted_sum', 'num_prefix_tokens': 0, 'num_prompt_tokens': 0},
        # {'mode': 'bitfit', 'num_prefix_tokens': 0, 'num_prompt_tokens': 0},
        # # {'mode': 'frozen', 'num_prefix_tokens': 0, 'num_prompt_tokens': 0},
        # {'mode': 'prompt', 'num_prefix_tokens': 0, 'num_prompt_tokens': 200},
        # {'mode': 'prefix', 'num_prefix_tokens': 50, 'num_prompt_tokens': 0},
        # {'mode': 'houlsby', 'num_prefix_tokens': 0, 'num_prompt_tokens': 0},
        # {'mode': 'prompt', 'num_prefix_tokens': 0, 'num_prompt_tokens': 200},
        # {'mode': 'prefix', 'num_prefix_tokens': 50, 'num_prompt_tokens': 0},
        {'mode': 'lora', 'num_prefix_tokens': 0, 'num_prompt_tokens': 0},
    ]

    for config in configs:
        mode = config['mode']
        num_prefix_tokens = config['num_prefix_tokens']
        num_prompt_tokens = config['num_prompt_tokens']

        print(f"Running with mode: {mode}")
        print(f"Number of prefix tokens: {num_prefix_tokens}")
        print(f"Number of prompt tokens: {num_prompt_tokens}")

        for lr in learning_rates:
            print(f"Running with learning rate: {lr}")
            # Train model
            model = SiameseViT(input_size=x_train.shape[2],
                            num_heads = 16, 
                            num_layers = 8, 
                            dropout=0.1, 
                            use_lora=(mode == 'lora'),
                            r=8, 
                            adapter_size=64,
                            use_houlsby=(mode == 'houlsby'), 
                            use_weighted_layer_sum=(mode == 'weighted_sum'),
                            ensembler_hidden_size=64,
                            apply_knowledge_ensembler=(mode == 'non_linear'),
                            enable_adapterbias=(mode == 'adapterbias'),
                            num_prefix_tokens=num_prefix_tokens,
                            num_prompt_tokens=num_prompt_tokens,
                            enable_prefix_tuning=(mode == 'prefix'),
                            enable_prompt_tuning=(mode == 'prompt'))

            # # Define the mode for setting parameter trainability
            mode = mode

            # Apply mode-specific parameter settings
        
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
            elif mode == 'weighted_sum':
                for name, param in model.named_parameters():
                    if 'layer_weights' in name or 'fc' in name or "knowledge_ensembler" in name or "pos_embedding" in name: 
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif mode == 'non_linear':
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
                    if 'down_project' in name or 'up_project' in name or 'layer_weights' in name or 'fc' in name or "pos_embedding" in name or "knowledge_ensembler" in name: #or "input_layer" in name or "pos_embedding" in name or "cls_token" in name or "knowledge_ensembler" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif mode == 'bitfit':
                for name, param in model.named_parameters():
                    if ('adapter' in name or 'fc' in name or 'bias' in name or 'layer_weights' in name or "pos_embedding" in name or "knowledge_ensembler" in name) and 'input_layer.bias' not in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif mode == 'adapterbias':
                for name, param in model.named_parameters():
                    if 'adapter' in name or 'fc' in name or 'layer_weights' in name or "knowledge_ensembler" in name or "pos_embedding" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            else:
                raise ValueError("Invalid mode. Choose from 'lora', 'frozen', or 'non_frozen', or 'houlsby', or 'bitfit'.")
            
            # Check the number of available GPUs
            num_gpus = torch.cuda.device_count()
            print("Number of available GPUs: ", num_gpus)

            if num_gpus > 1:
                # Wrap the model with nn.DataParallel
                model = nn.DataParallel(model)

            # Set model tag according to mode
            model_tag_base = f'verification_large_ViT_lr{lr}_pe'
            model_tag = f'{model_tag_base}_{mode}'

            # print the parameters in the model
            for name, param in model.named_parameters():
                print(name, param.size())

            # Define the tag for the pre-trained model (or set it to None if no pre-trained model is available)
            pretrain_tag = "halfyear_100_20000_30epochs_1"  # Change this value or set it to None
            # pretrain_tag = None
            
            if pretrain_tag is not None:
                pre_trained_model = torch.load(f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/pretrain/pretrainmodels/large_model_ViT_Siamese_with_status_{pretrain_tag}.pt")
                if isinstance(pre_trained_model, nn.DataParallel):
                    model_state_dict = pre_trained_model.module.state_dict()
                else:
                    model_state_dict = pre_trained_model.state_dict()
                for name in model_state_dict.keys():
                    print(name)
                model_dict = model.module.state_dict()
                for name in model_dict.keys():
                    print(name)

                parameters = sum(p.numel() for p in model_state_dict.values())
                print(f"Total parameters in pre-trained model: {parameters}")

                # print the parameters in the model
                for name, param in model_state_dict.items():
                    print(name, param.size())
                
                pre_trained_model_dict = {k: v for k, v in model_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                # Update the model's state dictionary with the filtered pre-trained model's state dictionary
                model_dict.update(pre_trained_model_dict)

                # Load the modified state dictionary into the model
                model.module.load_state_dict(model_dict)

                borrowed_params = sum(p.numel() for p in pre_trained_model_dict.values())
                print(f"Total parameters borrowed from pre-trained model: {borrowed_params}")

                for name, _ in pre_trained_model_dict.items():
                    print(f"Weight borrowed from pre-trained model: {name}")

            for name, param in model.named_parameters():
                print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

            # print total number of tranaible parameters in the model
            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters in model: {parameters}")
            #print total number of parameters in the model
            parameters = sum(p.numel() for p in model.parameters())
            print(f"Total parameters in model: {parameters}")

            if num_gpus > 1:
                for name, param in model.module.named_parameters():
                    print(f"{name}: {param.numel()} parameters")
            else:
                for name, param in model.named_parameters():
                    print(f"{name}: {param.numel()} parameters")

            for name, param in model.named_parameters():
                print(f"{name}: {param.numel()} parameters") 

            model.to(device)
            loss_fn = nn.BCELoss()
            acc_fn = classification_acc("binary")
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            EPOCHS = 100
            
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
                save_path=f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2_scarcity_25/models/{model_tag}_lr{lr}_verification_pretrain_{pretrain_tag}.pt' if pretrain_tag is not None else f'new/new_models/{model_tag}_model_verification.pt',
                device=device,
            )

            train_loss = history['train_loss']
            val_loss = history['val_loss']
            train_acc = history['train_acc']
            val_acc = history['val_acc']
            
            model = torch.load(f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2_scarcity_25/models/{model_tag}_lr{lr}_verification_pretrain_{pretrain_tag}.pt' if pretrain_tag is not None else f'new/new_models/{model_tag}_model_verification.pt')
            
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
                loss_path = f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2_scarcity_25/loss/{model_tag}_lr{lr}_verification_{pretrain_tag}_07_2050_2100.pkl"
            else:
                loss_path = f"/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2_scarcity_25/loss/{model_tag}_lr{lr}_verification_07_2050_2100.pkl"
            # loss_path = "loss/{model_tag}_5_class.pkl"
            
            pickle.dump([train_loss, val_loss, train_acc, val_acc], open(loss_path, "wb"))

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
                plt.savefig(f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2_scarcity_25/figures/{model_tag}_lr{lr}_verification_{pretrain_tag}_07_2050_2100.png')
            else:
                plt.savefig(f'/home/mhu3/research/PT_Gridcell_finetune/STPT-PEFT/finetune/verification/results2_scarcity_25/figures/{model_tag}_lr{lr}_verification_07_2050_2100.png')

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


    

