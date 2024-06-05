# input, seek trajectories or serving trajectories
# seek trajectories, label 0, serve trajectories, label 1
# Simple clasification

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.init as init
import torch.nn.functional as F
import loratorch as lora

# Binary classification for seek and serve using LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # initialize weights
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

class  SimpleATLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
            super(SimpleATLSTM, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size = self.input_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
            )

            self.wh = nn.Parameter(
                torch.Tensor(hidden_size, hidden_size)
            )

            self.omega = nn.Parameter(
                torch.Tensor(hidden_size)
            )

            self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

            for layer in self.fc:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
    def attention(self, H):
        # # attetion-based lstm
        # H, _ = self.lstm(x)
        # attention weights
        M = torch.tanh(torch.matmul(H, self.wh))
        dot_prod = torch.matmul(M, self.omega)
        alpha = torch.softmax(dot_prod, dim=1)
        r = torch.einsum("nqh,nq->nh", [H, alpha])
        return r

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.fc(x)
        return x

    
class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()
        self.input_size = input_size
        
        # self.conv = nn.Sequential(
        #     nn.Conv1d(input_size, 768, 4),
        #     # nn.Conv1d(input_size, 896, 4),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(4),
        #     # nn.Conv1d(896, 256, 4),
        #     nn.Conv1d(768, 256, 4),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(4),
        #     # nn.Conv1d(256, 128, 4),
        #     # nn.Dropout(0.2),
        #     # nn.ReLU(),
        #     # nn.MaxPool1d(4),
        #     nn.AdaptiveAvgPool1d(1)
        # )

        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1)
        )

        
        # # Large CNN
        # self.conv = nn.Sequential(
        #     # nn.Conv1d(input_size, 640, 4),
        #     nn.Conv1d(input_size, 896, 3),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3),
        #     nn.Conv1d(896, 256, 3),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3),
        #     nn.Conv1d(256, 128, 3),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3),
        #     nn.AdaptiveAvgPool1d(1)
        # )

        # # Base CNN
        # self.conv = nn.Sequential(
        #     nn.Conv1d(input_size, 256, 3),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3),
        #     nn.Conv1d(256, 128, 3),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3),
        #     nn.AdaptiveAvgPool1d(1)
        # )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            # nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # initialize weights
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x   

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # initialize weights
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super(SimpleGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # initialize weights
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

    
class Houlsby_adapter(nn.Module):
    def __init__(self, input_dim, adapter_size):
        super(Houlsby_adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, adapter_size)
        self.up_project = nn.Linear(adapter_size, input_dim)

    def forward(self, x):
        x_down = self.down_project(x)
        x_act = F.gelu(x_down)  # Apply GELU activation
        x_up = self.up_project(x_act)
        return x_up

class AdapterBias(nn.Module):
    """Implementation of Adapter with Bias Vector

    References: https://arxiv.org/abs/2205.00305.
    """

    def __init__(self, input_size, dropout=0.8):
        super().__init__()
        self.adapter_vector = nn.Parameter(torch.ones((input_size), requires_grad=True))

        self.adapter_alpha = nn.Linear(input_size, 1)


    def forward(self, x):
        return self.adapter_vector * self.adapter_alpha(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0, enable_lora=True, r=4, adapter_size=64, enable_houlsby=False, enable_adapterbias=False):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        if enable_lora:
            self.attn = lora.MultiheadAttention(embed_dim, num_heads, enable_lora=['q', 'k', 'v', 'o'], r=r)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.layer_norm_2 = nn.LayerNorm(embed_dim)

        self.enable_houlsby = enable_houlsby
        self.enable_adapterbias = enable_adapterbias

        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        if enable_houlsby:
            self.adapter = Houlsby_adapter(embed_dim, adapter_size)

        if enable_adapterbias:
            self.adapterbias = AdapterBias(embed_dim, adapter_size)


    def forward(self, input, prefix_embedding=None):
        inp_x = self.layer_norm_1(input)

        if prefix_embedding is not None:
            prefix_k = prefix_embedding.expand(inp_x.size(1), -1, -1)
            prefix_v = prefix_embedding.expand(inp_x.size(1), -1, -1)
            prefix_k = prefix_k.transpose(0, 1)
            prefix_v = prefix_v.transpose(0, 1)

            # Concatenation should maintain proper alignment for multi-head attention
            concat_k = torch.cat([prefix_k, inp_x], dim=0)
            concat_v = torch.cat([prefix_v, inp_x], dim=0)

            attn_output = self.attn(inp_x, concat_k, concat_v)[0]
        else:
            attn_output = self.attn(inp_x, inp_x, inp_x)[0]

        x = attn_output + input  # Apply skip connection after attention
        y = self.layer_norm_2(x)
        y = self.linear(y)  # Apply linear layers

        if self.enable_houlsby:
            y = self.adapter(y)  # Apply second adapter after feed-forward network

        if self.enable_adapterbias:
            y = self.adapterbias(y)  # Apply AdapterBias after feed-forward network and adapter (if enabled)

        return x + y  # Apply skip connection after linear layers
        

class VisionTransformer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 hidden_dim, 
                 num_heads, 
                 num_layers, 
                 dropout=0.0, 
                 normalize_before_sum=False, 
                 enable_lora=False, 
                 r=4, 
                 adapter_size=64, 
                 enable_houlsby=False, 
                 use_weighted_layer_sum=False,
                 ensembler_hidden_dim=128,
                 apply_knowledge_ensembler=False,
                 enable_adapterbias=False,
                 num_prefix_tokens=10,
                 num_prompt_tokens=10,
                 enable_prefix_tuning=False,
                 enable_prompt_tuning=False):
        
        super().__init__()

        self.input_layer = nn.Linear(4, embed_dim)

        self.transformer = nn.ModuleList([AttentionBlock(embed_dim, 
                                                         hidden_dim, 
                                                         num_heads, 
                                                         dropout=dropout, 
                                                         enable_lora=enable_lora, 
                                                         r=r, 
                                                         adapter_size=adapter_size, 
                                                         enable_houlsby=enable_houlsby,
                                                         enable_adapterbias=enable_adapterbias) for _ in range(num_layers)])
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 61 + num_prefix_tokens + num_prompt_tokens, embed_dim))

        self.num_prefix_tokens = num_prefix_tokens
        self.prefix_embedding = nn.Parameter(torch.randn(num_prefix_tokens, embed_dim))

        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_embedding = nn.Parameter(torch.randn(num_prompt_tokens, embed_dim))

        if apply_knowledge_ensembler:
            # Add the knowledge ensembler only if apply_knowledge_ensembler is True
            self.knowledge_ensembler = nn.Sequential(
                nn.Linear(num_layers * embed_dim, embed_dim),
            )

        self.apply_knowledge_ensembler = apply_knowledge_ensembler
        
        # Initialization starting at zero
        self.use_weighted_layer_sum = use_weighted_layer_sum
        if self.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        
        self.normalize_before_sum = normalize_before_sum
        self.enable_prefix_tuning = enable_prefix_tuning
        self.enable_prompt_tuning = enable_prompt_tuning

    def forward(self, x, 
                return_all_layers=False, 
                return_cls_representation=False, 
                return_weighted_layer_sum=False, 
                return_cls_representation_weightes_sum=False, 
                return_knowledge_ensembler=False):
        
        B, T, _ = x.shape
        x = self.input_layer(x)

        if self.enable_prompt_tuning:
            prompt_tokens = self.prompt_embedding.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([x[:, :1, :], prompt_tokens, x[:, 1:, :]], dim=1)
        
        cls_token = self.cls_token.repeat(B, 1, 1)
        if self.enable_prefix_tuning:
            prefix_tokens = self.prefix_embedding.unsqueeze(0).expand(B, -1, -1)  # Repeat prefix tokens for batch
            x = torch.cat([cls_token, prefix_tokens, x], dim=1)  # Concatenate CLS token, prefix tokens, and input
        else:
            x = torch.cat([cls_token, x], dim=1)  # Concatenate CLS token and input without prefix tokens

        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.dropout(x)
        x = x.transpose(0, 1)
        
        layer_outputs = []
        for block in self.transformer:
            if self.enable_prefix_tuning:
                x = block(x, prefix_embedding=self.prefix_embedding)
            else:
                x = block(x)
            layer_outputs.append(x.transpose(0, 1))  # Transpose to keep batch dimension first

        if return_all_layers:
            return layer_outputs
        
        elif return_cls_representation:
            cls_representation = layer_outputs[-1][:, 0, :]
            return cls_representation
        
        elif return_weighted_layer_sum:
            # Stack all layer outputs
            stacked_feature = torch.stack(layer_outputs, dim=0)

            # Normalize before summing
            if self.normalize_before_sum:
                stacked_feature = F.layer_norm(stacked_feature, stacked_feature.size()[1:])
            
            # Get the original shape, excluding the number of layers
            _, *origin_shape = stacked_feature.shape
            
            # Flatten all dimensions except the layers for weighting
            stacked_feature = stacked_feature.view(self.layer_weights.size(0), -1)  
            
            # Apply softmax to get normalized weights
            norm_weights = F.softmax(self.layer_weights, dim=0)

            # Weighted sum of features across layers
            weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)

            # Reshape back to the original shape (minus the layer dimension)
            weighted_feature = weighted_feature.view(*origin_shape)

            return weighted_feature
        
        elif return_cls_representation_weightes_sum:
            # Stack all layer outputs
            stacked_feature = torch.stack(layer_outputs, dim=0)

            # Normalize before summing
            if self.normalize_before_sum:
                stacked_feature = F.layer_norm(stacked_feature, stacked_feature.size()[1:])
            
            # Get the original shape, excluding the number of layers
            _, *origin_shape = stacked_feature.shape
            
            # Flatten all dimensions except the layers for weighting
            stacked_feature = stacked_feature.view(self.layer_weights.size(0), -1)
            
            # Apply softmax to get normalized weights
            norm_weights = F.softmax(self.layer_weights, dim=0)
            # Weighted sum of features across layers
            weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
            # Reshape back to the original shape (minus the layer dimension)
            weighted_feature = weighted_feature.view(*origin_shape)
            # Extract the representation corresponding to the cls token
            cls_representation = weighted_feature[:, 0, :]

            return cls_representation
        
        elif self.use_weighted_layer_sum:
            # Stack all layer outputs
            stacked_feature = torch.stack(layer_outputs, dim=0)

            # Normalize before summing
            if self.normalize_before_sum:
                stacked_feature = F.layer_norm(stacked_feature, stacked_feature.size()[1:])
            
            # Get the original shape, excluding the number of layers
            _, *origin_shape = stacked_feature.shape
            
            # Flatten all dimensions except the layers for weighting
            stacked_feature = stacked_feature.view(self.layer_weights.size(0), -1)
            
            # Apply softmax to get normalized weights
            norm_weights = F.softmax(self.layer_weights, dim=0)
            # Weighted sum of features across layers
            weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        
            # Reshape back to the original shape (minus the layer dimension)
            weighted_feature = weighted_feature.view(*origin_shape)

            # Extract the representation corresponding to the cls token
            cls_representation = weighted_feature[:, 0, :]  # Assuming the CLS token is the first element
        
        elif return_knowledge_ensembler:
            # Use knowledge_ensembler only if apply_knowledge_ensembler is True
            cls_representation_concat = torch.cat([layer[:, 0, :] for layer in layer_outputs], dim=1)
            cls_representation = self.knowledge_ensembler(cls_representation_concat)
            return cls_representation
        
        elif self.apply_knowledge_ensembler:
            # Use knowledge_ensembler only if apply_knowledge_ensembler is True
            cls_representation_concat = torch.cat([layer[:, 0, :] for layer in layer_outputs], dim=1)
            cls_representation = self.knowledge_ensembler(cls_representation_concat)
        
        else:
            cls_representation = layer_outputs[-1][:, 0, :]

        out = self.fc(cls_representation)

        return out

class ViT(nn.Module):
    ''' Build model for classification '''
    def __init__(self, input_size, 
                 num_heads=8, 
                 num_layers=1,
                 dropout=0.1, 
                 use_lora=True, 
                 r=4, 
                 adapter_size=64, 
                 use_houlsby=False, 
                 use_weighted_layer_sum=False, 
                 ensembler_hidden_size=128,
                 apply_knowledge_ensembler=False,
                 enable_adapterbias=False,
                 num_prefix_tokens=10,
                 num_prompt_tokens=10,
                 enable_prefix_tuning=False,
                 enable_prompt_tuning=False):
        
        
        super(ViT, self).__init__()
        self.input_size = input_size
        self.model = VisionTransformer(embed_dim=128, 
                                       hidden_dim=128, 
                                       num_heads=num_heads, 
                                       num_layers=num_layers, 
                                       dropout=dropout, 
                                       enable_lora=use_lora, 
                                       r=r, 
                                       adapter_size=adapter_size,
                                       enable_houlsby=use_houlsby, 
                                       use_weighted_layer_sum=use_weighted_layer_sum, 
                                       ensembler_hidden_dim=ensembler_hidden_size,
                                       apply_knowledge_ensembler=apply_knowledge_ensembler,
                                       enable_adapterbias=enable_adapterbias,
                                       num_prefix_tokens=num_prefix_tokens,
                                       num_prompt_tokens=num_prompt_tokens,
                                       enable_prefix_tuning=enable_prefix_tuning,
                                       enable_prompt_tuning=enable_prompt_tuning)
    
    def forward(self, x, 
                return_all_layers=False, 
                return_cls_representation=False, 
                return_weighted_layer_sum=False,
                return_cls_representation_weightes_sum=False,
                return_knowledge_ensembler=False):
        return self.model(x,
                          return_all_layers=return_all_layers, 
                          return_cls_representation=return_cls_representation, 
                          return_weighted_layer_sum=return_weighted_layer_sum,
                          return_cls_representation_weightes_sum=return_cls_representation_weightes_sum,
                          return_knowledge_ensembler=return_knowledge_ensembler)

    def predict(self, x):
        x = self.forward(x)
        _, pred = torch.max(x.data, 1)
        return pred

    def save(self, path, weights_only=False):
        if weights_only:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self, path)