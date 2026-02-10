import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int =5000): # params need to adopt with our implementation
        super().__init__()
        # compute the positional encodings once in log space
        pos_emb = torch.zeros(max_len, d_model).float()
        pos_emb.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)

        pos_emb = pos_emb.unsqueeze(0)
        self.register_buffer('pos_emb', pos_emb) # what is this
    
    def forward(self, x_len: int):
        # returns (1, x_len, d_model)
        return self.pos_emb[:, :x_len]

class ValueEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        patch_size: int,
        bias: bool = True
    ):
        super().__init__()
        self.proj = nn.Conv1d(
            1,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            padding=0
        )

        conv_output_length = in_channels
        conv_output_length = (
            conv_output_length - patch_size
        ) // patch_size + 1

        self.fc = nn.Linear(dim * conv_output_length, dim)
    
    def forward(self, x):
        batch_size, num_patches, patch_length = x.shape

        # process each patch independently
        x = x.view(-1, 1, patch_length) # (B x N, 1, L)

        x = self.proj(x)

        # flatten the output
        x = x.view(x.size(0), -1)

        # linear layer to transform to embedding dimension needed for encoder
        x = self.fc(x)

        # rehsape back and transpose
        x = x.view(batch_size, num_patches, -1)

        return x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model: int, d_inp: int):
        super().__init__()

        #freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        #d_inp = freq_map[freq]
        self.proj = nn.Linear(d_inp, d_model) # learnable
    
    def forward(self, x_mark: torch.Tensor):
        # x_mark: (B, N, patch_size, d_inp) -> (B, N, D)
        B, N, patch_size, d_inp = x_mark.shape
        x_mark = x_mark.view(-1, d_inp)             # (B*L*patch_size, d_inp)
        x_mark = self.proj(x_mark.float())          # (d_model, d_inp)

        # reshape to (B, N, D)
        x_mark = x_mark.view(B, N, patch_size, -1).mean(dim=2)
        return x_mark

class DataEmbedding(nn.Module):
    '''
        @brief: Embed the input into num_patches with d dimension
    '''
    def __init__(
        self,
        dim: int,
        in_channels: int,
        patch_size: int,
        time_inp_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.value_embedding = ValueEmbedding(dim, in_channels, patch_size)
        self.positional_embedding = PositionalEmbedding(dim)
        self.timefeature_embedding = TimeFeatureEmbedding(dim, time_inp_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, x_mark: torch.Tensor):
        # x: (B, C, T), x_mark: (B, L, d_inp)
        val = self.value_embedding(x)                   # (B, L, D)
        pos = self.positional_embedding(val.size(1))    # (1, L, D)
        out = val + pos

        if x_mark is not None and not torch.allclose(x_mark, torch.zeros_like(x_mark)):
            tim = self.timefeature_embedding(x_mark)    # (B, L, D)
            out = out + tim

        return self.dropout(out)