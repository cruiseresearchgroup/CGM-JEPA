import torch.nn as nn
import torch 
import math

class GluFormer(nn.Module):
    '''
        Glucose FM with autoregressive SSL
    '''
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        max_seq_length: int = 25000,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_token: int | None = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.pad_token = pad_token

        self.token_embedding = nn.Embedding(vocab_size + 1, embed_dim)

        self.register_buffer(
            "pos_embedding",
            self._create_pos_embedding(max_seq_length, embed_dim),
            persistent=False
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_head = nn.Linear(embed_dim, vocab_size)
    
    def _create_pos_embedding(self, max_seq_length: int, embed_dim: int) -> torch.Tensor:
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, tokens: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None, ret_embeds: bool = False):
        token_emb = self.token_embedding(tokens)
        T = tokens.size(1)
        pos_emb = self.pos_embedding[:T, :].unsqueeze(0).to(tokens.device)
        
        x = token_emb + pos_emb

        causal_mask = torch.triu(
            torch.ones(T, T, device=tokens.device, dtype=torch.float), diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))

        x = x.permute(1, 0, 2)

        x = self.transformer(
            x,
            mask=causal_mask
            # NOTE: Disabling this as it's deprecated to mix the masks and currently we don't use PAD)
            # src_key_padding_mask=padding_mask_float,
        )

        x = x.permute(1, 0, 2)

        if ret_embeds:
            return x
            
        logits = self.output_head(x)
        return logits