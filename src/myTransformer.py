import torch
import torch.nn as nn
from src.myLayers import EncoderLayer, DecoderLayer
from src.units import PositionalEncoding 

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_len):
        super().__init__()
        self.token_embed = nn.Embedding(input_vocab_size, d_model)

        self.position_embed = PositionalEncoding(d_model, max_seq_len)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
    def forward(self, src_seq, src_mask):

        x = self.token_embed(src_seq) 
        x = self.position_embed(x)
        

        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x 

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, output_vocab_size, max_seq_len):
        super().__init__()
        self.token_embed = nn.Embedding(output_vocab_size, d_model)
        self.position_embed = PositionalEncoding(d_model, max_seq_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
    def forward(self, tgt_seq, enc_output, src_mask, tgt_mask):

        x = self.token_embed(tgt_seq)
        x = self.position_embed(x)
        

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return x 

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            input_vocab_size=config.src_vocab_size,
            max_seq_len=config.max_seq_len
        )
        
        self.decoder = Decoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            output_vocab_size=config.tgt_vocab_size,
            max_seq_len=config.max_seq_len
        )
        

        self.output_head = nn.Linear(config.d_model, config.tgt_vocab_size)
        
    def forward(self, src_seq, tgt_seq, src_mask, tgt_mask):

        enc_output = self.encoder(src_seq, src_mask)
        

        dec_output = self.decoder(tgt_seq, enc_output, src_mask, tgt_mask)
        

        final_output = self.output_head(dec_output)
        
        return final_output