import torch.nn as nn

from src.units import MultiHeadAttention, PositionwiseFeedForward, AddNorm 

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = AddNorm(d_model) 

        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = AddNorm(d_model) 

    def forward(self, x, src_mask):
        attn_output = self.self_attn(Q=x, K=x, V=x, mask=src_mask)
        x = self.norm1(x, attn_output)  

        ffn_output = self.ffn(x)
        x = self.norm2(x, ffn_output)
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = AddNorm(d_model) 
        
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = AddNorm(d_model)
        
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm3 = AddNorm(d_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(Q=x, K=x, V=x, mask=tgt_mask)
        x = self.norm1(x, attn_output)
        
        cross_attn_output = self.cross_attn(Q=x, K=enc_output, V=enc_output, mask=src_mask)
        x = self.norm2(x, cross_attn_output)

        ffn_output = self.ffn(x)
        x = self.norm3(x, ffn_output)
        
        return x