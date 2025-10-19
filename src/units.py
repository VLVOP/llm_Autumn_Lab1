import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):

    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, V)

    return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
    

class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q_proj = self.W_q(Q)
        K_proj = self.W_k(K)
        V_proj = self.W_v(V)

        def split_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        Q_split = split_heads(Q_proj)
        K_split = split_heads(K_proj)
        V_split = split_heads(V_proj)

        attn_output, attn_weights = scaled_dot_product_attention(Q_split, K_split, V_split, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(attn_output)
        return output
    
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
    
def create_padding_mask(seq, pad_idx):
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def create_future_mask(seq_len, device):
    mask = torch.triu(
        torch.ones((1, 1, seq_len, seq_len), device=device), 
        diagonal=1
    ) == 0
    return mask.float()

def create_decoder_self_attn_mask(tgt_seq, pad_idx):
    seq_len = tgt_seq.size(1)
    
    device = tgt_seq.device 
    
    look_ahead_mask = create_future_mask(seq_len, device=device)

    padding_mask = create_padding_mask(tgt_seq, pad_idx) 

    return padding_mask * look_ahead_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) 

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]