import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import time
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 【修改】恢復使用 sacrebleu，移除 evaluate
import sacrebleu 
# 【新增】導入 ROUGE，這是一個輕量級庫，不會有衝突
from rouge_score import rouge_scorer

# 【注意】移除了 import evaluate 和 import nltk

from src.myTransformer import Transformer
from src.units import create_padding_mask, create_decoder_self_attn_mask

class Config:
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_len = 128

    batch_size = 64
    epochs = 1
    learning_rate = 1e-4

    warmup_steps = 4000
    clip_grad_norm = 1.0
    src_lang = 'en'
    tgt_lang = 'de'
    
    seed = 42

    pad_token_id = 0
    src_vocab_size = 0
    tgt_vocab_size = 0
    tokenizer = None

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 【保留】輔助函數：計算模型參數
def count_parameters(model):
    """计算并打印模型的可训练参数量"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数量: {num_params:,}")
    return num_params

class TransformerLRSchedule(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.inv_sqrt_d_model = d_model ** (-0.5)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step == 0:
            return [0.0 for _ in self.optimizer.param_groups]

        term1 = step ** (-0.5)
        term2 = step * (self.warmup_steps ** (-1.5))
        
        lr = self.inv_sqrt_d_model * min(term1, term2)
        
        return [lr for _ in self.optimizer.param_groups]
        
def preprocess_function(examples, tokenizer, src_lang, tgt_lang, max_seq_len):
    source = [ex[src_lang] for ex in examples["translation"]]
    target = [ex[tgt_lang] for ex in examples["translation"]]

    model_input = tokenizer(source, max_length=max_seq_len, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target, max_length=max_seq_len, truncation=True, padding="max_length")

    model_input["decoder_input_ids"] = labels["input_ids"]
    
    model_input["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]

    return model_input

def load_and_prepare_data(config):
    print("--- 1. 正在加载和预处理数据 (使用 AutoTokenizer) ---")

    # 【修改】移除了 NLTK 下載代碼
    
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    config.src_vocab_size = tokenizer.vocab_size
    config.tgt_vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.tokenizer = tokenizer 

    raw_datasets = load_dataset("iwslt2017", f"iwslt2017-{config.src_lang}-{config.tgt_lang}")
    print("--- 从 Hugging Face Hub (或本地缓存) 加载数据集 ---")

    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, tokenizer, config.src_lang, config.tgt_lang, config.max_seq_len),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    tokenized_datasets.set_format(type="torch", columns=["input_ids", "decoder_input_ids", "labels"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=config.batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=config.batch_size)

    print(f"词汇表大小: {config.src_vocab_size}")
    print(f"訓練集 Batch 总数: {len(train_dataloader)}")
    print("------------------------------------------")
    
    return train_dataloader, eval_dataloader, config

def train_model(model, train_loader, eval_loader, config, steps_per_epoch, use_residual=True, use_layer_norm=True):
    print("--- 2. 开始训练模型 ---")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)

    scheduler = TransformerLRSchedule(optimizer, config.d_model, config.warmup_steps)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses_epoch = [] 
    eval_losses = []
    train_losses_batch = [] 

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            src_ids = batch["input_ids"].to(device)
            tgt_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            src_mask = create_padding_mask(src_ids, config.pad_token_id)
            tgt_mask = create_decoder_self_attn_mask(tgt_ids, config.pad_token_id)

            output = model(src_ids, tgt_ids, src_mask, tgt_mask)
            
            loss = criterion(output.view(-1, config.tgt_vocab_size), labels.view(-1))
            
            train_losses_batch.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if (step + 1) % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1} | Step {step+1} | LR: {scheduler.get_last_lr()[0]:.6f} | Avg Train Loss: {avg_loss:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_losses_epoch.append(avg_train_loss)

        model.eval()
        avg_eval_loss = evaluate_model(model, eval_loader, criterion, device, config)
        eval_losses.append(avg_eval_loss)

        # 【保留】計算並打印 Perplexity (PPL)
        try:
            val_ppl = math.exp(avg_eval_loss)
        except OverflowError:
            val_ppl = float('inf') 

        print(f"\n--- Epoch {epoch+1} 完成 ---")
        # 【修改】在日誌中加入 PPL
        print(f"训练损失: {avg_train_loss:.4f} | 验证损失: {avg_eval_loss:.4f} | 验证 PPL: {val_ppl:.2f} | 耗时: {time.time() - start_time:.2f}s")
        print("-" * 40)

    return train_losses_epoch, eval_losses, train_losses_batch

def evaluate_model(model, eval_loader, criterion, device, config):
    total_loss = 0
    with torch.no_grad():
        for batch in eval_loader:
            src_ids = batch["input_ids"].to(device)
            tgt_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)

            src_mask = create_padding_mask(src_ids, config.pad_token_id)
            tgt_mask = create_decoder_self_attn_mask(tgt_ids, config.pad_token_id)

            output = model(src_ids, tgt_ids, src_mask, tgt_mask)
            
            loss = criterion(output.view(-1, config.tgt_vocab_size), labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(eval_loader)

# 【核心功能】：解碼函數 (貪婪搜索)
def simple_greedy_decode(model, src_ids, src_mask, config, device):
    
    model.eval()
    batch_size = src_ids.size(0)
    
    enc_output = model.encoder(src_ids, src_mask)
    
    START_ID = config.tokenizer.eos_token_id if hasattr(config.tokenizer, 'eos_token_id') else 1 
    
    tgt_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * START_ID
    
    EOS_ID = config.tokenizer.eos_token_id if hasattr(config.tokenizer, 'eos_token_id') else 1
    
    for _ in range(config.max_seq_len - 1):
        
        tgt_mask = create_decoder_self_attn_mask(tgt_ids, config.pad_token_id)
        
        dec_output = model.decoder(
            tgt_seq=tgt_ids, 
            enc_output=enc_output, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask
        )
        
        output = model.output_head(dec_output) 
        
        logits = output[:, -1, :] 
        
        next_token_ids = torch.argmax(logits, dim=-1).unsqueeze(1) 
        
        tgt_ids = torch.cat([tgt_ids, next_token_ids], dim=1)
        
        if torch.all(next_token_ids == EOS_ID):
            break
            
    return tgt_ids


# 【修改】：使用 'sacrebleu' 和 'rouge_score' 計算指標
def calculate_metrics(model, eval_loader, config, device):
    print("\n--- 3. 开始在验证集上进行评估 (计算 BLEU, TER, ROUGE-L) ---")
    
    model.eval()
    all_hypotheses = []
    all_references_sacrebleu = [] # 格式: [[ref1], [ref2], ...]
    all_references_flat = []      # 格式: [ref1, ref2, ...]
    
    if not torch.isfinite(model.output_head.weight.data).all():
        print("警告: 模型参数包含 NaN/Inf。无法进行有意义的解码。")
        return {"BLEU": 0.0, "TER": 100.0, "ROUGE-L": 0.0}

    tokenizer = config.tokenizer 

    # 【新增】初始化 ROUGE Scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_rouge_l = 0

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if (i+1) % 20 == 0:
                print(f"  ... 正在解码 Batch {i+1}/{len(eval_loader)}")
                
            src_ids = batch["input_ids"].to(device)
            labels = batch["labels"].clone()
            src_mask = create_padding_mask(src_ids, config.pad_token_id)
            
            generated_ids = simple_greedy_decode(model, src_ids, src_mask, config, device)
            
            labels[labels == -100] = config.pad_token_id 
            
            hypotheses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_hypotheses.extend(hypotheses)
            all_references_sacrebleu.extend([[ref] for ref in references]) 
            all_references_flat.extend(references)

    # 確保不為空
    if not all_hypotheses or not all_references_sacrebleu:
        print("错误：解码结果或参考译文为空。无法计算指标。")
        return {"BLEU": 0.0, "TER": 100.0, "ROUGE-L": 0.0}

    # 4. 計算指標
    print("--- 正在汇总计算所有指标... ---")
    
    # (a) 計算 BLEU (使用 sacrebleu)
    bleu = sacrebleu.corpus_bleu(all_hypotheses, all_references_sacrebleu).score
    
    # (b) 計算 TER (使用 sacrebleu)
    ter = sacrebleu.corpus_ter(all_hypotheses, all_references_sacrebleu).score 
    
    # (c) 計算 ROUGE-L
    for hyp, ref in zip(all_hypotheses, all_references_flat):
        score = scorer.score(ref, hyp)
        total_rouge_l += score['rougeL'].fmeasure
    
    avg_rouge_l = (total_rouge_l / len(all_hypotheses)) * 100.0

    metrics = {
        "BLEU": bleu,
        "TER": ter,
        "ROUGE-L": avg_rouge_l
    }
    
    print("--- 评估完成 ---")
    # 【修改】打印 BLEU, TER, 和 ROUGE-L
    print(f"最终 BLEU: {metrics['BLEU']:.2f}")
    print(f"最终 TER: {metrics['TER']:.2f}")
    print(f"最终 ROUGE-L (F1): {metrics['ROUGE-L']:.2f}")
    print("-------------------")
    
    return metrics


def save_results(train_losses_epoch, eval_losses, train_losses_batch, config, steps_per_epoch):

    df_epoch = pd.DataFrame({
        'Epoch': range(1, config.epochs + 1),
        'Train_Loss_Epoch_Avg': train_losses_epoch,
        'Eval_Loss': eval_losses
    })
    
    os.makedirs('result', exist_ok=True) 
    df_epoch.to_csv(os.path.join('result', 'loss_history_epoch.csv'), index=False)

    num_batches = len(train_losses_batch)
    batch_steps = range(1, num_batches + 1)

    eval_x = [(e + 1) * steps_per_epoch for e in range(config.epochs)]
    
    plt.figure(figsize=(12, 7))

    plt.plot(batch_steps, train_losses_batch, label='Train Loss (Per Batch)', alpha=0.5, linewidth=1) 

    plt.plot(eval_x, eval_losses, label='Validation Loss (Epoch End)', color='red', marker='o', linewidth=2)
    
    plt.title('Training and Validation Loss Over Steps (Baseline)')
    plt.xlabel('Training Step (Batch Number)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('result', 'training_curve_batch.png'))
    print(f"Batch 级训练曲线图已保存到 result/training_curve_batch.png")
    

def main():
    config = Config()
    
    set_seed(config.seed) 
    
    train_loader, eval_loader, config = load_and_prepare_data(config)

    model = Transformer(config, use_residual=True, use_layer_norm=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 【保留】在實例化主模型後計算參數
    print("--- Full Model (Baseline) ---")
    count_parameters(model) 

    print("--- Ablation Model Parameter Counts ---")
    models_to_ablate = {
        "No LayerNorm (-LN)": {"residual": True, "layernorm": False},
        "No Residual (-Res)": {"residual": False, "layernorm": True},
        "No LN & No Res (-LN-Res)": {"residual": False, "layernorm": False},
    }
    
    for name, params in models_to_ablate.items():
        ablate_model = Transformer(config, use_residual=params['residual'], use_layer_norm=params['layernorm']).to(device)
        print(f"{name} ", end="")
        count_parameters(ablate_model)
    print("------------------------------------------")

    train_losses_epoch, eval_losses, train_losses_batch = train_model(
        model, train_loader, eval_loader, config, len(train_loader), use_residual=True, use_layer_norm=True
    )

    save_results(train_losses_epoch, eval_losses, train_losses_batch, config, len(train_loader))
    torch.save(model.state_dict(), os.path.join('result', 'final_transformer_weights.pth'))
    print("模型权重和损失历史已保存。")

    # 【保留】在訓練結束後調用評估
    final_metrics = calculate_metrics(model, eval_loader, config, device)
    
    metrics_df = pd.DataFrame([final_metrics])
    metrics_df.to_csv(os.path.join('result', 'final_metrics.csv'), index=False)
    print(f"最终评估指标已保存到 result/final_metrics.csv")


if __name__ == '__main__':
    main()