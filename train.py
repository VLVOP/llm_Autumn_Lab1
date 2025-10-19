import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import time
import os

from src.myTransformer import Transformer
from src.units import create_padding_mask, create_future_mask, create_decoder_self_attn_mask

# --- 超参数 --- 

class Config:
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_len = 128

    batch_size = 64
    epochs = 5
    learning_rate = 1e-4

    warmup_steps = 4000
    clip_grad_norm = 1.0
    src_lang = 'en'
    tgt_lang = 'de'

    pad_token_id = 0
    src_vocab_size = 0
    tgt_vocab_size = 0

class TransformerLRSchedule(optim.lr_scheduler._LRScheduler):
    """
    Transformer 论文中的学习率调度器: 
    lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
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
    """Tokenize and prepare data, ensuring truncation and padding."""
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

    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    config.src_vocab_size = tokenizer.vocab_size
    config.tgt_vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id

    raw_datasets = load_dataset("iwslt2017", f"iwslt2017-{config.src_lang}-{config.tgt_lang}")

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
    print(f"训练集 Batch 总数: {len(train_dataloader)}")
    print("------------------------------------------")
    
    return train_dataloader, eval_dataloader, config

def train_model(model, train_loader, eval_loader, config):
    print("--- 2. 开始训练模型 ---")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)

    scheduler = TransformerLRSchedule(optimizer, config.d_model, config.warmup_steps)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    eval_losses = []
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            src_ids = batch["input_ids"].to(device)
            tgt_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            global_step += 1


            src_mask = create_padding_mask(src_ids, config.pad_token_id)

            tgt_mask = create_decoder_self_attn_mask(tgt_ids, config.pad_token_id)


            output = model(src_ids, tgt_ids, src_mask, tgt_mask)
            

            loss = criterion(output.view(-1, config.tgt_vocab_size), labels.view(-1))

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
        train_losses.append(avg_train_loss)

        model.eval()
        avg_eval_loss = evaluate_model(model, eval_loader, criterion, device, config)
        eval_losses.append(avg_eval_loss)

        print(f"\n--- Epoch {epoch+1} 完成 ---")
        print(f"训练损失: {avg_train_loss:.4f} | 验证损失: {avg_eval_loss:.4f} | 耗时: {time.time() - start_time:.2f}s")
        print("-" * 40)
        

    save_results(train_losses, eval_losses, config)

    torch.save(model.state_dict(), os.path.join('result', 'final_transformer_weights.pth'))
    print("模型权重和损失历史已保存。")

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

def save_results(train_losses, eval_losses, config):

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("警告: 缺少 pandas 或 matplotlib，无法保存图表。请安装以满足报告要求。")
        return

    df = pd.DataFrame({
        'Epoch': range(1, config.epochs + 1),
        'Train_Loss': train_losses,
        'Eval_Loss': eval_losses
    })
    
    os.makedirs('result', exist_ok=True) 
    df.to_csv(os.path.join('result', 'loss_history.csv'), index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Train_Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Eval_Loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('result', 'training_curve.png'))

def main():
    config = Config()
    
    model = Transformer(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {num_params:,}") 

    train_loader, eval_loader, config = load_and_prepare_data(config)

    model = Transformer(config)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    train_model(model, train_loader, eval_loader, config)


if __name__ == '__main__':
    main()