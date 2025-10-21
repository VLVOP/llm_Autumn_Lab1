import torch
import os
import matplotlib.pyplot as plt
import pandas as pd

# 从 train 模块导入所有必要的函数
# 注意：假设 calculate_metrics 在 train.py 中定义
from train import Config, set_seed, load_and_prepare_data, train_model, calculate_metrics 
from src.myTransformer import Transformer

def run_ablation_experiment(config, exp_name, use_residual, use_layer_norm):
    """
    运行单个消融实验并返回损失历史和 BLEU/TER 指标
    """
    print(f"\n--- 运行消融实验: {exp_name} ---")
    
    set_seed(config.seed) 
    
    # 1. 加载数据 (只在 run_ablation_experiment 内部加载一次)
    train_loader, eval_loader, config = load_and_prepare_data(config)

    # 2. 实例化模型，传入消融参数
    model = Transformer(
        config, 
        use_residual=use_residual, 
        use_layer_norm=use_layer_norm
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. 【修复点 1】：正确接收 train_model 的 3 个返回值
    train_loss_epoch, eval_loss, train_loss_batch = train_model(
        model, 
        train_loader, 
        eval_loader, 
        config, 
        steps_per_epoch=len(train_loader),
        use_residual=use_residual, 
        use_layer_norm=use_layer_norm
    )
    
    # 4. 【修复点 2】：确保 config 上的消融参数是最新的，供 calculate_metrics 使用
    config.use_residual = use_residual
    config.use_layer_norm = use_layer_norm
    
    # 5. 计算 BLEU/TER 分数
    # 注意：我们使用 eval_loader，但无需再次调用 load_and_prepare_data
    final_bleu, final_ter = calculate_metrics(model, eval_loader, config, device) 

    # 6. 【修复点 3】：返回正确的 epoch 损失变量名
    return train_loss_epoch, eval_loss, final_bleu, final_ter 


def main_ablation():
    """主消融实验函数"""
    
    config = Config()

    experiments = {
        "Full Model (Baseline)": {"residual": True, "layernorm": True, "color": 'blue'},
        "No LayerNorm (-LN)": {"residual": True, "layernorm": False, "color": 'orange'},
        "No Residual (-Res)": {"residual": False, "layernorm": True, "color": 'red'},
        "No LN & No Res (-LN-Res)": {"residual": False, "layernorm": False, "color": 'green'},
    }

    all_eval_losses = {}
    all_results = {}
    epochs = range(1, config.epochs + 1)

    for name, params in experiments.items():
        # 【修复点 4】：这里接收 run_ablation_experiment 返回的 4 个值
        train_loss_epoch, eval_loss, final_bleu, final_ter = run_ablation_experiment(
            config, name, params['residual'], params['layernorm']
        )
        
        all_results[name] = {
            "eval_loss": eval_loss, 
            "bleu": final_bleu, 
            "ter": final_ter  
        }

        all_eval_losses[name] = eval_loss

        print(f"实验 {name} 最终 BLEU: {final_bleu:.2f} | 最终 TER: {final_ter:.2f}")


    print("\n--- 绘制消融实验对比图 ---")
    plt.figure(figsize=(10, 7))
    
    os.makedirs('result', exist_ok=True)

    for name, losses in all_eval_losses.items():
        color = experiments[name]['color']
        plt.plot(epochs, losses, label=f"{name}", color=color, linewidth=2)

    plt.title('Ablation Study: Impact of LayerNorm and Residual Connections')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss (Average)')
    plt.legend()
    plt.grid(True)
    
    fig_filename_loss = os.path.join('result', 'ablation_study_comparison.png')
    plt.savefig(fig_filename_loss)
    print(f"损失对比图已保存到 {fig_filename_loss}")

    metrics_data = []
    for name, result in all_results.items():
        metrics_data.append([
            name, 
            f"{result['eval_loss'][-1]:.4f}", 
            f"{result['bleu']:.2f}",
            f"{result['ter']:.2f}"
        ])
        
    df_metrics = pd.DataFrame(metrics_data, columns=['Model', 'Final Val Loss', 'BLEU Score', 'TER Score'])
    print("\n指标对比:")
    print(df_metrics.to_string(index=False)) 


if __name__ == '__main__':
    main_ablation()