#!/bin/bash

# --- run.sh ---
# 用于一键运行 LAB1 的消融实验

echo "开始运行 LAB1 消融实验..."

# 1. 设置 MKL 环境变量 (解决 OMP 冲突)
echo "设置 KMP_DUPLICATE_LIB_OK=TRUE..."
export KMP_DUPLICATE_LIB_OK=TRUE

# 2. 运行消融实验主脚本
#    假设此脚本在 LAB1 根目录下运行
echo "运行 src/ablate.py..."
python ablate.py

echo "消融实验运行完成。结果已保存到 result/ 目录。"