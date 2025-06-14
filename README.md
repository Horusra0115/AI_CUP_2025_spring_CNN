# AI_CUP_2025_spring_CNN

這是一個用於 桌球智慧球拍資料的精準分析競賽 的 CNN模型 訓練程式碼。

## 目錄

- [專案概述](#專案概述)
- [運行環境](#運行環境)
- [使用方式](#使用方式)


## 專案概述

> 本專案旨在訓練四個 CNN 模型，用於對持拍者的gender、hold racket handed、play years與level進行預測。程式碼基於 PyTorch 實現。

## 運行環境

請先確保下列所需環境已安裝完成

### 建議版本

- Python 3.10.18
- PyTorch 2.5.1
- CUDA 12.4
- numpy 2.2.6
- pandas 2.3.0
- joblib 1.5.1
- scikit-learn 1.7.0
- scipy 1.15.2


## 使用方式

### 設置資料夾
請先執行 path_set_up.py 設置所需資料夾

執行後，會創建四個資料夾

project/
├── data/
│   ├── raw/          # 原始資料
│   ├── processed/    # 處理後的資料
│   └── splits/       # 訓練/驗證/測試集分割
├── models/           # 訓練好的模型權重
│   ├── checkpoint.pth
│   └── best_model.pth
├── notebooks/        # Jupyter 筆記本
├── scripts/         # 預處理或工具腳本
│   ├── preprocess.py
│   └── utils.py
└── src/             # 主要程式碼
    ├── train.py
    └── eval.py

