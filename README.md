# AI_CUP_2025_spring_CNN

這是一個用於 **桌球智慧球拍資料的精準分析競賽** 的 CNN 模型訓練程式碼，可預測持拍者的性別（gender）、持拍手（hold racket handed）、球齡（play years）與水平（level）。

## 目錄

- [專案概述](#專案概述)
- [特色](#特色)
- [運行環境](#運行環境)
- [使用方式](#使用方式)

---

## 專案概述

本專案基於 PyTorch 實作四個獨立的 1D CNN 模型，分別對應以下四項預測任務：
1. **性別（gender）**：二分類（男/女）
2. **持拍手（hold racket handed）**：二分類（左/右）
3. **球齡（play years）**：三分類（低/中/高）
4. **水平（level）**：四分類（大專甲組/大專乙組/青少年國手/青少年選手）

---

## 特色

1. 使用 `ver` 變數統一管理模型版本，避免實驗混淆
2. 自動記錄模型架構參數（`kernel_size` 與 `model_param_list`）
3. 自動儲存訓練所使用的 `StandardScaler`，方便模型重現  
4. 使用最暴力的方式(強制報錯)，避免使用者覆蓋已存在的資料
5. 宛如施工現場般雜亂不堪的代碼

---

## 運行環境

### 核心依賴
```

Python 3.10.18
PyTorch 2.5.1  （支援 CUDA 12.4，如使用 GPU）
numpy==2.2.6
pandas==2.3.0
scikit-learn==1.7.0
joblib==1.5.1

````
⚠️ 請確認上述套件版本已安裝，以確保程式正常執行。

---

## 使用方式

### 1. 建立目錄結構
執行以下指令建立所需資料夾：
```bash
python path_set_up.py
````

執行後的目錄結構：
```
.
├── TrainingData/            # 存放原始訓練資料
├── trained_model/           # 儲存訓練好的模型權重及參數
├── trained_model_scaler/    # 儲存對應的 Scaler 物件
├── output_Data/             # 輸出測試預測結果
├── training_function/
├── path_set_up.py
├── data_set_up_v7.py
├── training_cnn_v7.py
└── testing_cnn_v7.py
```

### 2. 資料準備

請將主辦單位提供的以下資料，放入 `TrainingData/` 資料夾中：

* `train_info.csv`
* `test_info.csv`
* `train_data` 資料夾
* `test_data` 資料夾

接著執行以下指令生成訓練/測試資料集：

```bash
python data_set_up_v7.py
```

### 3. 訓練模型

* 請先於 `training_cnn_v7.py` 中設定 `ver` 變數（用來標記當前訓練版本）
* 若需調整訓練參數，請修改 `train_cnn_model()` 函數的參數

執行模型訓練：

```bash
python training_cnn_v7.py
```

### 4. 測試模型

* 請先於 `testing_cnn_v7.py` 中設定 `ver` 變數（需與訓練版本一致）
* 確保對應的模型與 scaler 已正確儲存

執行模型推論：

```bash
python testing_cnn_v7.py
```



