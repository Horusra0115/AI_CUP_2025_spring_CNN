# AI_CUP_2025_spring_CNN

這是一個用於 **桌球智慧球拍資料的精準分析競賽** 的 CNN 模型訓練程式碼，可預測持拍者的性別（gender）、持拍手（hold racket handed）、球齡（play years）與水平（level）。

## 目錄

- [專案概述](#專案概述)
- [運行環境](#運行環境)
- [使用方式](#使用方式)

---

## 專案概述

本專案基於 PyTorch 實現四個獨立的 CNN 模型，分別預測以下目標：
1. **性別（gender）**：二分類（男/女）
2. **持拍手（hold racket handed）**：二分類（左/右）
3. **球齡（play years）**：多分類（低/中/高）
4. **水平（level）**：多分類（大專甲組/大專乙組/青少年國手/青少年選手）

---

## 運行環境

### 核心依賴
```markdown
- Python 3.10.18
- PyTorch 2.5.1 (+CUDA 12.4 如使用 GPU)
- 其他套件：
  numpy==2.2.6
  pandas==2.3.0
  scikit-learn==1.7.0
  joblib==1.5.1
  ```
運行本專案時，請務必確保核心依賴已安裝完成。

---

## 使用方式

### 初始化設置
```bash
python path_set_up.py  # 自動創建以下目錄結構
```
生成的目錄結構：
```
.
├── TrainingData/            # 存放原始訓練資料
├── trained_model/           # 保存訓練好的模型權重
├── trained_model_scaler/    # 保存預處理的 Scaler 物件
└── output_Data/             # 預測結果輸出
```



