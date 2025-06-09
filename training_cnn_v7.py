import training_function.set_path as set_path
import training_function.save_data as save_data
from training_function.model_architecture.CNN_2 import train_cnn_model
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 請使用 data_set_up_v7 輸出之 train_data_v7.pkl 進行訓練
# 初始化資料夾
path = set_path.initialization()

# 版本設置
ver = "v44"
os.makedirs(os.path.join(path["trained_model"], ver))
os.makedirs(os.path.join(path["trained_model_scaler"], ver))

# 儲存模型檔案位置
model_path = os.path.join(path["trained_model"], ver)
# 儲存 scaler 位置
scaler_path = os.path.join(path["trained_model_scaler"], ver)

# 讀取訓練資料 (訓練資料最大長度為 4696，最小為 272) (測試資料最大長度為 15355，最小為 128)
dataframe = pd.read_pickle(os.path.join(path["TrainingData"], "train_data_v7.pkl"))

features_columns = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]

features_columns_27 = []

for i in features_columns:
    for num in range(27):
        features_columns_27.append(f"{i}_{num+1}")


# 正規化
def fit_scaler(df, target_col):
    all_values = np.concatenate(
        [np.concatenate(df[col].values) for col in target_col]
    ).reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(all_values)
    return scaler


def scale_list(scaler, lst):
    return scaler.transform(np.array(lst).reshape(-1, 1)).flatten().tolist()


def transform_data(scaler, df, target_col):
    for col in target_col:
        df.loc[:, col] = df[col].apply(lambda x: scale_list(scaler, x))
    return df


for i in features_columns:
    scaler = fit_scaler(dataframe, [i])
    joblib.dump(scaler, os.path.join(scaler_path, f"{i}.pkl"))
    target_col = []
    for num in range(27):
        target_col.append(f"{i}_{num+1}")
    dataframe = transform_data(scaler, dataframe, target_col)

data = []
for num in range(27):
    x = None
    target_col = []
    for i in features_columns:
        target_col.append(f"{i}_{num+1}")
    x = pd.concat(
        [
            dataframe[target_col],
            dataframe["mode"],
            dataframe[["gender", "hold racket handed", "play years", "level"]],
        ],
        axis=1,
    )

    for i in features_columns:
        x.rename(columns={f"{i}_{num+1}": f"{i}"}, inplace=True)
    data.append(x)

data = pd.concat(data, axis=0, ignore_index=True)

sensor_features_data = pd.concat(
    [data[features_columns], data["mode"].astype("int32")],
    axis=1,
)
output_data = data[["gender", "hold racket handed", "play years", "level"]].astype(
    "int32"
)

# 資料預處理
sensor_features_data.loc[:, "mode"] = (
    sensor_features_data["mode"].apply(lambda x: x - 1).astype("int32")
)
output_data.loc[:, "gender"] = (
    output_data["gender"].apply(lambda x: x - 1).astype("int32")
)
# 訓練分類性別模型
_, gender_kernel_size, gender_model_param_list = train_cnn_model(
    sensor_features_data,
    output_data["gender"],
    kernel_size=[7, 5, 5, 5],
    num_classes=2,
    model_param_list=[64, 128, 256, 128, 64, 32],
    batch_size=128,
    num_epochs=100,
    model_path=os.path.join(model_path, "gender.pth"),
)

# 資料預處理
output_data.loc[:, "hold racket handed"] = (
    output_data["hold racket handed"].apply(lambda x: x - 1).astype("int32")
)
# 訓練分類持拍手模型
_, hold_kernel_size, hold_model_param_list = train_cnn_model(
    sensor_features_data,
    output_data["hold racket handed"],
    kernel_size=[7, 5, 5, 5],
    num_classes=2,
    model_param_list=[64, 128, 256, 128, 64, 32],
    batch_size=128,
    num_epochs=100,
    model_path=os.path.join(model_path, "hold racket handed.pth"),
)

# 訓練分類球齡模型
_, years_kernel_size, years_model_param_list = train_cnn_model(
    sensor_features_data,
    output_data["play years"],
    kernel_size=[7, 5, 5, 5],
    num_classes=3,
    model_param_list=[64, 128, 256, 128, 64, 32],
    batch_size=128,
    num_epochs=100,
    model_path=os.path.join(model_path, "play years.pth"),
)

# 資料預處理
output_data.loc[:, "level"] = (
    output_data["level"].apply(lambda x: x - 2).astype("int32")
)
# 訓練分類等級模型
_, level_kernel_size, level_model_param_list = train_cnn_model(
    sensor_features_data,
    output_data["level"],
    kernel_size=[7, 5, 5, 5],
    num_classes=4,
    model_param_list=[64, 128, 256, 128, 64, 32],
    batch_size=128,
    num_epochs=100,
    model_path=os.path.join(model_path, "level.pth"),
)

result = save_data.saving_dict(
    path=model_path,
    data_dict={
        "gender": {
            "model_param_list": gender_model_param_list,
            "kernel_size": gender_kernel_size,
        },
        "hold racket handed": {
            "model_param_list": hold_model_param_list,
            "kernel_size": hold_kernel_size,
        },
        "play years": {
            "model_param_list": years_model_param_list,
            "kernel_size": years_kernel_size,
        },
        "level": {
            "model_param_list": level_model_param_list,
            "kernel_size": level_kernel_size,
        },
    },
    file_extension=".json",
    function=2,
    full_auto=True,
)
for key, value in result.items():
    print(f"{key} : {value}")
