import training_function.set_path as set_path
import training_function.save_data as save_data
import training_function.load_data as load_data
from training_function.model_architecture.CNN_2 import test_cnn_model
import os
import joblib
import numpy as np
import pandas as pd


# 請使用 data_set_up_v7 輸出之 test_data_v7.pkl 進行訓練
# 初始化資料夾
path = set_path.initialization()

# 版本設置
ver = "v44"
os.makedirs(os.path.join(path["output_Data"], ver))

# 儲存模型檔案位置
model_path = os.path.join(path["trained_model"], ver)
# 儲存 scaler 位置
scaler_path = os.path.join(path["trained_model_scaler"], ver)

csv_folder_path = os.path.join(path["output_Data"], ver)

# 讀取測試資料 (訓練資料最大長度為 4696，最小為 272) (測試資料最大長度為 15355，最小為 128)
dataframe = pd.read_pickle(os.path.join(path["TrainingData"], "test_data_v7.pkl"))

features_columns = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]

sensor_features_data = pd.concat(
    [dataframe[features_columns], dataframe["mode"].astype("int32")], axis=1
)


# 正規化
def scale_list(scaler, lst):
    return scaler.transform(np.array(lst).reshape(-1, 1)).flatten().tolist()


def transform_data(scaler, df, target_col):
    for col in target_col:
        df.loc[:, col] = df[col].apply(lambda x: scale_list(scaler, x))
    return df


for i in features_columns:
    scaler = joblib.load(os.path.join(scaler_path, f"{i}.pkl"))
    sensor_features_data = transform_data(scaler, sensor_features_data, [i])

# 資料預處理
sensor_features_data.loc[:, "mode"] = (
    sensor_features_data["mode"].apply(lambda x: x - 1).astype("int32")
)

model_param = load_data.load_dict(
    path=model_path,
    file_extension=".json",
    function=3,
    file_name_list=[
        "gender.json",
        "hold racket handed.json",
        "level.json",
        "play years.json",
    ],
)

gender_df = test_cnn_model(
    sensor_features_data,
    batch_size=1,
    kernel_size=model_param["gender.json"]["kernel_size"],
    num_classes=2,
    model_param_list=model_param["gender.json"]["model_param_list"],
    features_name="gender",
    model_path=os.path.join(model_path, "gender.pth"),
)
hold_df = test_cnn_model(
    sensor_features_data,
    batch_size=1,
    kernel_size=model_param["hold racket handed.json"]["kernel_size"],
    num_classes=2,
    model_param_list=model_param["hold racket handed.json"]["model_param_list"],
    features_name="hold racket handed",
    model_path=os.path.join(model_path, "hold racket handed.pth"),
)
play_df = test_cnn_model(
    sensor_features_data,
    batch_size=1,
    kernel_size=model_param["play years.json"]["kernel_size"],
    num_classes=3,
    model_param_list=model_param["play years.json"]["model_param_list"],
    features_name="play years",
    model_path=os.path.join(model_path, "play years.pth"),
)
level_df = test_cnn_model(
    sensor_features_data,
    batch_size=1,
    kernel_size=model_param["level.json"]["kernel_size"],
    num_classes=4,
    model_param_list=model_param["level.json"]["model_param_list"],
    features_name="level",
    model_path=os.path.join(model_path, "level.pth"),
)

gender_df = gender_df.rename(columns={"gender_0": "gender"})
hold_df = hold_df.rename(columns={"hold racket handed_0": "hold racket handed"})
level_df = level_df.rename(
    columns={
        "level_0": "level_2",
        "level_1": "level_3",
        "level_2": "level_4",
        "level_3": "level_5",
    }
)

combined_df = pd.concat(
    [
        dataframe["unique_id"],
        gender_df["gender"],
        hold_df["hold racket handed"],
        play_df,
        level_df,
    ],
    axis=1,
)

combined_df = combined_df.round(8)

# 儲存 csv
saving_csv_param = {
    "path": csv_folder_path,  # 資料夾絕對路徑
    "dataframe_dict": {
        "result": combined_df,
    },  # 檔名及儲存資料
    "function": 2,  # 檔名存在時的處理方式
    "full_auto": True,  # function = 2 時，是否自動命名
    "encoding": "utf-8",  # 編碼方式
}
try:
    print("儲存 csv 檔案:")
    saving_csv_result_dict = save_data.saving_csv(**saving_csv_param)
    print("saving_csv_result_dict :")
    for key, value in saving_csv_result_dict.items():
        print(key)
        for x, y in value.items():
            print(f"{x} : {y}")
except Exception as e:
    print(f"saving_csv 出現錯誤 : {e}")
