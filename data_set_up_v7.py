import training_function.set_path as set_path
import training_function.load_data as load_data
import training_function.save_data as save_data
import training_function.General_functions as General_functions
import training_function.dataframe_set_up as dataframe_set_up
import os

# 這段代碼用來設置訓練及測試用資料 (請使用 training_cnn_v7.py 及 testing_cnn_v7.py 進行訓練及測試)
# 使用 training_cnn_v7.py 及 testing_cnn_v7.py 時，請確保 .pkl 檔案已生成，且數據正確 (查看 .csv)

function = General_functions.function_process(
    function=None, function_range={1: "設置訓練資料", 2: "設置測試資料"}
)
if function == 1:
    file_name = "train"
elif function == 2:
    file_name = "test"
else:
    raise ValueError("未選擇有效功能")

# 初始化資料夾
path = set_path.initialization()
# 設置版本號(更改時請確保與檔名對應)
version = "v7"
# txt 檔案資料夾位置
txt_folder_path = os.path.join(path["TrainingData"], f"{file_name}_data")
# train_info 或 test_info
load_csv_flie_name = f"{file_name}_info.csv"
# 儲存後的檔名
save_csv_flie_name = f"{file_name}_data_{version}"
save_pkl_flie_name = f"{file_name}_data_{version}.pkl"

# 讀取 csv 檔案
load_csv_param = {
    "path": path["TrainingData"],
    "function": 3,
    "file_name_list": [load_csv_flie_name],
    "encoding": "utf-8",
    "axis": 0,
    "drop_duplicates": False,
    "ignore_index": True,
}
try:
    print("讀取 csv 檔案:")
    dataframe, dataframe_info = load_data.load_csv(**load_csv_param)
except Exception as e:
    print(f"load_csv 出現錯誤 : {e}")

dataframe = dataframe_set_up.table_tennis_txt_input(dataframe, txt_folder_path)
print("txt 檔案資料提取完成")

# 儲存 csv (檢視用)
saving_csv_param = {
    "path": path["TrainingData"],
    "dataframe_dict": {
        save_csv_flie_name: dataframe,
        f"{save_csv_flie_name}_dataframe_info": dataframe_info,
    },
    "function": 2,
    "full_auto": True,
    "encoding": "utf-8",
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

# 儲存 pkl (正式使用) (不會自動檢測檔名，請注意)
dataframe.to_pickle(os.path.join(path["TrainingData"], save_pkl_flie_name))