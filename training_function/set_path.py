import os


# 於所在資料夾內創建 TrainingData、trained_model_scaler、trained_model、output_Data 資料夾
# return folder_path_dict
# folder_path_dict 格式 : { str(資料夾名稱) : str(資料夾路徑) }
def initialization():
    path = os.getcwd()
    # 創建資料夾之名稱
    folder_list = [
        "TrainingData",
        "trained_model_scaler",
        "trained_model",
        "output_Data",
    ]
    folder_path_dict = {}
    # 創建資料夾
    for name in folder_list:
        folder_path = os.path.join(path, name)
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"成功創建資料夾: {folder_path}")
            folder_path_dict[name] = folder_path
        except OSError as e:
            print(f"無法創建資料夾: {folder_path}，錯誤訊息: {e}")
            folder_path_dict[name] = None
    return folder_path_dict
