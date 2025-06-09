import re
import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from .General_functions import dataframe_column_check


# 設置處理資料時需要之 dict
def base_dict_initialization():
    return {"Ax": [], "Ay": [], "Az": [], "Gx": [], "Gy": [], "Gz": []}


# 將 txt_folder_path 中 txt 文件的資料，按照 dataframe 中的 unique_id 欄位值，轉移至指定欄位
def table_tennis_txt_input(dataframe, txt_folder_path):
    # 創建所需欄位
    data_groups_key = list(base_dict_initialization().keys())
    new_columns_list = []
    for key in data_groups_key:
        new_columns_list.append(key)
        new_columns_list.extend([f"{key}_{i}" for i in range(1, 28)])
    new_columns = {
        col: pd.Series([None] * len(dataframe), dtype=object)
        for col in new_columns_list
    }
    dataframe = pd.concat([dataframe, pd.DataFrame(new_columns)], axis=1)

    # 讀取對應 txt 檔案，並將 Ax Ay Az Gx Gy Gz 等欄位存入 dataframe 內，並處理 dataframe 中的 cut_point 將其轉換成 list
    # 後按照 cut_point 對 Ax Ay Az Gx Gy Gz 進行切分
    for idx in range(len(dataframe)):
        data_groups = base_dict_initialization()
        cut_point = re.findall(r"\d+", dataframe.loc[idx, "cut_point"])
        cut_point = [int(num) for num in cut_point]
        dataframe.at[idx, "cut_point"] = cut_point
        with open(
            os.path.join(txt_folder_path, f"{dataframe.loc[idx, 'unique_id']}.txt"), "r"
        ) as file:
            for line in file:
                cut_txt_value = line.strip().split()
                for i in range(6):
                    key = data_groups_key[i]
                    data_groups[key].append(int(cut_txt_value[i]))
        for key, values in data_groups.items():
            dataframe.at[idx, key] = values
            for cut in range(len(cut_point) - 1):
                cut_list = values[int(cut_point[cut]) : int(cut_point[cut + 1])]
                dataframe.at[idx, f"{key}_{cut+1}"] = cut_list
    return dataframe


# 資料進行特徵提取
def extract_features(data):
    return {
        "max": np.max(data),
        "min": np.min(data),
        "mean": np.mean(data),
        "std": np.std(data),
        "ptp": np.ptp(data),
        "rms": np.sqrt(np.mean(np.square(data))),
        "energy": np.sum(np.square(data)),
    }


# 對指定之 dataframe 中的欄位進行特徵提取
def get_features(dataframe, target_column_list):
    data_features_key = list(extract_features([0, 1]).keys())
    dataframe_column_check(
        dataframe=dataframe, target_column_list=target_column_list, display_info=True
    )
    if not target_column_list:
        raise ValueError("target_column_list 無效")
    # 創建所需欄位
    new_features_columns = []
    for key in target_column_list:
        new_features_columns.extend([f"{key}_{i}" for i in data_features_key])
    new_feature_df = pd.DataFrame(
        np.nan, index=dataframe.index, columns=new_features_columns, dtype="float64"
    )
    dataframe = pd.concat([dataframe, new_feature_df], axis=1)

    # 將數據添加進對應的新創建之欄位內
    for idx in range(len(dataframe)):
        for column in target_column_list:
            features_dict = extract_features(dataframe.loc[idx, column])
            for key, value in features_dict.items():
                dataframe.at[idx, f"{column}_{key}"] = value
    return dataframe


# 計算FFT
def compute_fft(data, sampling_rate):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, 1 / sampling_rate)[: n // 2]  # 取正頻率部分
    magnitude = 2 / n * np.abs(yf[: n // 2])  # 計算振幅
    return xf, magnitude


# 提取主要頻率
def extract_main_frequencies(xf, magnitude, top_n=3):
    # 找出所有的峰值索引
    peaks, _ = find_peaks(magnitude)

    # 取出這些峰值對應的 magnitude 和頻率
    peak_magnitudes = magnitude[peaks]
    peak_frequencies = xf[peaks]

    # 根據 magnitude 做排序（從高到低）
    sorted_indices = np.argsort(peak_magnitudes)[::-1]

    # 取出前 top_n 個頻率和對應的振幅
    main_freqs = peak_frequencies[sorted_indices[:top_n]]
    main_mags = peak_magnitudes[sorted_indices[:top_n]]

    return list(zip(main_freqs, main_mags))


# 提取主頻並加入 dataframe 中
def get_fft_features(dataframe, sampling_rate, top_n):
    # 生成不带数字的列名
    base_cols = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
    data_features_key = [f"fft_{i}" for i in range(1, top_n + 1)]
    # 創建所需欄位
    new_features_columns = []
    for key in base_cols:
        new_features_columns.extend([f"{key}_{i}_freq" for i in data_features_key])
        new_features_columns.extend([f"{key}_{i}_mag" for i in data_features_key])
    new_feature_df = pd.DataFrame(
        np.nan, index=dataframe.index, columns=new_features_columns, dtype="float64"
    )
    dataframe = pd.concat([dataframe, new_feature_df], axis=1)
    # 將數據添加進對應的新創建之欄位內
    for idx in range(len(dataframe)):
        for column in base_cols:
            xf, magnitude = compute_fft(
                np.array(dataframe.loc[idx, column]), sampling_rate
            )
            data_list = extract_main_frequencies(xf, magnitude, top_n)
            for index, data in enumerate(data_list):
                dataframe.at[idx, f"{column}_{data_features_key[index]}_freq"] = data[0]
                dataframe.at[idx, f"{column}_{data_features_key[index]}_mag"] = data[1]
    return dataframe
