import os
import json
import pandas as pd
from .General_functions import select_files, path_check, param_type_check


# 設置 encoding_type_support_check 函數編碼格式支援(可擴充)
SUPPORTED_ENCODING = ["utf-8", "utf-8-sig", "gbk", "big5", "latin1"]


# 使用 SUPPORTED_ENCODING 檢查 encoding 是否為支援的編碼格式
# encoding 為需要檢查之編碼格式
# return 為 bool 值，函數處理錯誤時返回 False
def encoding_type_support_check(encoding):
    try:
        # 檢查輸入資料類型
        param_type_check(param=encoding, param_name="encoding", expected_type=str)

        return encoding in SUPPORTED_ENCODING
    except TypeError as e:
        print(f"encoding_type_support_check 函數處理發生錯誤 : {e}")
        return False


# 檢查 read_concat_csv 和 read_dict 函數中 files_path 參數是否符合要求
# files_path 為 read_concat_csv 和 read_dict 函數中的 files_path 參數
# return 為 bool 值，函數處理錯誤時返回 False
def files_path_check(files_path):
    try:
        # 檢查輸入資料類型
        param_type_check(param=files_path, param_name="files_path", expected_type=list)
        if not files_path:
            raise ValueError("files_path 不得為空")
        if not all(
            isinstance(path, str) and bool(path) and path_check(path=path, function=4)
            for path in files_path
        ):
            raise ValueError("files_path 中的檔案絕對路徑資料必須是 str 類型且不為空")

        return True
    except (TypeError, ValueError) as e:
        print(f"files_path_check 函數處理發生錯誤 : {e}")
        return False


# load_csv 函數之衍生函數，處理 csv 檔案之讀取與合併
# files_path 為所需讀取之 csv 檔案的絕對路徑組成的 list
# encoding 設置編碼格式
# axis 設置 pd.concat() 函數的合併方式 (對於 1 的優化尚未完全，建議使用預設值 0)
# drop_duplicates 設定是否移除資料的重複行
# ignore_index 設定合併 DataFrame 時是否忽略原始索引並重新編排
# message 為自訂合併完成時所顯示的提示詞 (不設定也可正常運行)
# return combined_data(合併完成後的 dataframe) 及 data_info，函數處理錯誤時返回 None, None
# data_info : 一個 dataframe 紀錄 combined_data 中所包含的 dataframe 及是否移除 combined_data 的重複行
def read_concat_csv(
    files_path,
    encoding="utf-8",
    axis=0,
    drop_duplicates=False,
    ignore_index=True,
    message="",
):
    try:
        # 檢查輸入資料類型
        if not files_path_check(files_path=files_path):
            raise TypeError("files_path 不符合規範")
        if not encoding_type_support_check(encoding=encoding):
            raise ValueError(f"不支援 {encoding} 編碼格式")
        param_type_check(param=axis, param_name="axis", expected_type=int)
        if axis not in [0, 1]:
            raise ValueError(f"axis 不支援 {axis}")
        param_type_check(
            param=drop_duplicates, param_name="drop_duplicates", expected_type=bool
        )
        param_type_check(
            param=ignore_index, param_name="ignore_index", expected_type=bool
        )
        param_type_check(param=message, param_name="message", expected_type=str)

        # data_info 設置
        data_info = pd.DataFrame(
            columns=["dataframe中所包含的檔案位置", "drop_duplicates"]
        )
        data_list = []
        # 讀取資料(可優化)
        for csv_path in files_path:
            try:
                csv_data = pd.read_csv(csv_path, encoding=encoding)
                if csv_data.empty:
                    print(f"{csv_path} 檔案內無資料，已跳過處理")
                else:
                    data_list.append(csv_data)
                    data_info.loc[len(data_info)] = [csv_path, drop_duplicates]
                    print(f"成功讀取 {csv_path}")
            except (FileNotFoundError, pd.errors.ParserError, UnicodeDecodeError) as e:
                print(f"讀取文件 {csv_path} 時發生錯誤 : {e}")
        if not data_list:
            raise FileNotFoundError("未能成功讀取任何 CSV 檔案")
        # 合併資料
        combined_data = pd.concat(data_list, axis=axis, ignore_index=ignore_index)
        # 移除資料的重複行
        if drop_duplicates:
            print("正在移除重複行...")
            before = len(combined_data)
            combined_data.drop_duplicates(inplace=True)
            print(f"已移除 {before - len(combined_data)} 行重複數據")
        print(message or "所有資料已讀取並處理")
        print(
            f"共成功讀取 {len(data_list)} 個檔案，合併後共有 {len(combined_data)} 筆資料"
        )
        return combined_data, data_info
    except (FileNotFoundError, TypeError, ValueError) as e:
        print(f"read_concat_csv 函數處理發生錯誤 : {e}")
        return None, None


# 讀入 csv 檔案
# path 為需要讀取之 csv 檔案所在的資料夾的絕對路徑
# function 為讀取方式，可設置為 1 - 3
# function = 1 : 選擇全部檔案
# function = 2 : 選擇部分檔案
# function = 3 : 選擇指定檔案
# file_name_list 只有當 function = 3 時才需設置，內容為資料夾內的 csv 檔案名稱(須包含副檔名)
# encoding 設置編碼格式
# axis 設置 pd.concat() 函數的合併方式
# drop_duplicates 設定是否移除資料的重複行
# ignore_index 設定合併 DataFrame 時是否忽略原始索引並重新編排
# return data(合併完成後的 dataframe) 及 data_info，函數處理錯誤時返回 None, None
def load_csv(
    path,
    function=None,
    file_name_list=None,
    encoding="utf-8",
    axis=0,
    drop_duplicates=False,
    ignore_index=True,
):
    try:
        # 選擇檔案
        files_path_list = select_files(
            path=path,
            file_extension=".csv",
            function=function,
            file_name_list=file_name_list,
        )
        if not files_path_list:
            raise ValueError("未選擇檔案")
        # 讀取檔案
        data, data_info = read_concat_csv(
            files_path=files_path_list,
            encoding=encoding,
            axis=axis,
            drop_duplicates=drop_duplicates,
            ignore_index=ignore_index,
        )
        if data is None or data_info is None:
            raise ValueError("讀取 csv 檔案時發生錯誤")
        return data, data_info
    except ValueError as e:
        print(f"load_csv 函數處理發生錯誤 : {e}")
        return None, None


# load_dict 函數之衍生函數，處理 json 檔案或 txt 檔案中之 dict 讀取
# files_path 為所需讀取之 json 檔案或 txt 檔案的絕對路徑組成的 list
# encoding 設置編碼格式
# return data_dict，函數處理錯誤時返回 {}
# data_dict 格式 : { str(檔案名稱) : dict(檔案內容) }，如果文件讀取失敗則對應值為 None
def read_dict(files_path, encoding="utf-8"):
    try:
        # 檢查輸入資料類型
        if not files_path_check(files_path=files_path):
            raise TypeError("files_path 不符合規範")
        if not encoding_type_support_check(encoding=encoding):
            raise ValueError(f"不支援 {encoding} 編碼格式")

        # 讀取資料
        data_dict = {}
        success_flag = False
        for file_path in files_path:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    data = json.load(file)
                    data_dict[os.path.basename(file_path)] = data
                print(f"{file_path} 已成功讀取")
                success_flag = True
            except (
                FileNotFoundError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as e:
                print(f"讀取文件 {file_path} 時發生錯誤 : {e}")
                data_dict[os.path.basename(file_path)] = None
        if not success_flag:
            raise FileNotFoundError("未能成功讀取任何檔案")
        return data_dict
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"read_dict 函數處理發生錯誤 : {e}")
        return {}


# 讀入儲存至 json 檔案或 txt 檔案的 dict
# path 為需要讀取之檔案所在的資料夾的絕對路徑
# file_extension 為需要讀取的資料副檔名
# function 為讀取方式，可設置為 1 - 3
# function = 1 : 選擇全部檔案
# function = 2 : 選擇部分檔案
# function = 3 : 選擇指定檔案
# file_name_list 只有當 function = 3 時才需設置，內容為資料夾內的 json 或 txt 檔案名稱(須包含副檔名)
# encoding 設置編碼格式
# return data_dict，函數處理錯誤時返回 {}
# data_dict 格式 : { str(檔案名稱) : dict(檔案內容) }，如果文件讀取失敗則對應值為 None
def load_dict(
    path, file_extension=".json", function=None, file_name_list=None, encoding="utf-8"
):
    try:
        # 檢查輸入資料類型
        param_type_check(
            param=file_extension, param_name="file_extension", expected_type=str
        )
        if file_extension not in [".json", ".txt"]:
            raise ValueError(f"不支援讀取 {file_extension} 檔")

        # 選擇檔案
        files_path_list = select_files(
            path=path,
            file_extension=file_extension,
            function=function,
            file_name_list=file_name_list,
        )
        if not files_path_list:
            raise ValueError("未選擇檔案")
        # 讀取檔案
        data_dict = read_dict(files_path=files_path_list, encoding=encoding)
        if data_dict == {}:
            raise ValueError("讀取檔案時發生錯誤")
        return data_dict
    except (ValueError, TypeError) as e:
        print(f"load_dict 函數處理發生錯誤 : {e}")
        return {}


def jpg():
    pass
