import json
import pandas as pd
from .load_data import encoding_type_support_check
from .General_functions import file_named, param_type_check, dataframe_check, path_check


# 檢查 save_csv 和 save_dict 函數中 file_path 參數是否符合要求 (可優化路徑檢查)
# file_path 為 save_csv 和 save_dict 函數中的 file_path 參數
# return 為 bool 值，函數處理錯誤時返回 False
def file_path_check(file_path):
    try:
        # 檢查輸入資料類型
        param_type_check(param=file_path, param_name="file_path", expected_type=str)
        if not file_path.strip():
            raise ValueError("file_path 不得為空或僅包含空白字元")

        return True
    except (TypeError, ValueError) as e:
        print(f"file_path_check 函數處理發生錯誤 : {e}")
        return False


# 設定 result_dict 格式，並檢查輸入有無錯誤
# file_saved 設置檔案是否儲存
# path 儲存完成之檔案路徑
# error_message 錯誤資訊
# return result_dict
# result_dict 格式 : {"file_saved": file_saved,"path": path,"error_message": error_message}
def result_dict_set_up(file_saved, path, error_message):
    try:
        # 檢查輸入資料類型
        param_type_check(param=file_saved, param_name="file_saved", expected_type=bool)
        param_type_check(param=path, param_name="path", expected_type=(str, type(None)))
        param_type_check(
            param=error_message,
            param_name="error_message",
            expected_type=(str, type(None)),
        )
        if file_saved:
            if not path_check(path=path, function=1):
                raise ValueError("path 在儲存成功時必須存在")
            if error_message is not None:
                raise ValueError("error_message 在儲存成功時必須為 None")
        else:
            if path is not None:
                raise ValueError("path 在儲存失敗時必須為 None")
            if error_message is None:
                raise ValueError("error_message 在儲存失敗時必須存在")

        result_dict = {
            "file_saved": file_saved,
            "path": path,
            "error_message": error_message,
        }
        return result_dict
    except (TypeError, ValueError) as e:
        print(f"result_dict 函數處理發生錯誤 : {e}")
        result_dict = {
            "file_saved": False,
            "path": None,
            "error_message": f"result_dict 創建時出現錯誤 : {str(e)}",
        }
        return result_dict


# saving_csv 函數之衍生函數，處理檔案之儲存
# file_path 為檔案的絕對路徑
# dataframe 為需要儲存的 pd.DataFrame
# encoding 設置編碼格式
# return result_dict (請查看 result_dict_set_up)
def save_csv(file_path, dataframe, encoding="utf-8"):
    try:
        # 檢查輸入資料類型
        if not file_path_check(file_path=file_path):
            raise ValueError(f"file_path 不符合規範")
        if not dataframe_check(dataframe=dataframe):
            raise ValueError("dataframe 不符合要求")
        if not encoding_type_support_check(encoding=encoding):
            raise ValueError(f"不支援 {encoding} 編碼格式")

        # 儲存檔案
        dataframe.to_csv(file_path, index=False, encoding=encoding, float_format="%.8f")
        return result_dict_set_up(file_saved=True, path=file_path, error_message=None)
    except (FileNotFoundError, OSError, TypeError, ValueError) as e:
        print(f"save_csv 函數處理發生錯誤 : {e}")
        return result_dict_set_up(file_saved=False, path=None, error_message=str(e))


# 儲存 dataframe 至 csv 檔
# path 儲存資料夾位置
# dataframe_dict 為需要儲存之 dataframe 所組成的 dict
# dataframe_dict 格式 : { str(dataframe 名稱) : pd.DataFrame }，dataframe名稱不須包含副檔名
# function 為檔名存在時之處理方式，可設置為 1 - 3
# function = 1 : 手動修改檔名
# function = 2 : 自動修改檔名
# function = 3 : 覆蓋原有檔案
# full_auto 設置 function = 2 時是否持續自動修改檔名
# encoding 設置編碼格式
# return result_dict，函數處理錯誤時返回 {}
# result_dict 格式 : { dataframe 名稱 : result_dict(save_csv 函數的返回值) }
def saving_csv(path, dataframe_dict, function=None, full_auto=False, encoding="utf-8"):
    try:
        # 檢查輸入資料類型
        param_type_check(
            param=dataframe_dict, param_name="dataframe_dict", expected_type=dict
        )
        if not dataframe_dict:
            raise ValueError("dataframe_dict 不得為空")
        if not all(
            isinstance(name, str) and isinstance(dataframe, pd.DataFrame)
            for name, dataframe in dataframe_dict.items()
        ):
            raise ValueError("dataframe_dict 的鍵必須為 str，值必須為 pd.DataFrame")

        # 儲存檔案
        result_dict = {}
        for name, dataframe in dataframe_dict.items():
            file_path = file_named(
                path=path,
                name=name,
                file_extension=".csv",
                function=function,
                full_auto=full_auto,
            )
            result_dict[name] = save_csv(
                file_path=file_path, dataframe=dataframe, encoding=encoding
            )
        return result_dict
    except (TypeError, ValueError) as e:
        print(f"saving_csv 函數處理發生錯誤 : {e}")
        return {}


# saving_dict 函數之衍生函數，處理檔案之儲存
# file_path 為檔案的絕對路徑
# data_dict 為需要儲存的 dict
# encoding 設置編碼格式
# ensure_ascii 為 json.dump 函數之參數
# indent 為 json.dump 函數之參數，功能為指定縮排
# return result_dict (請查看 result_dict_set_up)
def save_dict(file_path, data_dict, encoding="utf-8", ensure_ascii=False, indent=4):
    try:
        # 檢查輸入資料類型
        if not file_path_check(file_path=file_path):
            raise ValueError(f"file_path 不符合規範")
        param_type_check(param=data_dict, param_name="data_dict", expected_type=dict)
        if not data_dict:
            raise ValueError("data_dict 不得為空")
        if not encoding_type_support_check(encoding=encoding):
            raise ValueError(f"不支援 {encoding} 編碼格式")
        param_type_check(
            param=ensure_ascii, param_name="ensure_ascii", expected_type=bool
        )
        param_type_check(param=indent, param_name="indent", expected_type=int)

        # 儲存檔案
        with open(file_path, "w", encoding=encoding) as file:
            json.dump(data_dict, file, ensure_ascii=ensure_ascii, indent=indent)
        return result_dict_set_up(file_saved=True, path=file_path, error_message=None)
    except (FileNotFoundError, OSError, TypeError, ValueError) as e:
        print(f"save_dict 函數處理發生錯誤 : {e}")
        return result_dict_set_up(file_saved=False, path=None, error_message=str(e))


# 儲存 dict 至 txt 或 json 檔內
# path 儲存資料夾位置
# data_dict 為需要儲存之 dict 所組成的 dict
# data_dict 格式 : { str(dict名稱) : dict }，dict名稱不須包含副檔名
# file_extension 定義儲存 dict 文件之副檔名
# function 為檔名存在時之處理方式，可設置為 1 - 3 (為 file_named 函數的功能)
# function = 1 : 手動修改檔名
# function = 2 : 自動修改檔名
# function = 3 : 覆蓋原有檔案
# full_auto 設置 function = 2 時是否持續自動修改檔名(為 file_named 函數的功能)
# encoding 設置編碼格式
# ensure_ascii 為 json.dump 函數之參數
# indent 為 json.dump 函數之參數
# return result_dict，函數處理錯誤時返回 : {}
# result_dict 格式 : { dict 名稱 : result_dict(save_dict 函數的返回值) }
def saving_dict(
    path,
    data_dict,
    file_extension=".json",
    function=None,
    full_auto=False,
    encoding="utf-8",
    ensure_ascii=False,
    indent=4,
):
    try:
        # 檢查輸入資料類型
        param_type_check(param=data_dict, param_name="data_dict", expected_type=dict)
        if not data_dict:
            raise ValueError("data_dict 不得為空")
        if not all(
            isinstance(name, str) and isinstance(data, dict)
            for name, data in data_dict.items()
        ):
            raise TypeError("data_dict 的鍵必須為 str，值必須為 dict")
        file_extension_support = [".json", ".txt"]
        if file_extension not in file_extension_support:
            raise ValueError(
                f"file_extension 只支援 : {', '.join(file_extension_support)}"
            )

        # 儲存檔案
        result_dict = {}
        for name, data in data_dict.items():
            file_path = file_named(
                path=path,
                name=name,
                file_extension=file_extension,
                function=function,
                full_auto=full_auto,
            )
            result_dict[name] = save_dict(
                file_path=file_path,
                data_dict=data,
                encoding=encoding,
                ensure_ascii=ensure_ascii,
                indent=indent,
            )
        return result_dict
    except (TypeError, ValueError) as e:
        print(f"saving_dict 函數處理發生錯誤 : {e}")
        return {}
