import os
import re
import glob
import pandas as pd


# 檢測輸入參數類型是否正確
# param 為須檢測的參數
# param_name 為參數名稱
# expected_type 為參數的類型，也支援多型別，請將參數的類型放入 tuple 內
def param_type_check(param, param_name, expected_type):
    if not isinstance(expected_type, (type, tuple)):
        raise TypeError(f"expected_type 必須是 type 或 (type, type, ...) 的 tuple")
    if not isinstance(param, expected_type):
        if isinstance(expected_type, tuple):
            type_names = ", ".join(t.__name__ for t in expected_type)
        else:
            type_names = expected_type.__name__
        raise TypeError(f"{param_name} 必須是 {type_names} 類型")


# 檢測 function 輸入是否合法
# function 格式為 int 或 None
# 使用此函數時，請將 function 預設為 None (手動處理模式)
# function_range 為功能的設定範圍
# function_range 格式 : { int(正整數 function) : str(功能) }
# max_error_count 設置最高可輸入錯誤之次數
# mission 為自定義之功能選擇說明
# return 合法 function，函數處理發生錯誤時返回 None，達到最高可輸入錯誤之次數時返回 0
def function_process(
    function=None, function_range={}, max_error_count=6, mission="請選擇處理方式"
):
    try:
        # 檢查輸入資料類型
        if function != None and not isinstance(function, int):
            raise TypeError("function 必須是 int 類型")
        param_type_check(
            param=function_range, param_name="function_range", expected_type=dict
        )
        if not function_range:
            raise ValueError("function_range 不能為空，請提供有效的功能範圍")
        if not all(
            isinstance(key, int) and key > 0 and isinstance(value, str)
            for key, value in function_range.items()
        ):
            raise ValueError("function_range 的鍵必須為正整數，值必須為字符串")
        if not isinstance(max_error_count, int) or max_error_count <= 0:
            raise TypeError("max_error_count 必須是為大於 0 的 int 類型")
        param_type_check(param=mission, param_name="mission", expected_type=str)

        # 處理 function
        # function 不合法
        if function is None or function not in function_range:
            # 錯誤記數
            error_count = 0
            # 修正 function
            while error_count < max_error_count:
                try:
                    # 提示詞
                    prompt = f"{mission}\n{''.join([f'{key}.{value}{chr(10)}' for key, value in function_range.items()])}請輸入選擇 :"
                    # 顯示功能選項並等待輸入
                    function = int(input(prompt))
                    if function in function_range:
                        return function
                    else:
                        print("請輸入有效的選項")
                        error_count += 1
                except ValueError:
                    print("請輸入一個有效的正整數選項")
                    error_count += 1
            # 處理達到最大錯誤次數
            print("已達最大錯誤次數，未能選擇有效選項")
            return 0
        # function 合法
        else:
            return function
    except (ValueError, TypeError) as e:
        print(f"function_process 函數處理發生錯誤 : {e}")
        return None


# 檢測路徑 path
# path 為需要檢測之路徑
# function 可設置為 1 - 3
# function = 1 : 確認路徑是否存在
# function = 2 : 確認路徑是否為資料夾
# function = 3 : 確認路徑是否為檔案
# function = 4 : 確認路徑是否為絕對路徑
# return 為 bool 值，函數處理錯誤時返回 False
def path_check(path, function=None):
    try:
        # 檢查輸入資料類型
        param_type_check(param=path, param_name="path", expected_type=str)
        if not path:
            raise ValueError("path 不得為空")

        function = function_process(
            function=function,
            function_range={
                1: "確認路徑是否存在",
                2: "確認路徑是否為資料夾",
                3: "確認路徑是否為檔案",
                4: "確認路徑是否為絕對路徑",
            },
        )
        # 確認路徑是否存在
        if function == 1:
            return os.path.exists(path)
        # 確認路徑是否為資料夾
        elif function == 2:
            return os.path.isdir(path)
        # 確認路徑是否為檔案
        elif function == 3:
            return os.path.isfile(path)
        # 確認路徑是否為絕對路徑
        elif function == 4:
            return os.path.isabs(path)
        elif function == 0:
            raise ValueError("輸入超過最大次數")
        elif function is None:
            raise ValueError("function_process 函數處理發生錯誤")
        else:
            raise ValueError("請檢查 function_process 函數之設定")
    except (ValueError, TypeError) as e:
        print(f"path_check 函數處理發生錯誤 : {e}")
        return False


# 定義支援之副檔名
SUPPORTED_EXTENSIONS = {".csv", ".txt", ".json", ".pkl", ".joblib"}


# 使用 SUPPORTED_EXTENSIONS 檢查 file_extension 是否為支援檔案類型
# file_extension 為需要檢查之副檔名
# return 為 bool 值，函數處理錯誤時返回 False
def file_extension_support_check(file_extension):
    try:
        # 檢查輸入資料類型
        param_type_check(
            param=file_extension, param_name="file_extension", expected_type=str
        )

        return file_extension in SUPPORTED_EXTENSIONS
    except TypeError as e:
        print(f"file_extension_support_check 函數處理發生錯誤 : {e}")
        return False


# 檢測檔名是否存在，如果存在便處理命名問題 (未修正 function = 1 及 2 時，所會遇到的命名問題，使用時請注意)
# 支援命名檔案類型請查看 file_extension_support_check 函數
# path 為資料夾的路徑
# name 為檔案名稱
# file_extension 為副檔名
# function 為檔名存在時之處理方式，可設置為 1 - 3
# function = 1 : 手動修改檔名
# function = 2 : 自動修改檔名
# function = 3 : 覆蓋原有檔案
# full_auto 設置 function = 2 時是否持續自動修改檔名
# return str(處理完成之檔案的絕對路徑)，函數處理錯誤時返回 None
def file_named(path, name, file_extension, function=None, full_auto=False):
    try:
        # 檢查輸入資料類型
        if not path_check(path=path, function=2):
            raise FileNotFoundError(f"指定的路徑無效 : {path}")
        param_type_check(param=name, param_name="name", expected_type=str)
        if not name:
            raise ValueError("name 不得為空")
        if not file_extension_support_check(file_extension):
            raise ValueError(f"不支援處理 {file_extension} 為副檔名之檔案")

        full_file_name = f"{name}{file_extension}"
        full_file_path = os.path.join(path, full_file_name)
        # 檢測檔名是否存在
        # 存在
        if path_check(path=full_file_path, function=1):
            print(f"{full_file_name} 已存在")
            function = function_process(
                function=function,
                function_range={
                    1: "手動修改檔名",
                    2: "自動修改檔名",
                    3: "覆蓋原有檔案",
                },
            )
            # 手動修改檔名
            if function == 1:
                # 使用 input 函數讀取輸入值 (後續可優化)
                name = input("請輸入新檔案名稱(不須包含副檔名) :")
                return file_named(path, name, file_extension, 1)
            # 自動修改檔名
            elif function == 2:
                # 檢查輸入資料類型
                param_type_check(
                    param=full_auto, param_name="full_auto", expected_type=bool
                )

                # 檢測 name 格式
                matched_number = re.search(r"^(.+?)_v(\d+)$", name)
                # 填充檔名
                if matched_number:
                    base_name = matched_number.group(1)
                    version_number = int(matched_number.group(2)) + 1
                    name = f"{base_name}_v{version_number}"
                else:
                    name = f"{name}_v1"
                return file_named(
                    path, name, file_extension, 2 if full_auto else 1, full_auto
                )
            # 覆蓋原有檔案
            elif function == 3:
                print("決定覆蓋原有檔案")
                return full_file_path
            elif function == 0:
                raise ValueError("輸入超過最大次數")
            elif function is None:
                raise ValueError("function_process 函數處理發生錯誤")
            else:
                raise ValueError("請檢查 function_process 函數之設定")
        # 不存在
        else:
            return full_file_path
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"file_named 函數處理發生錯誤 : {e}")
        return None


# 查找資料夾內特定檔案類型的檔案
# 支援查找檔案類型請查看 file_extension_support_check 函數
# path 為資料夾的路徑
# file_extension 為需要查找資料的副檔名
# display_info 控制是否顯示處理提示
# return [ 查找到之所有資料的絕對路徑 ]，函數處理錯誤時返回 []
def file_extension_search(path, file_extension, display_info=True):
    try:
        # 檢查輸入資料類型
        if not path_check(path=path, function=2):
            raise FileNotFoundError(f"指定的路徑無效 : {path}")
        if not file_extension_support_check(file_extension):
            raise ValueError(f"不支援處理 {file_extension} 為副檔名之檔案")
        param_type_check(
            param=display_info, param_name="display_info", expected_type=bool
        )

        # 尋找檔案
        if display_info:
            print(f"正在尋找 {file_extension} 檔案")
        search_files_path_list = glob.glob(os.path.join(path, f"*{file_extension}"))
        if search_files_path_list:
            if display_info:
                print(f"尋找到 {len(search_files_path_list)} 個 {file_extension} 檔案")
            return search_files_path_list
        else:
            if display_info:
                print(f"未尋找到 {file_extension} 檔案")
            return []
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"file_extension_search 函數處理發生錯誤 : {e}")
        return []


# 檢查 compare_list 中的值是否存在於 benchmarks_list 中，並進行分類
# 使用時請在調用函數中添加對 benchmarks_list 及 compare_list 中值的檢查
# benchmarks_list 為基準的 list
# compare_list 為需檢查的 list
# return exist_compare_list(存在的值) 及 not_exist_compare_list(不存在的值)，函數處理錯誤時返回 [], []
def list_compare(benchmarks_list, compare_list):
    try:
        # 檢查輸入資料類型
        param_type_check(
            param=benchmarks_list,
            param_name="benchmarks_list",
            expected_type=(list, set, tuple),
        )
        if not benchmarks_list:
            raise ValueError("benchmarks_list 不得為空")
        param_type_check(
            param=compare_list,
            param_name="compare_list",
            expected_type=(list, set, tuple),
        )
        if not compare_list:
            raise ValueError("compare_list 不得為空")

        # 開始分類
        benchmarks_list = set(benchmarks_list)
        exist_compare_list = []
        not_exist_compare_list = []
        for value in compare_list:
            if value in benchmarks_list:
                exist_compare_list.append(value)
            else:
                not_exist_compare_list.append(value)
        return exist_compare_list, not_exist_compare_list
    except (ValueError, TypeError) as e:
        print(f"list_compare 函數處理發生錯誤 : {e}")
        return [], []


# 檢查 dataframe 是否符合要求，類別正確且不為空
# dataframe 為須檢查之 pd.dataframe
# return 為 bool 值，函數處理錯誤時返回 False
def dataframe_check(dataframe):
    try:
        # 檢查輸入資料類型
        param_type_check(
            param=dataframe, param_name="dataframe", expected_type=pd.DataFrame
        )
        if dataframe.empty:
            raise ValueError("dataframe 不得為空")

        return True
    except (ValueError, TypeError) as e:
        print(f"dataframe_check 函數處理發生錯誤 : {e}")
        return False


# 檢查 DataFrame 中是否包含 target_column_list 中指定的欄位，並移除 target_column_list 中不存在的欄位
# dataframe 為須檢查之 pd.dataframe
# target_column_list 為包含需檢查之欄位名所組成的 list
# display_info 控制是否顯示處理提示
# return 處理完成之 target_column_list，函數處理錯誤時返回 []
def dataframe_column_check(dataframe, target_column_list, display_info=True):
    try:
        # 檢查輸入資料類型
        if not dataframe_check(dataframe=dataframe):
            raise ValueError("dataframe 不符合要求")
        param_type_check(
            param=target_column_list,
            param_name="target_column_list",
            expected_type=list,
        )
        param_type_check(
            param=display_info, param_name="display_info", expected_type=bool
        )
        if not target_column_list:
            if display_info:
                print("target_column_list 為空無需處理")
            return []
        if not all(isinstance(column_name, str) for column_name in target_column_list):
            raise ValueError("target_column_list 中的欄位名稱資料必須是 str 類型")

        if display_info:
            print("提取有效欄位中")
        # 檢測有效欄位
        existing_columns = set(dataframe.columns)
        exist_target_column_list, not_exist_target_column_list = list_compare(
            benchmarks_list=existing_columns, compare_list=target_column_list
        )
        if not exist_target_column_list and not not_exist_target_column_list:
            raise ValueError("檢測欄位時，發生錯誤")
        # 顯示處理結果
        if display_info:
            if exist_target_column_list:
                print(
                    f"target_column_list 中所包含有效的 column 名稱為 : {', '.join(exist_target_column_list)}"
                )
            if not_exist_target_column_list:
                print(
                    f"target_column_list 中所包含無效的 column 名稱為 : {', '.join(not_exist_target_column_list)}"
                )
        return exist_target_column_list
    except (ValueError, TypeError) as e:
        print(f"dataframe_column_check 函數處理發生錯誤 : {e}")
        return []


# 對指定資料夾下的特定副檔名之檔案進行選擇 (可考慮後續優化)
# path 為資料夾的路徑
# file_extension 為需要尋找的資料副檔名
# function 為讀取方式，可設置為 1 - 3
# function = 1 : 選擇全部檔案
# function = 2 : 選擇部分檔案
# function = 3 : 選擇指定檔案
# file_name_list 只有當 function = 3 時才需設置，內容為資料夾內的 file_extension 檔案名稱(須包含副檔名)
# return [ 查找到之資料的絕對路徑 ]，函數處理錯誤時返回 []
def select_files(path, file_extension, function=None, file_name_list=None):
    try:
        all_files_path = file_extension_search(path=path, file_extension=file_extension)
        # 檢查輸入資料類型
        if not all_files_path:
            raise FileNotFoundError(f"未找到 {file_extension} 檔案")

        function = function_process(
            function=function,
            function_range={1: "選擇全部檔案", 2: "選擇部分檔案", 3: "選擇指定檔案"},
        )
        # 選擇全部檔案
        if function == 1:
            return all_files_path
        # 選擇部分檔案 (可優化)
        elif function == 2:
            file_names = [os.path.basename(file_path) for file_path in all_files_path]
            file_select_indices = set()
            # 顯示檔案名稱及代號
            print(f"{'代號':<5} {'檔案名稱'}")
            for i, file_name in enumerate(file_names):
                print(f"{i:<5} {file_name}")
            # 選擇檔案
            while True:
                select_file = input(
                    f"請選擇讀入 {file_extension} 檔案代號 (讀入多個檔案用空格隔開，退出請單獨輸入q) :"
                ).split()
                if "q" in select_file and len(select_file) == 1:
                    break
                for i in select_file:
                    if i.isdigit() and int(i) in range(len(file_names)):
                        file_select_indices.add(int(i))
                    else:
                        print(f"{i} 非檔案代號")
            # 整理代號並確保代號存在
            file_select_indices = sorted(file_select_indices)
            if not file_select_indices:
                raise FileNotFoundError(f"未選擇任何 {file_extension} 檔案")
            # 提取所選擇的檔案之路徑並回傳
            return [all_files_path[i] for i in file_select_indices]
        # 選擇指定檔案
        elif function == 3:
            # 檢查輸入資料類型
            param_type_check(
                param=file_name_list, param_name="file_name_list", expected_type=list
            )
            if not file_name_list:
                raise ValueError("file_name_list 不得為空")
            if not all(isinstance(name, str) for name in file_name_list):
                raise ValueError("file_name_list 中的檔案名稱資料必須是 str 類型")

            file_names = [os.path.basename(file_path) for file_path in all_files_path]
            # 檢查 file_name_list
            exist_file_name_list, no_exist_file_name_list = list_compare(
                benchmarks_list=file_names, compare_list=file_name_list
            )
            if not exist_file_name_list and not no_exist_file_name_list:
                raise ValueError("檢測欄位時，發生錯誤")
            if not exist_file_name_list:
                raise FileNotFoundError(f"未選擇任何 {file_extension} 檔案")
            return [os.path.join(path, name) for name in exist_file_name_list]
        elif function == 0:
            raise ValueError("輸入超過最大次數")
        elif function is None:
            raise ValueError("function_process 函數處理發生錯誤")
        else:
            raise ValueError("請檢查 function_process 函數之設定")
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"select_files 函數處理發生錯誤 : {e}")
        return []
