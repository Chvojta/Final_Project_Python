import pandas as pd

file_path = "Inputs.xlsx"  

import pandas as pd


with pd.ExcelFile(file_path, engine="openpyxl") as xls:
    df = pd.read_excel(xls, sheet_name=0)

    data_dict = df.set_index("Name").T.to_dict()
    for key, value in data_dict.items():
        print(f"  {key}: {value}")



with pd.ExcelFile(file_path, engine="openpyxl") as xls:
    df = pd.read_excel(xls, sheet_name=2)  
    data_dict = df.set_index("ID").T.to_dict()
    for key, value in data_dict.items():
        print(f"  {key}: {value}")


with pd.ExcelFile(file_path, engine="openpyxl") as xls:
    df = pd.read_excel(xls, sheet_name=3)  
    data_dict = df.set_index("ID").T.to_dict()
    for key, value in data_dict.items():
        print(f"  {key}: {value}")


with pd.ExcelFile(file_path, engine="openpyxl") as xls:
    df = pd.read_excel(xls, sheet_name=4)  
    data_dict = df.set_index("Customer ID").T.to_dict()
    for key, value in data_dict.items():
        print(f"  {key}: {value}")





