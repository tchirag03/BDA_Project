import os   
import pandas as pd


xlsx_path = r"C:\Users\tchir\OneDrive\Desktop\BDA_project\dataset\technicals2"

try:
    for file in os.listdir(xlsx_path):
        if file.endswith(".xlsx"):
            xlsx_file = os.path.join(xlsx_path, file)
            df = pd.read_excel(xlsx_file , sheet_name="All Data")
            csv_file = os.path.join(xlsx_path, file.replace(".xlsx", ".csv"))
            print(f"Converting {file} to {csv_file}")
            #delete the xlsx file
            os.remove(xlsx_file)
            df.to_csv(csv_file, index=False)
            print(f"Converted {file} to {csv_file}")
except Exception as e:
    print(f"Error: {e}")

