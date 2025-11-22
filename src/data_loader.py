import pandas as pd

def load_raw_data(path: str):
    """
    Load raw sensor data with acc_x, acc_y, acc_z, mic, label.
    Supports CSV or Excel.
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or XLSX.")
    
    return df
