import pandas as pd
import os
import re
from src.utils.logging import get_logger

# Logger riêng
LOGGER = get_logger("DATA_IO")

class DataIO:
    """
    Class chuyên trách việc Đọc/Ghi file và chuẩn hóa tên cột.
    """

    @staticmethod
    def _log(message):
        LOGGER.info(message)

    @staticmethod
    def load_data(filepath):
        """
        Nạp dữ liệu từ các file (CSV, XLSX, JSON) vào DataFrame.

        Parameters
        ----------
        filepath : str
            Đường dẫn tới file dữ liệu. Hỗ trợ các định dạng: .csv, .xlsx, .xls, .json

        Returns
        -------
        DataFrame
            Dữ liệu đã nạp.

        Raises
        ------
        ValueError
            Nếu định dạng file không được hỗ trợ.
        FileNotFoundError
            Nếu không tìm thấy file.
        """
        DataIO._log(f"Loading data from {filepath}...")

        try:
            if filepath.endswith('.csv'):
                data = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(filepath)
            elif filepath.endswith('.json'):
                data = pd.read_json(filepath)
            else:
                raise ValueError("Unsupported file format. Please use .csv, .xlsx, or .json.")
            
            return data
            
        except FileNotFoundError:
            DataIO._log(f"Error: File not found at {filepath}")
            raise
        except Exception as e:
            DataIO._log(f"Error loading data: {e}")
            raise

    @staticmethod
    def save_data(data, filepath):
        """
        Lưu dữ liệu vào file CSV.

        Parameters
        ----------
        data : DataFrame
            Dữ liệu cần lưu.
        filepath : str
            Đường dẫn file CSV để lưu dữ liệu.

        Raises
        ------
        ValueError
            Nếu data là None.
        """
        if data is None:
            raise ValueError("No data to save.")

        DataIO._log(f"Saving processed data to {filepath}...")
        try:
            # Tạo thư mục cha nếu chưa có
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data.to_csv(filepath, index=False)
            DataIO._log("Save successful.")
        except Exception as e:
            DataIO._log(f"Error saving data: {e}")
            raise

    @staticmethod
    def clean_column_names(data):
        """
        Chuẩn hóa tên cột của DataFrame.
        
        Thực hiện các bước làm sạch tên cột: loại bỏ ký tự xuống dòng, khoảng trắng thừa,
        chuyển về snake_case, loại bỏ ký tự đặc biệt và xử lý trùng lặp tên cột.

        Parameters
        ----------
        data : DataFrame
            DataFrame cần chuẩn hóa tên cột.

        Returns
        -------
        DataFrame
            DataFrame với tên cột đã được chuẩn hóa.
        """
        DataIO._log("Cleaning column names...")
        
        cols = (
            pd.Series(data.columns, dtype="string")
                .str.replace('\n', ' ', regex=False)       # bỏ newline
                .str.strip()                               # bỏ khoảng trắng đầu/đuôi
                .str.replace(r'\s+', ' ', regex=True)      # 1 khoảng trắng
                .str.lower()
                .str.replace(' ', '_', regex=False)        # snake_case
                .str.normalize('NFKD')
                .str.replace(r'[^\w]+', '_', regex=True)   # ký tự lạ -> _
                .str.replace(r'_+', '_', regex=True)       # gộp nhiều _
                .str.strip('_')                            # bỏ _ đầu/đuôi
        )

        # Chống trùng tên cột
        seen = {}
        def dedup(name):
            n = name if name != '' else 'col'
            seen[n] = seen.get(n, 0) + 1
            return n if seen[n] == 1 else f"{n}_{seen[n]-1}"

        data.columns = [dedup(c) for c in cols]
        DataIO._log("Column names cleaned successfully.")
        return data
