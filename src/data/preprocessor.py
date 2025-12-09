from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.data.io import DataIO
from src.data.transformer import DataTransformer
from src.utils.logging import get_logger

# Logger riêng
LOGGER = get_logger("PREPROCESSOR")

class DataPreprocessor:
    """
    Class đóng gói các bước tiền xử lý dữ liệu (Orchestrator).
    
    Giữ trạng thái (state) của dữ liệu và gọi các hàm từ DataIO và DataTransformer.

    Attributes
    ----------
    data : DataFrame or None
        Dữ liệu được nạp vào và xử lý.
    numeric_cols : list
        Danh sách các cột có kiểu dữ liệu số.
    categorical_cols : list
        Danh sách các cột có kiểu dữ liệu phân loại.
    datetime_cols : list
        Danh sách các cột có kiểu dữ liệu ngày giờ.
    missing_strategy : str
        Phương pháp xử lý giá trị bị thiếu.
    scaling_strategy : str
        Phương pháp chuẩn hóa dữ liệu.
    outlier_method : str
        Phương pháp phát hiện ngoại lai.
    scaler : object or None
        Đối tượng scaler được fit với dữ liệu.
    encoders : dict
        Dictionary lưu trữ các encoder cho từng cột.
    """

    def __init__(
        self,
        num_strategy: str = 'median',
        cat_strategy: str = 'mode',
        dt_strategy: str = 'drop',
        scaling_strategy: str = 'standard',
        outlier_method: str = 'iqr'
    ) -> None:
        """
        Khởi tạo đối tượng DataPreprocessor.
        
        Parameters
        ----------
        num_strategy : str, optional
            Chiến lược xử lý giá trị thiếu cho cột số.
            Các giá trị hợp lệ: 'mean', 'median', 'mode', 'drop', 'ffill', 'bfill'.
            Mặc định là 'median'.
        cat_strategy : str, optional
            Chiến lược xử lý giá trị thiếu cho cột phân loại.
            Các giá trị hợp lệ: 'mode', 'constant', 'drop', 'ffill', 'bfill'.
            Mặc định là 'mode'.
        dt_strategy : str, optional
            Chiến lược xử lý giá trị thiếu cho cột datetime.
            Các giá trị hợp lệ: 'drop', 'ffill', 'bfill'.
            Mặc định là 'drop'.
        scaling_strategy : str, optional
            Phương pháp chuẩn hóa dữ liệu.
            Các giá trị hợp lệ: 'standard', 'robust'.
            Mặc định là 'standard'.
        outlier_method : str, optional
            Phương pháp phát hiện ngoại lai.
            Các giá trị hợp lệ: 'iqr', 'zscore', 'isolation_forest'.
            Mặc định là 'iqr'.
        """
        self.data: Optional[pd.DataFrame] = None
        
        # Config
        self.num_strategy: str = num_strategy
        self.cat_strategy: str = cat_strategy
        self.dt_strategy: str = dt_strategy
        self.scaling_strategy: str = scaling_strategy
        self.outlier_method: str = outlier_method

        # State
        self.scaler: Optional[Union[StandardScaler, RobustScaler]] = None
        self.encoders: dict[str, Any] = {}
        self.missing_num_values: dict[str, Any] = {}
        self.missing_cat_values: dict[str, Any] = {}
        self.scaled_cols_: list[str] = [] # Track scaled columns

    def __repr__(self) -> str:
        return (f"DataPreprocessor(num='{self.num_strategy}', cat='{self.cat_strategy}', dt='{self.dt_strategy}', "
                f"scaling='{self.scaling_strategy}', outlier='{self.outlier_method}')")

    def __str__(self) -> str:
        if self.data is None:
            return "DataPreprocessor (chưa nạp dữ liệu)"
        return f"DataPreprocessor: {self.data.shape[0]} dòng, {self.data.shape[1]} cột"

    @staticmethod
    def _log(message: str) -> None:
        LOGGER.info(message)

    def load_data(
        self,
        filepath: str,
        auto_convert_numeric: bool = True,
        auto_convert_threshold: float = 0.8
    ) -> DataPreprocessor:
        """
        Nạp dữ liệu, chuẩn hóa tên cột, tự động chuyển đổi các cột số nếu cần.
        
        Parameters
        ----------
        filepath : str
            Đường dẫn tới file dữ liệu. Hỗ trợ: .csv, .xlsx, .xls, .json.
        auto_convert_numeric : bool, optional
            Nếu True, tự động phát hiện và chuyển đổi các cột có thể là số.
            Mặc định là True.
        auto_convert_threshold : float, optional
            Ngưỡng tối thiểu để tự động chuyển đổi cột sang số (0.0 - 1.0).
            Mặc định là 0.8 (80%).
        
        Returns
        -------
        self
            Trả về instance để cho phép method chaining.
        """
        # 1. Load data
        self.data = DataIO.load_data(filepath)
        
        # 2. Clean column names
        self.data = DataIO.clean_column_names(self.data)
        
        # 3. Auto convert numeric columns
        if auto_convert_numeric:
            self.data = DataTransformer.auto_convert_numeric_columns(
                self.data, threshold=auto_convert_threshold
            )
            
        # 4. Auto detect types
        self.auto_detect_columns()
        return self

    def save_data(self, filepath: str) -> None:
        """
        Lưu dữ liệu đã xử lý vào file.
        
        Parameters
        ----------
        filepath : str
            Đường dẫn file CSV để lưu dữ liệu.
        """
        DataIO.save_data(self.data, filepath)

    def auto_detect_columns(self) -> DataPreprocessor:
        """
        Phân loại các cột theo kiểu dữ liệu.
        
        Cập nhật các thuộc tính numeric_cols, categorical_cols, datetime_cols
        của instance dựa trên kiểu dữ liệu của các cột trong DataFrame.
        
        Returns
        -------
        self
            Trả về instance để cho phép method chaining.
        
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp.
        """
        if self.data is None: raise ValueError("Data not loaded.")
        
        types = DataTransformer.auto_detect_columns(self.data)
        self.numeric_cols = types['numeric']
        self.categorical_cols = types['categorical']
        self.datetime_cols = types['datetime']
        
        self._log(f"Numeric cols: {self.numeric_cols}")
        self._log(f"Categorical cols: {self.categorical_cols}")
        self._log(f"Datetime cols: {self.datetime_cols}")
        return self

    def convert_to_datetime(
        self,
        columns: Optional[list[str]] = None,
        date_format: str = '%Y-%m-%d'
    ) -> DataPreprocessor:
        """
        Chuyển đổi các cột sang kiểu datetime.
        
        Parameters
        ----------
        columns : list or None, optional
            Danh sách các cột cần chuyển đổi. Nếu None, thử chuyển đổi
            tất cả các cột object/category. Mặc định là None.
        date_format : str, optional
            Định dạng ngày tháng. Mặc định là '%Y-%m-%d'.
        
        Returns
        -------
        self
            Trả về instance để cho phép method chaining.
        
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp.
        """
        if self.data is None: raise ValueError("Data not loaded.")
        self.data = DataTransformer.convert_to_datetime(self.data, columns, date_format)
        self.auto_detect_columns()
        return self

    def clean_negative_values(self) -> DataPreprocessor:
        """
        Làm sạch giá trị âm bằng cách thay thế bằng giá trị tuyệt đối.
        
        Returns
        -------
        self
            Trả về instance để cho phép method chaining.
        
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp.
        """
        if self.data is None: raise ValueError("Data not loaded.")
        self.data = DataTransformer.clean_negative_values(self.data)
        return self

    def handle_missing_values(
        self,
        data: Optional[pd.DataFrame] = None,
        exclude_features: Optional[list[str]] = None,
        fit: bool = False
    ) -> Union[DataPreprocessor, pd.DataFrame]:
        """
        Xử lý các giá trị thiếu trong dữ liệu.
        
        Parameters
        ----------
        data : DataFrame or None, optional
            DataFrame cần xử lý. Nếu None, xử lý dữ liệu nội bộ.
            Mặc định là None.
        exclude_features : list or None, optional
            Danh sách các cột không xử lý. Mặc định là None.
        fit : bool, optional
            Nếu True, học các giá trị thống kê từ data (cho train set).
            Nếu False, sử dụng giá trị đã học (cho test set).
            Mặc định là False.
        
        Returns
        -------
        self or DataFrame
            Nếu data=None, trả về self. Ngược lại trả về DataFrame đã xử lý.
        
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp và data=None.
        """
        # Determine target data
        if data is None:
            if self.data is None: raise ValueError("Data not loaded.")
            target = self.data
        else:
            target = data # Transformer copies it

        strategies = {
            'num': self.num_strategy,
            'cat': self.cat_strategy,
            'dt': self.dt_strategy
        }
        
        learned_values = {'num': self.missing_num_values, 'cat': self.missing_cat_values}
        
        processed_data, new_learned = DataTransformer.handle_missing_values(
            target, strategies, learned_values, exclude_features, fit
        )
        
        if fit:
            self.missing_num_values = new_learned['num']
            self.missing_cat_values = new_learned['cat']
            
        if data is None:
            self.data = processed_data
            self.auto_detect_columns()
            return self
        else:
            return processed_data

    def handle_outliers(
        self,
        data: Optional[pd.DataFrame] = None,
        exclude_features: Optional[list[str]] = None
    ) -> Union[DataPreprocessor, pd.DataFrame]:
        """
        Xử lý các giá trị ngoại lai.
        
        Loại bỏ các hàng chứa giá trị ngoại lai dựa trên phương pháp
        đã được cấu hình (outlier_method).
        
        Parameters
        ----------
        data : DataFrame or None, optional
            DataFrame cần xử lý. Nếu None, xử lý dữ liệu nội bộ.
            Mặc định là None.
        exclude_features : list or None, optional
            Danh sách các cột không xử lý ngoại lai.
            Mặc định là None.
        
        Returns
        -------
        self or DataFrame
            Nếu data=None, trả về self. Ngược lại trả về DataFrame đã xử lý.
        
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp và data=None.
        """
        if data is None:
            if self.data is None: raise ValueError("Data not loaded.")
            target = self.data
        else:
            target = data

        processed_data = DataTransformer.handle_outliers(
            target, self.outlier_method, exclude_features
        )
        
        if data is None:
            self.data = processed_data
            return self
        else:
            return processed_data

    def encode_categorical(
        self,
        data: Optional[pd.DataFrame] = None,
        strategy: str = 'onehot',
        fit: bool = False
    ) -> Union[DataPreprocessor, pd.DataFrame]:
        """
        Mã hóa các cột phân loại thành dạng số.
        
        Hỗ trợ fit/transform riêng biệt cho train/test set.
        - Label Encoding: Unknown values sẽ được gán giá trị -1.
        - One-Hot Encoding: Unknown values sẽ thành vector [0, 0, 0, ...].
        
        Parameters
        ----------
        data : DataFrame or None, optional
            DataFrame cần xử lý. Nếu None, xử lý dữ liệu nội bộ.
            Mặc định là None.
        strategy : str, optional
            Phương pháp mã hóa:
            - 'label': Label Encoding
            - 'onehot': One-Hot Encoding
            Mặc định là 'onehot'.
        fit : bool, optional
            Nếu True, fit encoders với data (cho train set).
            Nếu False, sử dụng encoders đã fit (cho test set).
            Mặc định là False.
        
        Returns
        -------
        self or DataFrame
            Nếu data=None, trả về self. Ngược lại trả về DataFrame đã xử lý.
        
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp và data=None.
        """
        if data is None:
            if self.data is None: raise ValueError("Data not loaded.")
            target = self.data
        else:
            target = data
        
        processed_data, new_encoders = DataTransformer.encode_categorical(
            target, strategy, self.encoders, fit
        )
        
        if fit:
            self.encoders = new_encoders
        
        if data is None:
            self.data = processed_data
            self.auto_detect_columns()
            return self
        else:
            return processed_data

    def scale_features(
        self,
        data: Optional[pd.DataFrame] = None,
        exclude_features: Optional[list[str]] = None,
        fit: bool = False
    ) -> Union[DataPreprocessor, pd.DataFrame]:
        """
        Chuẩn hóa (scaling) các cột số.
        
        Parameters
        ----------
        data : DataFrame or None, optional
            DataFrame cần chuẩn hóa. Nếu None, xử lý dữ liệu nội bộ.
            Mặc định là None.
        exclude_features : list or None, optional
            Danh sách các cột không scale. Mặc định là None.
        fit : bool, optional
            Nếu True, fit scaler với data (cho train set).
            Nếu False, chỉ transform sử dụng scaler đã fit (cho test set).
            Mặc định là False.
        
        Returns
        -------
        self or DataFrame
            Nếu data=None, trả về self. Ngược lại trả về DataFrame đã scale.
        
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp và data=None.
        """
        if data is None:
            if self.data is None: raise ValueError("Data not loaded.")
            target = self.data
        else:
            target = data

        processed_data, self.scaler, scaled_cols = DataTransformer.scale_features(
            target, self.scaling_strategy, self.scaler, exclude_features, fit
        )
        
        if fit:
            self.scaled_cols_ = scaled_cols

        if data is None:
            self.data = processed_data
            return self
        else:
            return processed_data

    def remove_duplicates(self, subset: Optional[list[str]] = None) -> DataPreprocessor:
        """
        Xóa các hàng trùng lặp trong DataFrame.
        
        Parameters
        ----------
        subset : list or None, optional
            Danh sách các cột để xác định trùng lặp.
            Nếu None, sử dụng tất cả các cột. Mặc định là None.
        
        Returns
        -------
        self
            Trả về instance để cho phép method chaining.
        
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp.
        """
        if self.data is None: raise ValueError("Data not loaded.")
        self.data = DataTransformer.remove_duplicates(self.data, subset)
        return self

    def drop_null_targets(self, target_column: str) -> DataPreprocessor:
        """
        Loại bỏ các hàng có giá trị target null.
        
        Parameters
        ----------
        target_column : str
            Tên cột target cần kiểm tra.
        
        Returns
        -------
        self
            Trả về instance để cho phép method chaining.
        
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp.
        """
        if self.data is None: raise ValueError("Data not loaded.")
        self.data = DataTransformer.drop_null_targets(self.data, target_column)
        return self

    def drop_features(self, features_to_drop: list[str]) -> DataPreprocessor:
        """
        Xóa các cột được chỉ định khỏi DataFrame.
        
        Parameters
        ----------
        features_to_drop : list
            Danh sách tên các cột cần xóa.
        
        Returns
        -------
        self
            Trả về instance để cho phép method chaining.
        
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp.
        """
        if self.data is None: raise ValueError("Data not loaded.")
        self.data = DataTransformer.drop_features(self.data, features_to_drop)
        self.auto_detect_columns()
        return self

    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Trả về dữ liệu đã được xử lý.
        
        Returns
        -------
        DataFrame or None
            DataFrame đã xử lý, hoặc None nếu chưa nạp dữ liệu.
        """
        return self.data
