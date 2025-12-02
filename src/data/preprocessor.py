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

    def __init__(self, num_strategy='median', cat_strategy='mode', dt_strategy='drop', scaling_strategy='standard', outlier_method='iqr'):
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
        self.data = None
        
        # Config
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.dt_strategy = dt_strategy
        self.scaling_strategy = scaling_strategy
        self.outlier_method = outlier_method

        # State
        self.scaler = None
        self.encoders = {}
        self.missing_num_values = {}
        self.missing_cat_values = {}
        self.scaled_cols_ = [] # Track scaled columns

    def __repr__(self):
        return (f"DataPreprocessor(num='{self.num_strategy}', cat='{self.cat_strategy}', dt='{self.dt_strategy}', "
                f"scaling='{self.scaling_strategy}', outlier='{self.outlier_method}')")

    def __str__(self):
        if self.data is None:
            return "DataPreprocessor (chưa nạp dữ liệu)"
        return f"DataPreprocessor: {self.data.shape[0]} dòng, {self.data.shape[1]} cột"

    @staticmethod
    def _log(message):
        LOGGER.info(message)

    def load_data(self, filepath, numeric_cols=None, auto_convert_numeric=False, auto_convert_threshold=0.8):
        """
        Nạp dữ liệu, chuẩn hóa tên cột, chuyển đổi các cột số nếu cần thiết.
        
        Parameters
        ----------
        filepath : str
            Đường dẫn tới file dữ liệu. Hỗ trợ: .csv, .xlsx, .xls, .json.
        numeric_cols : list or None, optional
            Danh sách các cột cần chuyển đổi sang kiểu số. Mặc định là None.
        auto_convert_numeric : bool, optional
            Nếu True, tự động phát hiện và chuyển đổi các cột có thể là số.
            Mặc định là False.
        auto_convert_threshold : float, optional
            Ngưỡng tối thiểu để tự động chuyển đổi cột sang số. Mặc định là 0.8.
        
        Returns
        -------
        self
            Trả về instance để cho phép method chaining.
        """
        # 1. Load data
        self.data = DataIO.load_data(filepath)
        
        # 2. Clean column names
        self.data = DataIO.clean_column_names(self.data)
        
        # 3. Convert numeric
        if numeric_cols:
            self.data = DataTransformer.convert_columns_to_numeric(self.data, numeric_cols)
        elif auto_convert_numeric:
            self.data = DataTransformer.auto_convert_numeric_columns(self.data, threshold=auto_convert_threshold)
            
        # 4. Auto detect types
        self.auto_detect_columns()
        return self

    def save_data(self, filepath):
        """
        Lưu dữ liệu đã xử lý vào file.
        
        Parameters
        ----------
        filepath : str
            Đường dẫn file CSV để lưu dữ liệu.
        """
        DataIO.save_data(self.data, filepath)

    def auto_detect_columns(self):
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

    def convert_to_datetime(self, columns=None, date_format='%Y-%m-%d'):
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

    def clean_negative_values(self):
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

    def handle_missing_values(self, data=None, exclude_features=None, fit=False):
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

    def handle_outliers(self, data=None, exclude_features=None):
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

    def encode_categorical(self, strategy='onehot'):
        """
        Mã hóa các cột phân loại thành dạng số.
        
        Parameters
        ----------
        strategy : str, optional
            Phương pháp mã hóa:
            - 'label': Label Encoding
            - 'onehot': One-Hot Encoding (drop_first=True)
            Mặc định là 'onehot'.
        
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
        
        self.data, self.encoders = DataTransformer.encode_categorical(self.data, strategy)
        self.auto_detect_columns()
        return self

    def scale_features(self, data=None, exclude_features=None, fit=False):
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

    def remove_duplicates(self, subset=None):
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

    def drop_features(self, features_to_drop):
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

    def get_processed_data(self):
        """
        Trả về dữ liệu đã được xử lý.
        
        Returns
        -------
        DataFrame or None
            DataFrame đã xử lý, hoặc None nếu chưa nạp dữ liệu.
        """
        return self.data
