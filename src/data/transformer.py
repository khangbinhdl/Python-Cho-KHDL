from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler

from src.utils.logging import get_logger

# Logger riêng
LOGGER = get_logger("DATA_TRANSFORMER")

class DataTransformer:
    """
    Class chuyên chứa các thuật toán xử lý dữ liệu.
    (Missing, Outlier, Scaling, Encoding, Type Conversion)
    """

    @staticmethod
    def _log(message: str) -> None:
        LOGGER.info(message)

    @staticmethod
    def auto_detect_columns(data: pd.DataFrame) -> dict[str, list[str]]:
        """
        Tự động phát hiện và phân loại các cột theo kiểu dữ liệu.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame cần phân loại các cột.
        
        Returns
        -------
        dict
            Dictionary chứa danh sách các cột với các key:
            - 'numeric': Danh sách các cột số
            - 'categorical': Danh sách các cột phân loại
            - 'datetime': Danh sách các cột ngày giờ
        """
        numeric_cols: list[str] = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols: list[str] = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols: list[str] = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }

    @staticmethod
    def auto_convert_numeric_columns(data: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        """
        Tự động phát hiện và chuyển đổi các cột object có thể là số.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame chứa các cột cần kiểm tra.
        threshold : float, optional
            Tỷ lệ tối thiểu giá trị có thể chuyển đổi được để quyết định
            chuyển đổi cột. Mặc định là 0.8 (80%).
        
        Returns
        -------
        DataFrame
            DataFrame với các cột đã được tự động chuyển đổi sang kiểu số.
        """
        DataTransformer._log(f"Auto-detecting numeric columns (threshold={threshold:.0%})...")
        object_cols = data.select_dtypes(include=['object']).columns.tolist()
        converted = []
        
        for col in object_cols:
            cleaned = (
                data[col].astype(str)
                .str.replace(',', '', regex=False)
                .str.replace(' ', '', regex=False)
                .str.strip()
            )
            numeric_values = pd.to_numeric(cleaned, errors='coerce')
            
            original_non_null = data[col].notna().sum()
            if original_non_null == 0:
                continue
                
            valid_ratio = numeric_values.notna().sum() / original_non_null
            
            if valid_ratio >= threshold:
                data[col] = numeric_values
                converted.append(col)
        
        if converted:
            DataTransformer._log(f"Auto-converted {len(converted)} columns to numeric: {converted}")
        return data

    @staticmethod
    def convert_to_datetime(
        data: pd.DataFrame,
        columns: Optional[list[str]] = None,
        date_format: str = '%Y-%m-%d'
    ) -> pd.DataFrame:
        """
        Chuyển đổi các cột sang kiểu datetime.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame chứa các cột cần chuyển đổi.
        columns : list or None, optional
            Danh sách các cột cần chuyển đổi. Nếu None, sẽ thử chuyển đổi
            tất cả các cột object/category. Mặc định là None.
        date_format : str, optional
            Định dạng ngày tháng. Mặc định là '%Y-%m-%d'.
        
        Returns
        -------
        DataFrame
            DataFrame với các cột đã được chuyển đổi sang datetime.
        """
        DataTransformer._log("Converting columns to datetime...")
        
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        converted_cols = []
        for col in columns:
            if col not in data.columns:
                continue
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                continue

            try:
                converted_col = pd.to_datetime(data[col], format=date_format, errors='coerce')
                if converted_col.notna().any():
                    data[col] = converted_col
                    converted_cols.append(col)
            except (ValueError, TypeError):
                pass

        if converted_cols:
            DataTransformer._log(f"Converted {len(converted_cols)} columns to datetime: {converted_cols}")
        return data

    @staticmethod
    def clean_negative_values(data: pd.DataFrame) -> pd.DataFrame:
        """
        Thay thế giá trị âm bằng giá trị tuyệt đối trong các cột số.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame chứa các cột cần xử lý.
        
        Returns
        -------
        DataFrame
            DataFrame với tất cả giá trị âm đã được thay thế bằng giá trị tuyệt đối.
        """
        DataTransformer._log("Cleaning negative values in all numeric columns...")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] = np.abs(data[col])
        return data

    @staticmethod
    def _handle_missing_numeric(
        data: pd.DataFrame,
        columns: list[str],
        strategy: str,
        learned_values: dict[str, Any],
        exclude_features: list[str],
        fit: bool
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Xử lý giá trị thiếu cho các cột số.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame chứa dữ liệu cần xử lý.
        columns : list
            Danh sách các cột số cần xử lý.
        strategy : str
            Chiến lược xử lý ('mean', 'median', 'mode', 'drop', 'ffill', 'bfill').
        learned_values : dict
            Dictionary chứa các giá trị đã học (tên cột -> giá trị điền).
        exclude_features : list
            Danh sách các cột không xử lý.
        fit : bool
            Nếu True, học các giá trị thống kê từ data.
        
        Returns
        -------
        tuple
            (DataFrame, dict) - DataFrame đã xử lý và dictionary learned_values cập nhật.
        """
        if not columns:
            return data, learned_values
        
        cols_to_process = [c for c in columns if c not in exclude_features]
        new_learned = learned_values.copy()
        
        # FIT: Học tham số
        if fit and strategy in ("mean", "median", "mode"):
            for col in cols_to_process:
                if data[col].isna().any():
                    if strategy == "mean":
                        val = data[col].mean()
                    elif strategy == "median":
                        val = data[col].median()
                    else:  # mode
                        mode = data[col].mode()
                        val = mode.iloc[0] if not mode.empty else data[col].median()
                    new_learned[col] = val
        
        # TRANSFORM: Áp dụng
        if strategy == "drop":
            if cols_to_process:
                data = data.dropna(subset=cols_to_process)
        elif strategy in ("mean", "median", "mode"):
            for col in cols_to_process:
                val = new_learned.get(col)
                if val is not None:
                    data[col] = data[col].fillna(val)
        elif strategy == "ffill":
            if cols_to_process:
                data[cols_to_process] = data[cols_to_process].ffill()
        elif strategy == "bfill":
            if cols_to_process:
                data[cols_to_process] = data[cols_to_process].bfill()
        
        return data, new_learned

    @staticmethod
    def _handle_missing_categorical(
        data: pd.DataFrame,
        columns: list[str],
        strategy: str,
        learned_values: dict[str, Any],
        exclude_features: list[str],
        fit: bool
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Xử lý giá trị thiếu cho các cột phân loại.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame chứa dữ liệu cần xử lý.
        columns : list
            Danh sách các cột phân loại cần xử lý.
        strategy : str
            Chiến lược xử lý ('mode', 'constant', 'drop', 'ffill', 'bfill').
        learned_values : dict
            Dictionary chứa các giá trị đã học (tên cột -> giá trị điền).
        exclude_features : list
            Danh sách các cột không xử lý.
        fit : bool
            Nếu True, học các giá trị thống kê từ data.
        
        Returns
        -------
        tuple
            (DataFrame, dict) - DataFrame đã xử lý và dictionary learned_values cập nhật.
        """
        if not columns:
            return data, learned_values
        
        cols_to_process = [c for c in columns if c not in exclude_features]
        new_learned = learned_values.copy()
        
        # FIT: Học tham số
        if fit and strategy in ("mode", "constant"):
            for col in cols_to_process:
                if data[col].isna().any():
                    if strategy == "mode":
                        mode = data[col].mode()
                        val = mode.iloc[0] if not mode.empty else "Unknown"
                    else:  # constant
                        val = "Unknown"
                    new_learned[col] = val
        
        # TRANSFORM: Áp dụng
        if strategy == "drop":
            if cols_to_process:
                data = data.dropna(subset=cols_to_process)
        elif strategy == "mode":
            for col in cols_to_process:
                val = new_learned.get(col)
                if val is not None:
                    data[col] = data[col].fillna(val)
                else:
                    # Fallback nếu không tìm thấy trong learned (e.g. cột mới)
                    mode = data[col].mode()
                    val = mode.iloc[0] if not mode.empty else "Unknown"
                    data[col] = data[col].fillna(val)
        elif strategy == "constant":
            for col in cols_to_process:
                val = new_learned.get(col, "Unknown")
                data[col] = data[col].fillna(val)
        elif strategy == "ffill":
            if cols_to_process:
                data[cols_to_process] = data[cols_to_process].ffill()
        elif strategy == "bfill":
            if cols_to_process:
                data[cols_to_process] = data[cols_to_process].bfill()
        
        return data, new_learned

    @staticmethod
    def _handle_missing_datetime(
        data: pd.DataFrame,
        columns: list[str],
        strategy: str
    ) -> pd.DataFrame:
        """
        Xử lý giá trị thiếu cho các cột datetime.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame chứa dữ liệu cần xử lý.
        columns : list
            Danh sách các cột datetime cần xử lý.
        strategy : str
            Chiến lược xử lý ('drop', 'ffill', 'bfill').
        
        Returns
        -------
        DataFrame
            DataFrame đã xử lý.
        """
        if not columns:
            return data
        
        if strategy == "drop":
            data = data.dropna(subset=columns)
        elif strategy == "ffill":
            data[columns] = data[columns].ffill()
        elif strategy == "bfill":
            data[columns] = data[columns].bfill()
        
        return data

    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        strategies: dict[str, str],
        learned_values: Optional[dict[str, dict[str, Any]]] = None,
        exclude_features: Optional[list[str]] = None,
        fit: bool = False
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        """
        Xử lý giá trị thiếu theo các chiến lược được chỉ định.
        
        Hàm này tổng hợp việc xử lý missing values cho 3 loại dữ liệu:
        numeric, categorical và datetime.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame chứa dữ liệu cần xử lý.
        strategies : dict
            Dictionary chứa chiến lược xử lý cho từng loại cột:
            - 'num': Chiến lược cho cột số ('mean', 'median', 'mode', 'drop', 'ffill', 'bfill')
            - 'cat': Chiến lược cho cột phân loại ('mode', 'constant', 'drop', 'ffill', 'bfill')
            - 'dt': Chiến lược cho cột datetime ('drop', 'ffill', 'bfill')
        learned_values : dict or None, optional
            Dictionary chứa các giá trị đã học từ tập training:
            - 'num': dict mapping tên cột -> giá trị điền
            - 'cat': dict mapping tên cột -> giá trị điền
            Mặc định là None.
        exclude_features : list or None, optional
            Danh sách các cột không xử lý. Mặc định là None.
        fit : bool, optional
            Nếu True, học các giá trị thống kê từ data.
            Nếu False, sử dụng learned_values đã có. Mặc định là False.
        
        Returns
        -------
        tuple
            (DataFrame, dict) - DataFrame đã xử lý và dictionary learned_values.
        """
        target = data.copy()
        exclude_features = exclude_features or []
        
        # Lấy chiến lược cho từng loại
        num_strategy = strategies.get('num', 'median')
        cat_strategy = strategies.get('cat', 'mode')
        dt_strategy = strategies.get('dt', 'drop')
        
        # Phát hiện loại cột
        col_types = DataTransformer.auto_detect_columns(target)
        numeric_cols = col_types['numeric']
        categorical_cols = col_types['categorical']
        datetime_cols = col_types['datetime']
        
        # Khởi tạo learned_values
        new_learned_values = {'num': {}, 'cat': {}}
        if learned_values:
            new_learned_values = {
                'num': learned_values.get('num', {}).copy(),
                'cat': learned_values.get('cat', {}).copy()
            }

        DataTransformer._log(f"Handling missing values fit={fit} | strategies={strategies}")
        initial_rows = len(target)

        # 1. Xử lý datetime
        target = DataTransformer._handle_missing_datetime(
            target, datetime_cols, dt_strategy
        )

        # 2. Xử lý numeric
        target, new_learned_values['num'] = DataTransformer._handle_missing_numeric(
            target, numeric_cols, num_strategy,
            new_learned_values['num'], exclude_features, fit
        )

        # 3. Xử lý categorical
        target, new_learned_values['cat'] = DataTransformer._handle_missing_categorical(
            target, categorical_cols, cat_strategy,
            new_learned_values['cat'], exclude_features, fit
        )

        # Log số hàng đã xóa
        rows_removed = initial_rows - len(target)
        if rows_removed > 0:
            DataTransformer._log(f"Dropped {rows_removed} rows due to missing values")
            
        return target.reset_index(drop=True), new_learned_values

    @staticmethod
    def handle_outliers(
        data: pd.DataFrame,
        method: str = 'iqr',
        exclude_features: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Xử lý ngoại lai trong các cột số.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame chứa dữ liệu cần xử lý.
        method : str, optional
            Phương pháp phát hiện ngoại lai:
            - 'iqr': Sử dụng quy tắc IQR (loại bỏ ngoài 1.5*IQR)
            - 'zscore': Sử dụng Z-score (loại bỏ |z| > 3)
            - 'isolation_forest': Sử dụng Isolation Forest
            Mặc định là 'iqr'.
        exclude_features : list or None, optional
            Danh sách các cột không xử lý ngoại lai. Mặc định là None.
        
        Returns
        -------
        DataFrame
            DataFrame đã loại bỏ các hàng chứa ngoại lai.
        """
        target = data.copy()
        if exclude_features is None: exclude_features = []
        
        DataTransformer._log(f"[Outlier] method='{method}' - Removing outliers")
        initial_rows = len(target)
        
        numeric_cols = target.select_dtypes(include=np.number).columns.tolist()
        cols_to_process = [c for c in numeric_cols if c not in exclude_features]

        if method in ('iqr', 'zscore'):
            mask = pd.Series(True, index=target.index)
            for col in cols_to_process:
                if method == 'iqr':
                    Q1 = target[col].quantile(0.25)
                    Q3 = target[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                else: # zscore
                    mean = target[col].mean()
                    std = target[col].std()
                    lower = mean - 3 * std
                    upper = mean + 3 * std
                mask &= (target[col] >= lower) & (target[col] <= upper)
            target = target[mask]
            
        elif method == 'isolation_forest':
            if cols_to_process:
                iso = IsolationForest(random_state=42)
                yhat = iso.fit_predict(target[cols_to_process])
                target = target[yhat == 1]

        rows_removed = initial_rows - len(target)
        DataTransformer._log(f"Removed {rows_removed} rows as outliers")
        return target.reset_index(drop=True)

    @staticmethod
    def encode_categorical(
        data: pd.DataFrame,
        strategy: str = 'onehot',
        encoders: Optional[dict[str, Any]] = None,
        fit: bool = False
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Mã hóa các cột phân loại thành dạng số.
        
        Hỗ trợ fit/transform riêng biệt cho train/test set.
        - Label Encoding: Unknown values sẽ được gán giá trị -1.
        - One-Hot Encoding: Unknown values sẽ thành vector [0, 0, 0, ...].
        
        Parameters
        ----------
        data : DataFrame
            DataFrame chứa các cột phân loại cần mã hóa.
        strategy : str, optional
            Phương pháp mã hóa:
            - 'label': Label Encoding (gán số nguyên cho mỗi category)
            - 'onehot': One-Hot Encoding (tạo cột dummy)
            Mặc định là 'onehot'.
        encoders : dict or None, optional
            Dictionary chứa thông tin encoding đã fit từ train set.
            Nếu None và fit=True, sẽ tạo encoders mới. Mặc định là None.
        fit : bool, optional
            Nếu True, fit encoders với data (cho train set).
            Nếu False, sử dụng encoders đã fit (cho test set).
            Mặc định là False.
        
        Returns
        -------
        tuple
            (DataFrame, dict) - DataFrame đã mã hóa và dictionary encoders.
        """
        target = data.copy()
        categorical_cols = target.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if encoders is None:
            encoders = {}
        
        DataTransformer._log(f"[Encode] strategy='{strategy}', fit={fit}")
        
        if strategy == 'label':
            if fit:
                # Fit: Học mapping từ train set
                for col in categorical_cols:
                    unique_vals = target[col].unique().tolist()
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    encoders[col] = mapping
                    target[col] = target[col].map(mapping)
            else:
                # Transform: Áp dụng mapping đã học, unknown -> -1
                for col in categorical_cols:
                    if col in encoders:
                        mapping = encoders[col]
                        target[col] = target[col].map(lambda x: mapping.get(x, -1))
                    else:
                        DataTransformer._log(f"Warning: No encoder found for column '{col}'")
                        
        elif strategy == 'onehot':
            if not categorical_cols:
                return target, encoders
                
            if fit:
                # Fit: Tạo và fit OneHotEncoder
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_data = ohe.fit_transform(target[categorical_cols])
                
                # Lưu encoder và tên cột
                encoders['_onehot_encoder'] = ohe
                encoders['_onehot_cols'] = categorical_cols
                
                # Tạo tên cột mới
                feature_names = ohe.get_feature_names_out(categorical_cols)
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=target.index)
                
                # Ghép với data gốc và xóa cột categorical
                target = target.drop(columns=categorical_cols)
                target = pd.concat([target, encoded_df], axis=1)
            else:
                # Transform: Sử dụng encoder đã fit
                if '_onehot_encoder' in encoders:
                    ohe = encoders['_onehot_encoder']
                    original_cols = encoders['_onehot_cols']
                    
                    # Chỉ transform các cột có trong encoder gốc
                    cols_to_encode = [c for c in categorical_cols if c in original_cols]
                    
                    if cols_to_encode:
                        # Unknown values sẽ tự động thành [0, 0, 0, ...] với handle_unknown='ignore'
                        encoded_data = ohe.transform(target[cols_to_encode])
                        feature_names = ohe.get_feature_names_out(original_cols)
                        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=target.index)
                        
                        target = target.drop(columns=cols_to_encode)
                        target = pd.concat([target, encoded_df], axis=1)
                else:
                    DataTransformer._log("Warning: No OneHotEncoder found in encoders")
            
        return target, encoders

    @staticmethod
    def scale_features(
        data: pd.DataFrame,
        strategy: str = 'standard',
        scaler: Optional[Union[StandardScaler, RobustScaler]] = None,
        exclude_features: Optional[list[str]] = None,
        fit: bool = False
    ) -> tuple[pd.DataFrame, Union[StandardScaler, RobustScaler], list[str]]:
        """
        Chuẩn hóa (scaling) các cột số.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame chứa dữ liệu cần chuẩn hóa.
        strategy : str, optional
            Phương pháp chuẩn hóa:
            - 'standard': StandardScaler (z-score normalization)
            - 'robust': RobustScaler (sử dụng median và IQR)
            Mặc định là 'standard'.
        scaler : object or None, optional
            Scaler đã được fit. Nếu None và fit=True, sẽ tạo scaler mới.
            Mặc định là None.
        exclude_features : list or None, optional
            Danh sách các cột không scale. Mặc định là None.
        fit : bool, optional
            Nếu True, fit scaler với data. Nếu False, chỉ transform. Mặc định là False.
        
        Returns
        -------
        tuple
            (DataFrame, scaler, list) - DataFrame đã scale, scaler object,
            và danh sách các cột đã được scale.
        """
        target = data.copy()
        if exclude_features is None: exclude_features = []
        
        numeric_cols = target.select_dtypes(include=np.number).columns.tolist()
        cols_to_scale = [c for c in numeric_cols if c not in exclude_features]
        
        DataTransformer._log(f"[Scale] strategy='{strategy}', fit={fit}")
        
        if fit or scaler is None:
            if strategy == 'standard': scaler = StandardScaler()
            elif strategy == 'robust': scaler = RobustScaler()
            else: scaler = StandardScaler()
            
            if cols_to_scale:
                target[cols_to_scale] = scaler.fit_transform(target[cols_to_scale])
        else:
            if cols_to_scale:
                try:
                    target[cols_to_scale] = scaler.transform(target[cols_to_scale])
                except ValueError as e:
                    DataTransformer._log(f"Warning: Scaling failed (feature mismatch?): {e}")
                    
        return target, scaler, cols_to_scale

    @staticmethod
    def remove_duplicates(
        data: pd.DataFrame,
        subset: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Loại bỏ các hàng trùng lặp trong DataFrame.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame cần loại bỏ trùng lặp.
        subset : list or None, optional
            Danh sách các cột để xác định trùng lặp.
            Nếu None, sử dụng tất cả các cột. Mặc định là None.
        
        Returns
        -------
        DataFrame
            DataFrame đã loại bỏ các hàng trùng lặp.
        """
        initial = len(data)
        data = data.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)
        removed = initial - len(data)
        DataTransformer._log(f"Removed {removed} duplicate rows")
        return data

    @staticmethod
    def drop_null_targets(
        data: pd.DataFrame,
        target_column: str
    ) -> pd.DataFrame:
        """
        Loại bỏ các hàng có giá trị target null.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame chứa dữ liệu cần xử lý.
        target_column : str
            Tên cột target cần kiểm tra.
        
        Returns
        -------
        DataFrame
            DataFrame đã loại bỏ các hàng có target null.
        """
        initial_rows = len(data)
        data = data.dropna(subset=[target_column]).reset_index(drop=True)
        rows_removed = initial_rows - len(data)
        
        if rows_removed > 0:
            DataTransformer._log(f"Dropped {rows_removed} rows with null target '{target_column}'")
        
        return data

    @staticmethod
    def drop_features(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """
        Xóa các cột được chỉ định khỏi DataFrame.
        
        Parameters
        ----------
        data : DataFrame
            DataFrame cần xóa cột.
        features : list
            Danh sách tên các cột cần xóa.
        
        Returns
        -------s
        DataFrame
            DataFrame đã xóa các cột được chỉ định.
        """
        DataTransformer._log(f"Dropping features: {features}")
        return data.drop(columns=features, errors='ignore')
