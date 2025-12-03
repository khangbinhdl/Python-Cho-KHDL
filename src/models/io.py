from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from typing import Any, Optional, Union

import joblib
import pandas as pd

from src.utils.logging import get_logger

# Logger riêng
LOGGER = get_logger("MODEL_IO")

class ModelIO:
    """
    Class chuyên trách việc Lưu/Đọc (Save/Load) model và kết quả.
    """

    @staticmethod
    def _log(message: str) -> None:
        LOGGER.info(message)

    @staticmethod
    def load_model(filepath: str, method: str = 'joblib') -> Any:
        """
        Nạp mô hình đã lưu từ file.

        Parameters
        ----------
        filepath : str
            Đường dẫn tuyệt đối hoặc tương đối đến file mô hình.
        method : str, optional
            Phương thức deserialization ('joblib' hoặc 'pickle').
            Mặc định là 'joblib'.

        Returns
        -------
        object
            Model object đã được nạp từ file.

        Raises
        ------
        ValueError
            Nếu method không hợp lệ.
        """
        try:
            if method == 'joblib':
                model = joblib.load(filepath)
            elif method == 'pickle':
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
            else:
                raise ValueError("method phải là 'joblib' hoặc 'pickle'")
                
            ModelIO._log(f"✓ Đã nạp model từ: {filepath}")
            return model
            
        except Exception as e:
            ModelIO._log(f"✗ Lỗi khi nạp model: {str(e)}")
            raise

    @staticmethod
    def save_model(
        model: Any,
        model_name: str,
        filepath: Optional[str] = None,
        method: str = 'joblib'
    ) -> str:
        """
        Lưu mô hình đã huấn luyện vào file.

        Parameters
        ----------
        model : object
            Mô hình cần lưu.
        model_name : str
            Tên của mô hình (dùng để đặt tên file nếu filepath=None).
        filepath : str or None, optional
            Đường dẫn file để lưu model.
            Nếu None, tự động tạo tên file với timestamp trong thư mục 'outputs/models/'.
            Mặc định là None.
        method : str, optional
            Phương thức serialization ('joblib' hoặc 'pickle').
            Mặc định là 'joblib'.

        Returns
        -------
        str
            Đường dẫn tuyệt đối của file model đã lưu.

        Raises
        ------
        ValueError
            Nếu method không hợp lệ.
        """
        # Tạo tên file nếu chưa có
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if method == 'joblib':
                filepath = f"outputs/models/{model_name}_{timestamp}.pkl"
            else:
                filepath = f"outputs/models/{model_name}_{timestamp}.pickle"
        
        # Tạo thư mục cha nếu chưa có
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
        try:
            if method == 'joblib':
                joblib.dump(model, filepath)
            elif method == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            else:
                raise ValueError("method phải là 'joblib' hoặc 'pickle'")
                
            ModelIO._log(f"✓ Đã lưu model '{model_name}' tại: {filepath}")
            return filepath
            
        except Exception as e:
            ModelIO._log(f"✗ Lỗi khi lưu model: {str(e)}")
            raise

    @staticmethod
    def save_results(
        results: list[dict[str, Any]],
        filepath: Optional[str] = None,
        format: str = 'csv'
    ) -> Optional[str]:
        """
        Lưu kết quả đánh giá các mô hình vào file.

        Parameters
        ----------
        results : list of dict
            Danh sách kết quả đánh giá.
        filepath : str or None, optional
            Đường dẫn file để lưu kết quả.
            Nếu None, tự động tạo tên file với timestamp trong thư mục 'outputs/results/'.
            Mặc định là None.
        format : str, optional
            Định dạng file output ('csv' hoặc 'json').
            Mặc định là 'csv'.

        Returns
        -------
        str
            Đường dẫn tuyệt đối của file đã lưu.

        Raises
        ------
        ValueError
            Nếu format không hợp lệ.
        """
        if not results:
            ModelIO._log("Không có kết quả nào để lưu.")
            return None
        
        # Tạo tên file nếu chưa có
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"outputs/results/model_results_{timestamp}.{format}"
        
        # Tạo thư mục cha nếu chưa có
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        try:
            if format == 'csv':
                df = pd.DataFrame(results)
                df.to_csv(filepath, index=False)
            elif format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError("format phải là 'csv' hoặc 'json'")
                
            ModelIO._log(f"✓ Đã lưu kết quả tại: {filepath}")
            return filepath
            
        except Exception as e:
            ModelIO._log(f"✗ Lỗi khi lưu kết quả: {str(e)}")
            raise
