"""
OutlierHandler - Xử lý giá trị ngoại lai trong dữ liệu.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils.logging import get_logger

LOGGER = get_logger("OUTLIER_HANDLER")


class OutlierHandler:
	"""
	Class chuyên xử lý giá trị ngoại lai (outliers) trong DataFrame.
	"""

	@staticmethod
	def _log(message: str) -> None:
		LOGGER.info(message)

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
		OutlierHandler._log("Cleaning negative values in all numeric columns...")
		numeric_columns = data.select_dtypes(include=[np.number]).columns
		for col in numeric_columns:
			data[col] = np.abs(data[col])
		return data

	@staticmethod
	def handle(
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
		if exclude_features is None:
			exclude_features = []
		
		OutlierHandler._log(f"[Outlier] method='{method}' - Removing outliers")
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
				else:  # zscore
					mean = target[col].mean()
					std = target[col].std()
					lower = mean - 3 * std
					upper = mean + 3 * std
				mask &= (target[col] >= lower) & (target[col] <= upper)
			target = target[mask]
			
		elif method == 'isolation_forest':
			if cols_to_process:
				iso = IsolationForest(
					n_estimators=200,
					contamination=0.05,
					random_state=42
				)
				yhat = iso.fit_predict(target[cols_to_process])
				target = target[yhat == 1]

		rows_removed = initial_rows - len(target)
		OutlierHandler._log(f"Removed {rows_removed} rows as outliers")
		return target.reset_index(drop=True)
