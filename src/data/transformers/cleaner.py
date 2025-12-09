"""
DataCleaner - Các thao tác làm sạch dữ liệu.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger("DATA_CLEANER")


class DataCleaner:
	"""
	Class chuyên thực hiện các thao tác làm sạch dữ liệu cơ bản.
	"""

	@staticmethod
	def _log(message: str) -> None:
		LOGGER.info(message)

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
		DataCleaner._log(f"Removed {removed} duplicate rows")
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
			DataCleaner._log(f"Dropped {rows_removed} rows with null target '{target_column}'")
		
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
		-------
		DataFrame
			DataFrame đã xóa các cột được chỉ định.
		"""
		DataCleaner._log(f"Dropping features: {features}")
		return data.drop(columns=features, errors='ignore')
