"""
TypeConverter - Xử lý phát hiện và chuyển đổi kiểu dữ liệu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger("TYPE_CONVERTER")


class TypeConverter:
	"""
	Class chuyên xử lý phát hiện và chuyển đổi kiểu dữ liệu.
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
		TypeConverter._log(f"Auto-detecting numeric columns (threshold={threshold:.0%})...")
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
			TypeConverter._log(f"Auto-converted {len(converted)} columns to numeric: {converted}")
		return data

	@staticmethod
	def convert_to_datetime(
		data: pd.DataFrame,
		columns: list[str] | None = None,
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
		TypeConverter._log("Converting columns to datetime...")
		
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
			TypeConverter._log(f"Converted {len(converted_cols)} columns to datetime: {converted_cols}")
		return data
