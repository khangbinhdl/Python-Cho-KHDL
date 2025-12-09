"""
MissingValueHandler - Xử lý giá trị thiếu trong dữ liệu.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.data.transformers.type_converter import TypeConverter

LOGGER = get_logger("MISSING_HANDLER")


class MissingValueHandler:
	"""
	Class chuyên xử lý giá trị thiếu (missing values) trong DataFrame.
	"""

	@staticmethod
	def _log(message: str) -> None:
		LOGGER.info(message)

	@staticmethod
	def _handle_numeric(
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
		
		# FIT: Học tham số cho mean/median/mode
		if fit and strategy in ("mean", "median", "mode"):
			for col in cols_to_process:
				if strategy == "mean":
					val = data[col].mean()
				elif strategy == "median":
					val = data[col].median()
				else:  # mode
					mode = data[col].mode()
					val = mode.iloc[0] if not mode.empty else data[col].median()
				new_learned[col] = val
		
		# FIT: Học fallback cho strategy="drop" (dùng median)
		if fit and strategy == "drop":
			for col in cols_to_process:
				val = data[col].median()
				new_learned[col] = val
		
		# FIT: Học median cho ffill/bfill (để dùng khi transform)
		if fit and strategy in ("ffill", "bfill"):
			for col in cols_to_process:
				val = data[col].median()
				new_learned[col] = val
		
		# TRANSFORM: Áp dụng
		if strategy == "drop":
			if fit:
				if cols_to_process:
					data = data.dropna(subset=cols_to_process)
			else:
				for col in cols_to_process:
					val = new_learned.get(col)
					if val is not None:
						data[col] = data[col].fillna(val)
		elif strategy in ("mean", "median", "mode"):
			for col in cols_to_process:
				val = new_learned.get(col)
				if val is not None:
					data[col] = data[col].fillna(val)
		elif strategy in ("ffill", "bfill"):
			if fit:
				if cols_to_process:
					if strategy == "ffill":
						data[cols_to_process] = data[cols_to_process].ffill()
					else:
						data[cols_to_process] = data[cols_to_process].bfill()
			else:
				for col in cols_to_process:
					val = new_learned.get(col)
					if val is not None:
						data[col] = data[col].fillna(val)
		
		return data, new_learned

	@staticmethod
	def _handle_categorical(
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
		
		# FIT: Học tham số cho mode/constant
		if fit and strategy in ("mode", "constant"):
			for col in cols_to_process:
				if strategy == "mode":
					mode = data[col].mode()
					val = mode.iloc[0] if not mode.empty else "Unknown"
				else:  # constant
					val = "Unknown"
				new_learned[col] = val
		
		# FIT: Học fallback cho strategy="drop"
		if fit and strategy == "drop":
			for col in cols_to_process:
				mode = data[col].mode()
				val = mode.iloc[0] if not mode.empty else "Unknown"
				new_learned[col] = val
		
		# FIT: Học mode cho ffill/bfill
		if fit and strategy in ("ffill", "bfill"):
			for col in cols_to_process:
				mode = data[col].mode()
				val = mode.iloc[0] if not mode.empty else "Unknown"
				new_learned[col] = val
		
		# TRANSFORM: Áp dụng
		if strategy == "drop":
			if fit:
				if cols_to_process:
					data = data.dropna(subset=cols_to_process)
			else:
				for col in cols_to_process:
					val = new_learned.get(col, "Unknown")
					data[col] = data[col].fillna(val)
		elif strategy == "mode":
			for col in cols_to_process:
				val = new_learned.get(col)
				if val is not None:
					data[col] = data[col].fillna(val)
				else:
					mode = data[col].mode()
					val = mode.iloc[0] if not mode.empty else "Unknown"
					data[col] = data[col].fillna(val)
		elif strategy == "constant":
			for col in cols_to_process:
				val = new_learned.get(col, "Unknown")
				data[col] = data[col].fillna(val)
		elif strategy in ("ffill", "bfill"):
			if fit:
				if cols_to_process:
					if strategy == "ffill":
						data[cols_to_process] = data[cols_to_process].ffill()
					else:
						data[cols_to_process] = data[cols_to_process].bfill()
			else:
				for col in cols_to_process:
					val = new_learned.get(col, "Unknown")
					data[col] = data[col].fillna(val)
		
		return data, new_learned

	@staticmethod
	def _handle_datetime(
		data: pd.DataFrame,
		columns: list[str],
		strategy: str,
		learned_values: dict[str, Any],
		fit: bool
	) -> tuple[pd.DataFrame, dict[str, Any]]:
		"""
		Xử lý giá trị thiếu cho các cột datetime.
		
		Parameters
		----------
		data : DataFrame
			DataFrame chứa dữ liệu cần xử lý.
		columns : list
			Danh sách các cột datetime cần xử lý.
		strategy : str
			Chiến lược xử lý ('drop', 'ffill', 'bfill', 'median').
		learned_values : dict
			Dictionary chứa các giá trị đã học (tên cột -> giá trị điền).
		fit : bool
			Nếu True, học các giá trị thống kê từ data.
		
		Returns
		-------
		tuple
			(DataFrame, dict) - DataFrame đã xử lý và dictionary learned_values cập nhật.
		"""
		if not columns:
			return data, learned_values
		
		new_learned = learned_values.copy()
		
		# FIT: Học median cho tất cả strategy
		if fit:
			for col in columns:
				non_null = data[col].dropna()
				if len(non_null) > 0:
					val = non_null.median()
				else:
					val = pd.NaT
				new_learned[col] = val
		
		# TRANSFORM: Áp dụng
		if strategy == "drop":
			if fit:
				data = data.dropna(subset=columns)
			else:
				for col in columns:
					val = new_learned.get(col)
					if val is not None and pd.notna(val):
						data[col] = data[col].fillna(val)
		elif strategy in ("ffill", "bfill"):
			if fit:
				if strategy == "ffill":
					data[columns] = data[columns].ffill()
				else:
					data[columns] = data[columns].bfill()
			else:
				for col in columns:
					val = new_learned.get(col)
					if val is not None and pd.notna(val):
						data[col] = data[col].fillna(val)
		elif strategy == "median":
			for col in columns:
				val = new_learned.get(col)
				if val is not None and pd.notna(val):
					data[col] = data[col].fillna(val)
		
		return data, new_learned

	@staticmethod
	def handle(
		data: pd.DataFrame,
		strategies: dict[str, str],
		learned_values: dict[str, dict[str, Any]] | None = None,
		exclude_features: list[str] | None = None,
		fit: bool = False
	) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
		"""
		Xử lý giá trị thiếu theo các chiến lược được chỉ định.
		
		Parameters
		----------
		data : DataFrame
			DataFrame chứa dữ liệu cần xử lý.
		strategies : dict
			Dictionary chứa chiến lược xử lý cho từng loại cột:
			- 'num': Chiến lược cho cột số ('mean', 'median', 'mode', 'drop', 'ffill', 'bfill')
			- 'cat': Chiến lược cho cột phân loại ('mode', 'constant', 'drop', 'ffill', 'bfill')
			- 'dt': Chiến lược cho cột datetime ('drop', 'ffill', 'bfill', 'median')
		learned_values : dict or None, optional
			Dictionary chứa các giá trị đã học từ tập training. Mặc định là None.
		exclude_features : list or None, optional
			Danh sách các cột không xử lý. Mặc định là None.
		fit : bool, optional
			Nếu True, học các giá trị thống kê từ data. Mặc định là False.
		
		Returns
		-------
		tuple
			(DataFrame, dict) - DataFrame đã xử lý và dictionary learned_values.
		"""
		target = data.copy()
		exclude_features = exclude_features or []
		
		num_strategy = strategies.get('num', 'median')
		cat_strategy = strategies.get('cat', 'mode')
		dt_strategy = strategies.get('dt', 'drop')
		
		col_types = TypeConverter.auto_detect_columns(target)
		numeric_cols = col_types['numeric']
		categorical_cols = col_types['categorical']
		datetime_cols = col_types['datetime']
		
		new_learned_values = {'num': {}, 'cat': {}, 'dt': {}}
		if learned_values:
			new_learned_values = {
				'num': learned_values.get('num', {}).copy(),
				'cat': learned_values.get('cat', {}).copy(),
				'dt': learned_values.get('dt', {}).copy()
			}

		MissingValueHandler._log(f"Handling missing values fit={fit} | strategies={strategies}")
		initial_rows = len(target)

		# 1. Xử lý datetime
		target, new_learned_values['dt'] = MissingValueHandler._handle_datetime(
			target, datetime_cols, dt_strategy,
			new_learned_values['dt'], fit
		)

		# 2. Xử lý numeric
		target, new_learned_values['num'] = MissingValueHandler._handle_numeric(
			target, numeric_cols, num_strategy,
			new_learned_values['num'], exclude_features, fit
		)

		# 3. Xử lý categorical
		target, new_learned_values['cat'] = MissingValueHandler._handle_categorical(
			target, categorical_cols, cat_strategy,
			new_learned_values['cat'], exclude_features, fit
		)

		rows_removed = initial_rows - len(target)
		if rows_removed > 0:
			MissingValueHandler._log(f"Dropped {rows_removed} rows due to missing values")
			
		return target.reset_index(drop=True), new_learned_values
