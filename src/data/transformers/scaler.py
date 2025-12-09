"""
FeatureScaler - Chuẩn hóa các features số.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils.logging import get_logger

LOGGER = get_logger("FEATURE_SCALER")


class FeatureScaler:
	"""
	Class chuyên chuẩn hóa (scaling) các cột số.
	"""

	@staticmethod
	def _log(message: str) -> None:
		LOGGER.info(message)

	@staticmethod
	def scale(
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

		Raises
		------
		ValueError
			Nếu strategy không thuộc {'standard', 'robust'}.
		"""
		target = data.copy()
		if exclude_features is None:
			exclude_features = []
		
		numeric_cols = target.select_dtypes(include=np.number).columns.tolist()
		cols_to_scale = [c for c in numeric_cols if c not in exclude_features]
		
		FeatureScaler._log(f"[Scale] strategy='{strategy}', fit={fit}")
		
		if fit or scaler is None:
			if strategy == 'standard':
				scaler = StandardScaler()
			elif strategy == 'robust':
				scaler = RobustScaler()
			else:
				raise ValueError(f"Invalid scaling strategy '{strategy}'. Supported: 'standard', 'robust'.")
			
			if cols_to_scale:
				target[cols_to_scale] = scaler.fit_transform(target[cols_to_scale])
		else:
			if cols_to_scale:
				try:
					target[cols_to_scale] = scaler.transform(target[cols_to_scale])
				except ValueError as e:
					FeatureScaler._log(f"Warning: Scaling failed (feature mismatch?): {e}")
					
		return target, scaler, cols_to_scale
