"""
FeatureEncoder - Mã hóa các cột categorical.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.utils.logging import get_logger

LOGGER = get_logger("FEATURE_ENCODER")


class FeatureEncoder:
	"""
	Class chuyên mã hóa các cột categorical thành dạng số.
	"""

	@staticmethod
	def _log(message: str) -> None:
		LOGGER.info(message)

	@staticmethod
	def encode(
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
		
		FeatureEncoder._log(f"[Encode] strategy='{strategy}', fit={fit}")
		
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
						FeatureEncoder._log(f"Warning: No encoder found for column '{col}'")
						
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
					FeatureEncoder._log("Warning: No OneHotEncoder found in encoders")
			
		return target, encoders
