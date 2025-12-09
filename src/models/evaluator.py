from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logging import get_logger

# Logger riêng
LOGGER = get_logger("MODEL_EVALUATOR")

class ModelEvaluator:
	"""
	Class chuyên trách việc tính toán metrics và đánh giá mô hình.
	"""

	@staticmethod
	def _log(message: str) -> None:
		LOGGER.info(message)

	@staticmethod
	def evaluate_model(
		model: Any,
		X_test: pd.DataFrame,
		y_test: pd.Series,
		model_name: str
	) -> Optional[dict[str, Union[str, float]]]:
		"""
		Đánh giá một mô hình trên tập test.

		Parameters
		----------
		model : object
			Mô hình đã huấn luyện.
		X_test : DataFrame
			Features của tập test.
		y_test : Series
			Target của tập test.
		model_name : str
			Tên của mô hình.

		Returns
		-------
		dict or None
			Dictionary chứa các metrics đánh giá (MSE, RMSE, MAE, R2).
			Trả về None nếu xảy ra lỗi trong quá trình đánh giá mô hình.
		"""
		try:
			y_pred = model.predict(X_test)
			
			mse = mean_squared_error(y_test, y_pred)
			rmse = np.sqrt(mse)
			mae = mean_absolute_error(y_test, y_pred)
			r2 = r2_score(y_test, y_pred)

			result: dict[str, Union[str, float]] = {
				'model_name': model_name,
				'mse': mse,
				'rmse': rmse, 
				'mae': mae,
				'r2_score': r2
			}
			
			ModelEvaluator._log(f"✓ {model_name} evaluated: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
			return result
			
		except Exception as e:
			ModelEvaluator._log(f"✗ Lỗi khi đánh giá {model_name}: {str(e)}")
			return None

	@staticmethod
	def get_feature_importance(
		model: Any,
		feature_names: ArrayLike,
		model_name: str,
		top_n: Optional[int] = None
	) -> pd.DataFrame:
		"""
		Lấy độ quan trọng của các features từ mô hình.

		Parameters
		----------
		model : object
			Mô hình đã huấn luyện.
		feature_names : array-like
			Danh sách tên các features.
		model_name : str
			Tên của mô hình.
		top_n : int, optional
			Số lượng features quan trọng nhất để trả về.
			Nếu None, trả về tất cả. Mặc định là None.

		Returns
		-------
		DataFrame
			DataFrame chứa tên feature và độ quan trọng.
		
		Raises
		------
		ValueError
			Nếu model không hỗ trợ trích xuất feature importance.
		"""
		# Lấy feature importance dựa trên loại mô hình
		if hasattr(model, 'feature_importances_'):
			# Tree-based models: RandomForest, LightGBM
			importances = model.feature_importances_
			ModelEvaluator._log(f"Sử dụng feature_importances_ cho {model_name}")
			
		elif hasattr(model, 'coef_'):
			# Linear models: LinearRegression, Ridge, Lasso, ElasticNet
			# Sử dụng giá trị tuyệt đối của hệ số
			importances = np.abs(model.coef_)
			ModelEvaluator._log(f"Sử dụng absolute coefficients cho {model_name}")
			
		else:
			raise ValueError(f"Model '{model_name}' không hỗ trợ trích xuất feature importance")
		
		# Tạo DataFrame và sắp xếp
		importance_df = pd.DataFrame({
			'feature': feature_names,
			'importance': importances
		}).sort_values('importance', ascending=False)
		
		if top_n:
			importance_df = importance_df.head(top_n)
			
		ModelEvaluator._log(f"✓ Đã trích xuất feature importance cho {model_name}")
		return importance_df
