from __future__ import annotations

import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger

# Thiết lập style cho seaborn
sns.set_theme(style="whitegrid")

# Logger riêng cho ModelVisualize
MODEL_VIS_LOGGER = get_logger("MODEL_VISUALIZE")

SEP_LEN = 60
SEP = "=" * SEP_LEN

class ModelVisualizer:
	"""
	Class trực quan hóa kết quả đánh giá và phân tích mô hình Machine Learning.
	
	Attributes
	----------
	evaluation_results : dict
		Dictionary chứa 'results' (list of dicts) và 'best_model_name' (str)
	"""
	
	def __init__(self, evaluation_results: dict[str, Any]) -> None:
		"""
		Parameters
		----------
		evaluation_results : dict
			Kết quả từ ModelTrainer.evaluate_models()
		"""
		self.evaluation_results: dict[str, Any] = evaluation_results
		
	@staticmethod
	def _log(message: str) -> None:
		MODEL_VIS_LOGGER.info(message)
	
	def plot_model_comparison(self, save_path: Optional[str] = None) -> None:
		"""
		Vẽ 4 biểu đồ so sánh MSE, RMSE, MAE, R² của các models.
		
		Parameters
		----------
		save_path : str, optional
			Đường dẫn lưu file. Nếu None, chỉ hiển thị
		"""
		if not self.evaluation_results or not self.evaluation_results.get('results'):
			raise ValueError("No evaluation results found.")
			
		results = self.evaluation_results['results']
		best_model = self.evaluation_results.get('best_model_name', 'Unknown')
		df = pd.DataFrame(results)
		
		metrics = [
			('mse', 'Mean Squared Error (MSE)', True),
			('rmse', 'Root Mean Squared Error (RMSE)', True),
			('mae', 'Mean Absolute Error (MAE)', True),
			('r2_score', 'R² Score', False)
		]
		
		fig, axes = plt.subplots(2, 2, figsize=(16, 10))
		axes = axes.flatten()
		
		for idx, (metric, title, ascending) in enumerate(metrics):
			df_sorted = df.sort_values(by=metric, ascending=ascending)
			palette = ['crimson' if m == best_model else 'steelblue' for m in df_sorted['model_name']]
			
			ax = axes[idx]
			sns.barplot(data=df_sorted, x='model_name', y=metric, palette=palette, 
						hue='model_name', legend=False, edgecolor='black', linewidth=0.8, ax=ax)
			
			# Thêm giá trị
			for patch, val in zip(ax.patches, df_sorted[metric]):
				ax.text(patch.get_x() + patch.get_width()/2, patch.get_height(),
						f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
			
			ax.set_title(title, fontsize=12, fontweight='bold')
			ax.set_xlabel('Models', fontsize=10)
			ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
			ax.tick_params(axis='x', rotation=45, labelsize=9)
			ax.grid(axis='y', alpha=0.3, linestyle='--')
			ax.set_axisbelow(True)
		
		fig.suptitle(f'Model Performance Comparison | Best: {best_model}', 
					fontsize=14, fontweight='bold', y=0.995)
		plt.tight_layout(rect=[0, 0, 1, 0.985])
		
		if save_path:
			os.makedirs(os.path.dirname(save_path), exist_ok=True)
			plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
			self._log(f"✓ Plot saved: {save_path}")
		plt.show()
	
	def plot_feature_importance(
		self,
		importance_df: pd.DataFrame,
		save_path: Optional[str] = None
	) -> None:
		"""
		Vẽ biểu đồ feature importance.
		
		Parameters
		----------
		importance_df : DataFrame
			DataFrame với columns ['feature', 'importance'], đã sorted
		save_path : str, optional
			Đường dẫn lưu file
		"""
		if importance_df is None or importance_df.empty:
			raise ValueError("importance_df is empty")
		
		# Lấy model name từ evaluation_results
		model_name = self.evaluation_results.get('best_model_name', 'Model') if self.evaluation_results else 'Model'
		
		# Sắp xếp ascending để feature quan trọng nhất ở trên
		importance_df = importance_df.sort_values('importance', ascending=True)
		
		plt.figure(figsize=(10, max(6, len(importance_df) * 0.3)))
		
		colors = sns.color_palette("viridis", n_colors=len(importance_df))
		ax = sns.barplot(data=importance_df, y='feature', x='importance', 
						palette=colors, hue='feature', legend=False, edgecolor='black', linewidth=0.6)
		
		# Thêm giá trị
		max_imp = importance_df['importance'].max()
		for patch, val in zip(ax.patches, importance_df['importance']):
			ax.text(patch.get_width() + max_imp * 0.01, patch.get_y() + patch.get_height()/2,
					f'{val:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
		
		ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
		ax.set_ylabel('Features', fontsize=11, fontweight='bold')
		ax.set_title(f'Feature Importance | {model_name}', fontsize=13, fontweight='bold')
		ax.grid(axis='x', alpha=0.3, linestyle='--')
		ax.set_axisbelow(True)
		
		plt.tight_layout()
		
		if save_path:
			os.makedirs(os.path.dirname(save_path), exist_ok=True)
			plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
			self._log(f"✓ Feature importance saved: {save_path}")
		plt.show()