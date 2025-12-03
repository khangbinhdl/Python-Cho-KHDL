from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger

# Thiết lập style cho seaborn
sns.set_theme(style="whitegrid")

# Logger riêng cho EDA
LOGGER = get_logger("EDA")

SEP_LEN = 60
SEP = "=" * SEP_LEN

class EDA:
	"""
	Class thực hiện phân tích dữ liệu khám phá (Exploratory Data Analysis)
	
	Cung cấp các phương thức để phân tích thống kê mô tả, tương quan,
	phân phối dữ liệu và phát hiện ngoại lai thông qua trực quan hóa.

	Attributes
	----------
	data : DataFrame
		Dữ liệu cần phân tích
	"""
	
	def __init__(self, data: pd.DataFrame, show_plots: bool = True) -> None:
		"""
		Khởi tạo đối tượng EDA với dữ liệu
		
		Parameters
		----------
		data : DataFrame
			DataFrame chứa dữ liệu cần phân tích
		show_plots : bool, optional
			Quyết định có hiển thị các biểu đồ hay không. Mặc định là True.
		"""
		self.data: pd.DataFrame = data
		self.show_plots: bool = show_plots

	def __str__(self) -> str:
		"""
		Biểu diễn chuỗi thân thiện với người dùng
		
		Returns
		-------
		str
			Chuỗi mô tả trạng thái của EDA
		"""
		if self.data is None:
			return "EDA (chưa có dữ liệu)"
		return f"EDA: {self.data.shape[0]} dòng, {self.data.shape[1]} cột"

	def __repr__(self) -> str:
		"""
		Biểu diễn chuỗi dành cho developer
		
		Returns
		-------
		str
			Chuỗi mô tả chi tiết
		"""
		shape = self.data.shape if self.data is not None else (0, 0)
		return f"EDA(rows={shape[0]}, cols={shape[1]}, show_plots={self.show_plots})"
		
	@staticmethod
	def _log(message: str) -> None:
		"""
		Ghi log thông điệp với logger của EDA
		
		Parameters
		----------
		message : str
			Thông điệp cần ghi log
		"""
		LOGGER.info(message)

	@staticmethod
	def _sep() -> None:
		"""
		In ra một dòng phân cách trong log
		
		Ghi một dòng trống và một dòng dấu "=" để phân tách các phần trong log,
		giúp dễ đọc hơn khi xem output.
		"""
		LOGGER.info("")      # một dòng trống có timestamp
		LOGGER.info(SEP)     # separator có timestamp

	def summary_statistics(self, save_path: Optional[str] = None) -> None:
		"""
		In ra các thống kê mô tả cơ bản của các cột số
		
		Hiển thị các chỉ số thống kê như count, mean, std, min, max, 
		và các phân vị (25%, 50%, 75%) cho tất cả các cột số trong DataFrame.
		Bổ sung thông tin về kiểu dữ liệu, giá trị thiếu, trùng lặp, value counts và độ lệch.

		Parameters
		----------
		save_path : str, optional
			Đường dẫn thư mục để lưu biểu đồ. Nếu None, chỉ hiển thị.
			Mặc định là None

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		
		Notes
		-----
		Sử dụng phương thức describe() của pandas để tính toán các thống kê
		"""
		if self.data is None:
			raise ValueError("Data not loaded. Call load_data() first.")
		
		# 1. Tổng quan info
		self._sep()
		self._log("DATASET INFORMATION")
		self._log(SEP)
		from io import StringIO
		buf = StringIO()
		self.data.info(buf=buf)
		for line in buf.getvalue().splitlines():
			self._log(line)
		
		# 2. Thống kê mô tả
		self._sep()
		self._log("SUMMARY STATISTICS FOR NUMERIC COLUMNS")
		self._log(SEP)
		desc = self.data.describe().to_string()
		for line in desc.splitlines():
			self._log(line)
		
		# 3. Missing values
		missing_counts = self.data.isnull().sum()
		missing_percentages = (missing_counts / len(self.data)) * 100
		missing_data = pd.DataFrame({
			'Missing_Count': missing_counts,
			'Missing_Percentage': missing_percentages
		})
		missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
			by='Missing_Percentage', ascending=False
		)
		
		if not missing_data.empty:
			self._sep()
			self._log("MISSING VALUES")
			self._log(SEP)
			for line in missing_data.to_string().splitlines():
				self._log(line)
			# Vẽ biểu đồ như cũ
			plt.figure(figsize=(10, 5))
			import seaborn as sns
			sns.barplot(x=missing_data.index, y=missing_data['Missing_Percentage'],
						palette='rocket', hue=missing_data.index, legend=False)
			plt.title('Missing Values Percentage by Column', fontsize=14, fontweight='bold')
			plt.xlabel('Columns', fontsize=12)
			plt.ylabel('Missing Percentage (%)', fontsize=12)
			plt.xticks(rotation=45, ha='right')
			plt.tight_layout()
			
			# Lưu biểu đồ nếu có đường dẫn
			if save_path:
				os.makedirs(save_path, exist_ok=True)
				plt.savefig(f'{save_path}/missing_values.png', dpi=300, bbox_inches='tight', facecolor='white')
				self._log(f"✓ Missing values plot saved to: {save_path}/missing_values.png")
			
			if self.show_plots:
				plt.show()
			else:
				plt.close()
		else:
			self._sep()
			self._log("MISSING VALUES: No missing values found")
			self._log(SEP)
		
		# 4. Duplicate rows
		duplicate_count = self.data.duplicated().sum()
		self._sep()
		self._log(f"DUPLICATE ROWS: {duplicate_count} rows")
		self._log(SEP)
		
		# 5. Value counts (Top 5 cho mỗi cột)
		self._sep()
		self._log("VALUE COUNTS (TOP 5 FOR EACH COLUMN)")
		self._log(SEP)
		for col in self.data.columns:
			self._log(f"--- {col} ---")
			top5 = self.data[col].value_counts().head(5).to_string()
			for line in top5.splitlines():
				self._log(line)
		
		# 6. Skewness cho các cột số
		numeric_cols = self.data.select_dtypes(include=np.number).columns
		if len(numeric_cols) > 0:
			self._sep()
			self._log("SKEWNESS FOR NUMERIC COLUMNS")
			self._log(SEP)
			skewness = self.data[numeric_cols].skew().sort_values(ascending=False)
			skewness_df = pd.DataFrame({'Column': skewness.index, 'Skewness': skewness.values})
			for line in skewness_df.to_string(index=False).splitlines():
				self._log(line)

			self._log("\nInterpretation:")
			self._log("  - Highly skewed: |skewness| > 1")
			self._log("  - Moderately skewed: 0.5 < |skewness| <= 1")
			self._log("  - Fairly symmetric: |skewness| <= 0.5")

	def correlation_analysis(
		self,
		method: str = 'pearson',
		save_path: Optional[str] = None
	) -> None:
		"""
		Phân tích tương quan giữa các cột số và vẽ heatmap
		
		Tính toán ma trận tương quan giữa tất cả các cặp cột số,
		sau đó trực quan hóa bằng heatmap với các giá trị tương quan được hiển thị.

		Parameters
		----------
		method : str, optional
			Phương pháp tính tương quan.
			Các giá trị hợp lệ: 'pearson', 'spearman', 'kendall'.
			Mặc định là 'pearson'
		save_path : str, optional
			Đường dẫn thư mục để lưu biểu đồ. Nếu None, chỉ hiển thị.
			Mặc định là None

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		
		Notes
		-----
		- Chỉ các cột có kiểu dữ liệu số mới được tính tương quan
		- Pearson: Đo lường mối quan hệ tuyến tính (hệ số tương quan nằm trong [-1, 1])
		- Spearman: Đo lường mối quan hệ đơn điệu (không nhất thiết tuyến tính)
		- Kendall: Đo lường sự phù hợp thứ tự giữa hai biến
		"""
		if self.data is None:
			raise ValueError("Data not loaded. Call load_data() first.")
		self._log(f"Running correlation_analysis(method='{method}')")
		# Chỉ chọn các cột số để tính toán ma trận tương quan
		numeric_data = self.data.select_dtypes(include=np.number)

		if not numeric_data.empty:
			correlation_matrix = numeric_data.corr(method=method)

			# Vẽ heatmap cho ma trận tương quan sử dụng Seaborn
			plt.figure(figsize=(10, 8))
			sns.heatmap(correlation_matrix, 
						annot=True,  # Hiển thị giá trị tương quan
						fmt='.2f',   # Định dạng 2 chữ số thập phân
						cmap='coolwarm',  # Bảng màu
						center=0,    # Đặt trung tâm tại 0
						square=True,  # Các ô vuông
						linewidths=0.5,  # Đường viền giữa các ô
						cbar_kws={"shrink": 0.8})  # Thanh màu
			plt.title(f'Correlation Heatmap ({method.capitalize()})', fontsize=16, fontweight='bold')
			plt.tight_layout()
			
			# Lưu biểu đồ nếu có đường dẫn
			if save_path:
				os.makedirs(save_path, exist_ok=True)
				plt.savefig(f'{save_path}/correlation_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
				self._log(f"✓ Correlation heatmap saved to: {save_path}/correlation_heatmap.png")
			
			if self.show_plots:
				plt.show()
			else:
				plt.close()
		else:
			print("Không có cột số nào để tính toán tương quan.")

	def data_distribution(self, save_path: Optional[str] = None) -> None:
		"""
		Trực quan hóa phân phối của các cột số bằng Histogram và KDE
		
		Vẽ biểu đồ histogram kết hợp với đường cong ước lượng mật độ hạt nhân (KDE)
		cho tất cả các cột số trên cùng một figure với nhiều subplots.

		Parameters
		----------
		save_path : str, optional
			Đường dẫn thư mục để lưu biểu đồ. Nếu None, chỉ hiển thị.
			Mặc định là None

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		
		Notes
		-----
		- Sử dụng seaborn histplot với kde=True để tự động vẽ KDE
		- Histogram được chuẩn hóa để tổng diện tích bằng 1
		- KDE (Kernel Density Estimation) được tính tự động bởi seaborn
		- Sử dụng 30 bins cho histogram
		- Tất cả các subplots được vẽ trên cùng một figure
		"""
		if self.data is None:
			raise ValueError("Data not loaded. Call load_data() first.")
		self._log("Plotting data distributions (Histogram + KDE) for numeric columns...")
		
		# Lấy các cột số
		numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
		n_cols = len(numeric_cols)
		
		if n_cols == 0:
			self._log("No numeric columns found for distribution plot.")
			return
		
		# Tính số hàng và cột cho subplots
		n_rows = (n_cols + 2) // 3  # 3 subplots mỗi hàng
		n_cols_grid = min(n_cols, 3)
		
		# Tạo figure với subplots
		fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(15, 4.5 * n_rows))
		
		# Đảm bảo axes luôn là mảng 2D
		if n_cols == 1:
			axes = np.array([[axes]])
		elif n_rows == 1:
			axes = axes.reshape(1, -1)
		
		# Flatten axes để dễ iterate
		axes_flat = axes.flatten()
		
		# Vẽ từng subplot
		for idx, col in enumerate(numeric_cols):
			ax = axes_flat[idx]
			
			# Vẽ histogram + KDE bằng seaborn
			sns.histplot(data=self.data, x=col, kde=True, bins=30, 
						color='skyblue', edgecolor='black', ax=ax)
			
			ax.set_title(f'{col}', fontsize=11, fontweight='bold')
			ax.set_xlabel('Value', fontsize=9)
			ax.set_ylabel('Density', fontsize=9)
			ax.tick_params(axis='both', labelsize=8)
			ax.grid(axis='y', alpha=0.3, linestyle='--')
			ax.set_axisbelow(True)
		
		# Ẩn các subplot thừa
		for idx in range(n_cols, len(axes_flat)):
			axes_flat[idx].set_visible(False)
		
		# Tiêu đề chung
		fig.suptitle('Data Distribution (Histogram + KDE)', fontsize=14, fontweight='bold', y=0.995)
		plt.tight_layout(rect=[0, 0, 1, 0.99])
		
		# Lưu biểu đồ nếu có đường dẫn
		if save_path:
			os.makedirs(save_path, exist_ok=True)
			plt.savefig(f'{save_path}/distribution_all.png', dpi=300, bbox_inches='tight', facecolor='white')
			self._log(f"✓ Distribution plot saved to: {save_path}/distribution_all.png")
		
		if self.show_plots:
			plt.show()
		else:
			plt.close()
	
	def boxplot_analysis(self, save_path: Optional[str] = None) -> None:
		"""
		Trực quan hóa các boxplot để phát hiện ngoại lai (outliers)
		
		Vẽ biểu đồ boxplot cho tất cả các cột số trên cùng một figure với nhiều subplots
		để dễ dàng nhận diện các giá trị bất thường, phân vị và khoảng tứ phân vị (IQR).

		Parameters
		----------
		save_path : str, optional
			Đường dẫn thư mục để lưu biểu đồ. Nếu None, chỉ hiển thị.
			Mặc định là None

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		
		Notes
		-----
		- Boxplot hiển thị: min, Q1, median (Q2), Q3, max và outliers
		- Outliers được xác định bằng quy tắc IQR.
		- Giá trị NaN được tự động loại bỏ trước khi vẽ
		- Sử dụng seaborn boxplot để có giao diện đẹp hơn
		- Tất cả các subplots được vẽ trên cùng một figure
		"""
		if self.data is None:
			raise ValueError("Data not loaded. Call load_data() first.")
		self._log("Boxplot analysis for numeric columns...")
		
		# Lấy các cột số
		numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
		n_cols = len(numeric_cols)
		
		if n_cols == 0:
			self._log("No numeric columns found for boxplot.")
			return
		
		# Tính số hàng và cột cho subplots
		n_rows = (n_cols + 2) // 3  # 3 subplots mỗi hàng
		n_cols_grid = min(n_cols, 3)
		
		# Tạo figure với subplots
		fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(15, 4 * n_rows))
		
		# Đảm bảo axes luôn là mảng 2D
		if n_cols == 1:
			axes = np.array([[axes]])
		elif n_rows == 1:
			axes = axes.reshape(1, -1)
		
		# Flatten axes để dễ iterate
		axes_flat = axes.flatten()
		
		# Vẽ từng subplot
		for idx, col in enumerate(numeric_cols):
			ax = axes_flat[idx]
			
			# Vẽ boxplot bằng seaborn
			sns.boxplot(y=self.data[col], color='skyblue', width=0.5, ax=ax)
			
			ax.set_title(f'{col}', fontsize=11, fontweight='bold')
			ax.set_ylabel('Value', fontsize=9)
			ax.tick_params(axis='both', labelsize=8)
			ax.grid(axis='y', alpha=0.3, linestyle='--')
			ax.set_axisbelow(True)
		
		# Ẩn các subplot thừa
		for idx in range(n_cols, len(axes_flat)):
			axes_flat[idx].set_visible(False)
		
		# Tiêu đề chung
		fig.suptitle('Boxplot Analysis for Outlier Detection', fontsize=14, fontweight='bold', y=0.995)
		plt.tight_layout(rect=[0, 0, 1, 0.99])
		
		# Lưu biểu đồ nếu có đường dẫn
		if save_path:
			os.makedirs(save_path, exist_ok=True)
			plt.savefig(f'{save_path}/boxplot_all.png', dpi=300, bbox_inches='tight', facecolor='white')
			self._log(f"✓ Boxplot saved to: {save_path}/boxplot_all.png")
		
		if self.show_plots:
			plt.show()
		else:
			plt.close()

	def perform_eda(
		self,
		corr_method: str = 'pearson',
		save_path: str = 'EDA'
	) -> None:
		"""
		Thực hiện toàn bộ quy trình EDA cho các cột số
		
		Chạy tuần tự tất cả các phương thức phân tích dữ liệu khám phá:
		thống kê mô tả, phân tích tương quan, phân phối dữ liệu và phát hiện ngoại lai.

		Parameters
		----------
		corr_method : str, optional
			Phương pháp tính tương quan cho correlation_analysis.
			Các giá trị hợp lệ: 'pearson', 'spearman', 'kendall'.
			Mặc định là 'pearson'
		save_path : str, optional
			Đường dẫn thư mục để lưu tất cả biểu đồ. Nếu None, chỉ hiển thị.
			Mặc định là 'EDA'

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		
		Notes
		-----
		Thứ tự thực hiện:
		1. summary_statistics() - Thống kê mô tả
		2. correlation_analysis() - Ma trận tương quan
		3. data_distribution() - Histogram phân phối
		4. boxplot_analysis() - Boxplot phát hiện ngoại lai
		"""
		if self.data is None:
			raise ValueError("Data not loaded. Call load_data() first.")
		self._log("Starting full EDA pipeline...")
		self.summary_statistics(save_path=save_path)                    # Thống kê mô tả
		self.correlation_analysis(method=corr_method, save_path=save_path)  # Ma trận tương quan
		self.data_distribution(save_path=save_path)                     # Histogram phân phối
		self.boxplot_analysis(save_path=save_path)                      # Boxplot phát hiện ngoại lai
		self._log("EDA pipeline completed.")