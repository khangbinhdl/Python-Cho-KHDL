# Khai báo thư viện cần thiết
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import logging

# Thiết lập style cho seaborn
sns.set_theme(style="whitegrid")

# Logger riêng cho EDA
LOGGER = logging.getLogger("EDA")
if not LOGGER.handlers:
	LOGGER.propagate = True
	LOGGER.setLevel(logging.INFO)

# Độ dài “vừa vừa” cho separator (giảm từ 80 -> 60, bạn có thể chỉnh tùy thích)
SEP_LEN = 60
SEP = "=" * SEP_LEN

class EDAData:
	"""
	Class thực hiện phân tích dữ liệu khám phá (Exploratory Data Analysis)
	
	Cung cấp các phương thức để phân tích thống kê mô tả, tương quan,
	phân phối dữ liệu và phát hiện ngoại lai thông qua trực quan hóa.

	Attributes
	----------
	data : DataFrame
		Dữ liệu cần phân tích
	"""
	
	def __init__(self, data):
		"""
		Khởi tạo đối tượng EDA với dữ liệu
		
		Parameters
		----------
		data : DataFrame
			DataFrame chứa dữ liệu cần phân tích
		"""
		self.data = data
		
	def _log(self, message):
		"""
		Ghi log thông điệp với logger của EDA
		
		Parameters
		----------
		message : str
			Thông điệp cần ghi log
		"""
		LOGGER.info(message)

	def _sep(self):
		"""
		In ra một dòng phân cách trong log
		
		Ghi một dòng trống và một dòng dấu "=" để phân tách các phần trong log,
		giúp dễ đọc hơn khi xem output.
		"""
		LOGGER.info("")      # một dòng trống có timestamp
		LOGGER.info(SEP)     # separator có timestamp

	def summary_statistics(self):
		"""
		In ra các thống kê mô tả cơ bản của các cột số
		
		Hiển thị các chỉ số thống kê như count, mean, std, min, max, 
		và các phân vị (25%, 50%, 75%) cho tất cả các cột số trong DataFrame.
		Bổ sung thông tin về kiểu dữ liệu, giá trị thiếu, trùng lặp, value counts và độ lệch.

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
		# describe có thể dài — ghi từng dòng để tránh vượt giới hạn message
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
			plt.show()
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

			
			# Vẽ biểu đồ skewness ngang
			plt.figure(figsize=(8, max(5, len(numeric_cols) * 0.35)))
			
			# Sắp xếp theo giá trị tuyệt đối của skewness (lớn nhất ở trên)
			skewness_sorted = skewness.sort_values(key=abs, ascending=True)
			
			# Tạo màu sắc gradient dựa trên giá trị skewness
			# Normalize skewness values để map vào colormap
			norm = plt.Normalize(vmin=skewness_sorted.min(), vmax=skewness_sorted.max())
			colors = plt.cm.RdYlGn_r(norm(skewness_sorted.values))
			
			# Vẽ barh (horizontal bar)
			bars = plt.barh(skewness_sorted.index, skewness_sorted.values, color=colors, edgecolor='black')
			
			# Thêm đường tham chiếu dọc
			plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
			plt.axvline(x=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
			plt.axvline(x=-1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
			plt.axvline(x=0.5, color='lightgray', linestyle=':', linewidth=0.8, alpha=0.7)
			plt.axvline(x=-0.5, color='lightgray', linestyle=':', linewidth=0.8, alpha=0.7)
			
			plt.title('Skewness by Column', fontsize=14, fontweight='bold')
			plt.xlabel('Skewness', fontsize=12)
			plt.ylabel('Columns', fontsize=12)
			plt.tight_layout()
			plt.show()

	def correlation_analysis(self, method='pearson'):
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
		- Sử dụng seaborn heatmap với colormap 'coolwarm' để dễ phân biệt tương quan dương/âm
		- Giá trị tương quan được làm tròn đến 2 chữ số thập phân
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
			plt.show()
		else:
			print("Không có cột số nào để tính toán tương quan.")

	def data_distribution(self):
		"""
		Trực quan hóa phân phối của các cột số bằng Histogram và KDE
		
		Vẽ biểu đồ histogram kết hợp với đường cong ước lượng mật độ hạt nhân (KDE)
		cho từng cột số để hiểu rõ hơn về phân phối dữ liệu.

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
		"""
		if self.data is None:
			raise ValueError("Data not loaded. Call load_data() first.")
		self._log("Plotting data distributions (Histogram + KDE) for numeric columns...")
		print("\nData Distribution for Numeric Columns (Histogram + KDE):")

		for col in self.data.select_dtypes(include=np.number).columns:
			plt.figure(figsize=(8, 5))
			
			# Vẽ histogram + KDE bằng seaborn
			sns.histplot(data=self.data, x=col, kde=True, bins=30, 
						color='skyblue', edgecolor='black')
			
			# Lấy line (KDE curve) cuối cùng và đổi màu thành đỏ
			ax = plt.gca()
			lines = ax.get_lines()
			if lines:  # Nếu có đường KDE
				lines[-1].set_color('red')
				lines[-1].set_linewidth(2)
			
			plt.title(f'Distribution of {col} (with KDE)', fontsize=14, fontweight='bold')
			plt.xlabel('Value', fontsize=12)
			plt.ylabel('Density', fontsize=12)
			plt.tight_layout()
			plt.show()
	
	def boxplot_analysis(self):
		"""
		Trực quan hóa các boxplot để phát hiện ngoại lai (outliers)
		
		Vẽ biểu đồ boxplot cho từng cột số để dễ dàng nhận diện các giá trị
		bất thường, phân vị và khoảng tứ phân vị (IQR) của dữ liệu.

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		
		Notes
		-----
		- Boxplot hiển thị: min, Q1, median (Q2), Q3, max và outliers
		- Outliers được xác định bằng quy tắc IQR: Q1 - 1.5*IQR và Q3 + 1.5*IQR
		- Giá trị NaN được tự động loại bỏ trước khi vẽ
		- Sử dụng seaborn boxplot để có giao diện đẹp hơn
		"""
		print("\nBoxplot Analysis for Numeric Columns:")
		if self.data is None:
			raise ValueError("Data not loaded. Call load_data() first.")
		self._log("Boxplot analysis for numeric columns...")
		for col in self.data.select_dtypes(include=np.number).columns:
			plt.figure(figsize=(8, 5))
			
			# Vẽ boxplot bằng seaborn
			sns.boxplot(y=self.data[col], color='skyblue', width=0.5)
			
			plt.title(f'Boxplot of {col}', fontsize=14, fontweight='bold')
			plt.ylabel('Value', fontsize=12)
			plt.tight_layout()
			plt.show()

	def perform_eda(self, corr_method='pearson'):
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
		self.summary_statistics()                    # Thống kê mô tả
		self.correlation_analysis(method=corr_method)  # Ma trận tương quan
		self.data_distribution()                     # Histogram phân phối
		self.boxplot_analysis()                      # Boxplot phát hiện ngoại lai
		self._log("EDA pipeline completed.")