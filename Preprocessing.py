import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import logging

# THAY cho logging.basicConfig(...): tạo logger riêng cho Preprocessor
LOGGER = logging.getLogger("PREPROCESSOR")
if not LOGGER.handlers:
	# để việc cấu hình handler/formatter được làm ở main.py (tránh trùng handler)
	LOGGER.propagate = True
	LOGGER.setLevel(logging.INFO)

class DataPreprocessor:
	"""
	Class đóng gói các bước tiền xử lý dữ liệu
	
	Bao gồm các chức năng: nạp dữ liệu, xử lý kiểu dữ liệu, xử lý giá trị thiếu, 
	loại bỏ ngoại lai, mã hóa, chuẩn hóa và lưu kết quả ra file.

	Attributes
	----------
	data : DataFrame or None
		Dữ liệu được nạp vào và xử lý
	numeric_cols : list
		Danh sách các cột có kiểu dữ liệu số
	categorical_cols : list
		Danh sách các cột có kiểu dữ liệu phân loại
	datetime_cols : list
		Danh sách các cột có kiểu dữ liệu ngày giờ
	missing_strategy : str
		Phương pháp xử lý giá trị bị thiếu
	scaling_strategy : str
		Phương pháp chuẩn hóa dữ liệu
	outlier_method : str
		Phương pháp phát hiện ngoại lai
	scaler : object or None
		Đối tượng scaler được fit với dữ liệu
	encoders : dict
		Dictionary lưu trữ các encoder cho từng cột
	"""

	def __init__(self, missing_strategy='mean', scaling_strategy='standard', outlier_method='iqr'):
		"""
		Khởi tạo đối tượng DataPreprocessor với các chiến lược xử lý

		Parameters
		----------
		missing_strategy : str, optional
			Phương pháp xử lý giá trị bị thiếu. 
			Các giá trị hợp lệ: 'mean', 'median', 'mode', 'ffill', 'drop'.
			Mặc định là 'mean'
		scaling_strategy : str, optional
			Phương pháp chuẩn hóa dữ liệu.
			Các giá trị hợp lệ: 'standard', 'minmax'.
			Mặc định là 'standard'
		outlier_method : str, optional
			Phương pháp phát hiện ngoại lai.
			Các giá trị hợp lệ: 'iqr', 'zscore', 'isolation_forest'.
			Mặc định là 'iqr'
		"""
		self.data = None
		self.numeric_cols = []
		self.categorical_cols = []
		self.datetime_cols = []

		# Thiết lập các chiến lược xử lý
		self.missing_strategy = missing_strategy
		self.scaling_strategy = scaling_strategy
		self.outlier_method = outlier_method

		# Khởi tạo nơi lưu trữ các đối tượng 'fit'
		self.scaler = None
		self.encoders = {}

	def __repr__(self):
		"""
		Định nghĩa cách đối tượng được in ra
		
		Returns
		-------
		str
			Chuỗi hiển thị các chiến lược đã chọn
		"""
		return (f"DataPreprocessor(missing='{self.missing_strategy}', "
				f"scaling='{self.scaling_strategy}', "
				f"outlier='{self.outlier_method}')")

	@staticmethod
	def _log(message):
		"""
		Hàm tiện ích static để gọi logging.info một cách nhất quán
		
		Parameters
		----------
		message : str
			Thông điệp cần ghi log
		"""
		LOGGER.info(message)
	
	def _clean_column_names(self):
		"""
		Chuẩn hóa tên cột của DataFrame
		
		Thực hiện các bước làm sạch tên cột: loại bỏ ký tự xuống dòng, khoảng trắng thừa,
		chuyển về snake_case, loại bỏ ký tự đặc biệt và xử lý trùng lặp tên cột.

		Notes
		-----
		Các bước xử lý:
		1. Loại bỏ ký tự xuống dòng (\\n)
		2. Loại bỏ khoảng trắng đầu/cuối
		3. Chuẩn hóa nhiều khoảng trắng thành một khoảng trắng
		4. Chuyển về chữ thường
		5. Thay khoảng trắng bằng dấu gạch dưới (snake_case)
		6. Chuẩn hóa Unicode (NFKD)
		7. Thay ký tự đặc biệt bằng dấu gạch dưới
		8. Gộp nhiều dấu gạch dưới liên tiếp thành một
		9. Loại bỏ dấu gạch dưới ở đầu/cuối
		10. Xử lý trùng lặp bằng cách thêm suffix số
		"""
		self._log("Cleaning column names...")
		import re
		cols = (
			pd.Series(self.data.columns, dtype="string")
				.str.replace('\n', ' ', regex=False)      # bỏ newline
				.str.strip()                               # bỏ khoảng trắng đầu/đuôi
				.str.replace(r'\s+', ' ', regex=True)      # 1 khoảng trắng
				.str.lower()
				.str.replace(' ', '_', regex=False)        # snake_case
				.str.normalize('NFKD')
				.str.replace(r'[^\w]+', '_', regex=True)   # ký tự lạ -> _
				.str.replace(r'_+', '_', regex=True)       # gộp nhiều _
				.str.strip('_')                            # bỏ _ đầu/đuôi
		)

		# Chống trùng tên cột
		seen = {}
		def dedup(name):
			n = name if name != '' else 'col'
			seen[n] = seen.get(n, 0) + 1
			return n if seen[n] == 1 else f"{n}_{seen[n]-1}"

		self.data.columns = [dedup(c) for c in cols]
		self._log("Column names cleaned successfully.")
	
	def convert_columns_to_numeric(self, start_col=2):
		"""
		Chuyển đổi các cột từ vị trí chỉ định trở đi trong DataFrame thành kiểu số
		
		Phương thức này sẽ loại bỏ dấu phẩy, khoảng trắng và chuyển đổi sang kiểu số.
		Đây là đặc trưng của bộ dữ liệu nhóm đã chọn.

		Parameters
		----------
		start_col : int, optional
			Chỉ số cột bắt đầu chuyển đổi (mặc định là 2, tương ứng với cột thứ 3).
			Mặc định là 2

		Returns
		-------
		self
			Trả về chính đối tượng để có thể chain methods

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		"""
		self._log(f"Converting columns starting from column {start_col+1} to numeric...")
		if self.data is None:
			raise ValueError("Data not loaded. Call load_data() first.")

		# Chọn các cột từ cột thứ 3 trở đi (do đây là đặc trưng của bộ dữ liệu nhóm đã chọn )
		for col in self.data.columns[start_col:]:
			# Kiểm tra xem cột có kiểu dữ liệu 'object'
			if self.data[col].dtype == 'object':
				# Loại bỏ dấu phẩy và khoảng trắng trước khi chuyển đổi
				self.data[col] = self.data[col].str.replace(',', '').str.strip()
				# Chuyển đổi thành kiểu số (float hoặc int)
				self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

		# Kiểm tra lại kiểu dữ liệu sau khi chuyển đổi
		self._log(f"Data types after conversion:\n{self.data.dtypes}")
		return self

	def load_data(self, filepath):
		"""
		Nạp dữ liệu từ các file (CSV, XLSX, JSON) vào DataFrame
		
		Tự động chuẩn hóa tên cột, chuyển đổi kiểu dữ liệu số, loại bỏ trùng lặp
		và phân loại các cột sau khi nạp dữ liệu.

		Parameters
		----------
		filepath : str
			Đường dẫn tới file dữ liệu. Hỗ trợ các định dạng: .csv, .xlsx, .xls, .json

		Returns
		-------
		self
			Trả về chính đối tượng để có thể chain methods

		Raises
		------
		ValueError
			Nếu định dạng file không được hỗ trợ
		FileNotFoundError
			Nếu không tìm thấy file tại đường dẫn được cung cấp
		Exception
			Các lỗi khác trong quá trình đọc dữ liệu
		
		Notes
		-----
		Các bước xử lý tự động sau khi nạp:
		1. Chuẩn hóa tên cột (_clean_column_names)
		2. Chuyển đổi cột từ vị trí thứ 3 sang kiểu số
		3. Loại bỏ các dòng trùng lặp
		4. Tự động phân loại cột (numeric, categorical, datetime)
		"""
		self._log(f"Loading data from {filepath}...")

		try:
			if filepath.endswith('.csv'):
				self.data = pd.read_csv(filepath)  # Đọc dữ liệu từ file CSV
			elif filepath.endswith(('.xlsx', '.xls')):
				self.data = pd.read_excel(filepath)  # Đọc dữ liệu từ file Excel
			elif filepath.endswith('.json'):
				self.data = pd.read_json(filepath)  # Đọc dữ liệu từ file JSON
			else:
				# Lỗi nếu định dạng tệp không được hỗ trợ
				raise ValueError("Unsupported file format. Please use .csv, .xlsx, or .json.")
		except FileNotFoundError:
			self._log(f"Error: File not found at {filepath}")  # Báo lỗi nếu không tìm thấy file
			raise
		except Exception as e:
			self._log(f"Error loading data: {e}")  # Báo lỗi khi đọc dữ liệu
			raise
		# Chuẩn hóa tên cột sau khi nạp dữ liệu
		self._clean_column_names()
		# Chuyển đổi các cột từ cột thứ 3 trở đi thành kiểu số
		self.convert_columns_to_numeric(start_col=2)
		# Loại bỏ các cột trùng lặp
		self.data = self.data.drop_duplicates()
		# Tự động phân loại các cột ngay sau khi nạp
		self.auto_detect_columns()
		return self

	def auto_detect_columns(self):
		"""
		Tự động phát hiện và phân loại các cột theo kiểu dữ liệu
		
		Phương thức này sẽ tự động nhận diện cột số, cột phân loại và cột ngày giờ 
		trong DataFrame, sau đó lưu vào các thuộc tính tương ứng.

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp

		Notes
		-----
		Phân loại cột:
		- Cột số: Các cột có kiểu dữ liệu numeric (int, float)
		- Cột phân loại: Các cột có kiểu dữ liệu object hoặc category
		- Cột datetime: Các cột có kiểu datetime hoặc có thể chuyển đổi sang datetime
		
		Quy tắc chuyển đổi datetime:
		- Cột được coi là datetime nếu có ít nhất 1 giá trị hợp lệ sau khi parse
		- Định dạng mặc định: '%Y-%m-%d'
		- Cột không parse được sẽ giữ nguyên là categorical
		"""
		if self.data is None:
			raise ValueError("Data not loaded. Call load_data() first.")

		self._log("Auto-detecting column types...")
		self.numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
		self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

		self.datetime_cols = [] #Khởi tạo lại danh sách datetime

		# Phát hiện các cột ngày giờ (datetime)
		for col in self.data.columns:
			if pd.api.types.is_datetime64_any_dtype(self.data[col]):
				self.datetime_cols.append(col)
			elif col in self.categorical_cols:
				try:
					# chuyển đổi sang datetime
					converted_col = pd.to_datetime(self.data[col], format='%Y-%m-%d', errors='coerce')

					# Chỉ chấp nhận là datetime nếu CÓ ÍT NHẤT 1 GIÁ TRỊ HỢP LỆ
					if converted_col.notna().any():
						self._log(f"Column '{col}' detected as datetime and converted.")
						self.data[col] = converted_col # LƯU LẠI KẾT QUẢ
						self.datetime_cols.append(col)
					# Nếu không có giá trị nào hợp lệ (ví dụ: cột 'Company'/'Item' bị ép thành NaT),
					# nó sẽ KHÔNG được thêm vào datetime_cols và vẫn là categorical.

				except (ValueError, TypeError):
					pass # Bỏ qua nếu cột gây lỗi nghiêm trọng khi cố chuyển đổi

		self.datetime_cols = list(set(self.datetime_cols))

		# CẬP NHẬT LẠI danh sách sau khi đã chuyển đổi
		self.numeric_cols = [c for c in self.numeric_cols if c not in self.datetime_cols]
		self.categorical_cols = [c for c in self.categorical_cols if c not in self.datetime_cols]

		self._log(f"Numeric cols: {self.numeric_cols}")
		self._log(f"Categorical cols: {self.categorical_cols}")
		self._log(f"Datetime cols: {self.datetime_cols}")

	def clean_negative_values(self):
		"""
		Thay thế tất cả giá trị âm trong các cột số thành giá trị dương
		
		Sử dụng hàm abs() để chuyển đổi các giá trị âm thành giá trị tuyệt đối.

		Returns
		-------
		self
			Trả về chính đối tượng để có thể chain methods

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		"""
		if self.data is None:
			raise ValueError("Data not loaded.")

		self._log("Cleaning negative values in all numeric columns...")

		# Lấy tất cả các cột số
		numeric_columns = self.data.select_dtypes(include=[np.number]).columns

		# Dùng numpy để thay thế các giá trị âm bằng giá trị tuyệt đối cho tất cả các cột số
		for col in numeric_columns:
			self.data[col] = np.abs(self.data[col])

		return self
	
	def handle_missing_values(self, data=None, num_strategy=None, cat_strategy=None, dt_strategy="drop", exclude_features=None, fit=False):
		"""
		Xử lý giá trị bị thiếu với phương pháp riêng cho từng loại cột
		
		Áp dụng các chiến lược khác nhau cho cột số, cột phân loại và cột datetime.
		Nếu không truyền tham số, sẽ sử dụng giá trị mặc định từ self.missing_strategy.

		Parameters
		----------
		data : DataFrame or None
			DataFrame cần xử lý. 
			- None: dùng self.data như cũ.
			- Khác None: xử lý trực tiếp trên DataFrame truyền vào và trả về DataFrame đó.
		num_strategy : str, optional
			Chiến lược xử lý cho cột số.
			Các giá trị hợp lệ: 'mean', 'median', 'mode', 'ffill', 'bfill', 'drop'.
			Mặc định lấy từ self.missing_strategy
		cat_strategy : str, optional
			Chiến lược xử lý cho cột phân loại.
			Các giá trị hợp lệ: 'mode', 'ffill', 'bfill', 'drop', 'constant'.
			Mặc định lấy từ self.missing_strategy
		dt_strategy : str, optional
			Chiến lược xử lý cho cột datetime.
			Các giá trị hợp lệ: 'ffill', 'bfill', 'drop'.
			Mặc định là 'drop'
		exclude_features : list of str, optional
			Danh sách tên các cột muốn bỏ qua không xử lý missing values.
			Mặc định là None (xử lý tất cả các cột)
		fit : bool, optional
			- True : fit và tính toán tham số dựa trên 'data' (thường là train).
			- False: chỉ apply tham số đã fit (thường là test/val).

		Returns
		-------
		self
			Trả về chính đối tượng để có thể chain methods

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		"""

		# Chọn target DataFrame
		if data is None:
			if self.data is None:
				raise ValueError("Data not loaded.")
			target = self.data
		else:
			target = data

		if exclude_features is None:
			exclude_features = []

		# Mặc định kế thừa self.missing_strategy nếu không được truyền riêng
		num_strategy = num_strategy or self.missing_strategy
		cat_strategy = cat_strategy or self.missing_strategy

		self._log(
			f"Handling missing values fit={fit} | num: '{num_strategy}', cat: '{cat_strategy}', dt: '{dt_strategy}'"
		)
		# Lưu lại chiến lược datetime (để biết test nên xử lý kiểu gì, mặc dù dt không cần thống kê)
		if fit:
			self.missing_dt_strategy = dt_strategy

		# ======== FIT THAM SỐ TRÊN TRAIN (fit=True) ======== #
		if fit:
			# Numeric
			self.missing_num_values = {}
			if self.numeric_cols and num_strategy in ("mean", "median", "mode"):
				for col in self.numeric_cols:
					if col in exclude_features:
						continue
					if target[col].isna().any():
						if num_strategy == "mean":
							val = target[col].mean()
						elif num_strategy == "median":
							val = target[col].median()
						else:
							mode = target[col].mode()
							val = mode.iloc[0] if not mode.empty else target[col].median()
						self.missing_num_values[col] = val

			# Categorical
			self.missing_cat_values = {}
			if self.categorical_cols and cat_strategy in ("mode", "constant"):
				for col in self.categorical_cols:
					if col in exclude_features:
						continue
					if target[col].isna().any():
						if cat_strategy == "mode":
							mode = target[col].mode()
							val = mode.iloc[0] if not mode.empty else "Unknown"
						else:  # constant
							val = "Unknown"
						self.missing_cat_values[col] = val

		# ======== ÁP DỤNG THAM SỐ ======== #

		# Datetime
		if self.datetime_cols:
			use_dt_strategy = dt_strategy if fit else (self.missing_dt_strategy or dt_strategy)
			if use_dt_strategy == "drop":
				target = target.dropna(subset=self.datetime_cols)
			elif use_dt_strategy == "ffill":
				for col in self.datetime_cols:
					target[col] = target[col].ffill()
			elif use_dt_strategy == "bfill":
				for col in self.datetime_cols:
					target[col] = target[col].bfill()

		# Numeric
		if self.numeric_cols:
			numeric_cols_to_process = [c for c in self.numeric_cols if c not in exclude_features]
			if num_strategy == "drop":
				if numeric_cols_to_process:
					target = target.dropna(subset=numeric_cols_to_process)
			elif num_strategy in ("mean", "median", "mode"):
				for col in numeric_cols_to_process:
					# Lấy giá trị đã fit nếu có
					val = self.missing_num_values.get(col, None)
					if val is not None:
						target[col] = target[col].fillna(val)
			elif num_strategy == "ffill":
				if numeric_cols_to_process:
					for col in numeric_cols_to_process:
						target[col] = target[col].ffill()
			elif num_strategy == "bfill":
				if numeric_cols_to_process:
					for col in numeric_cols_to_process:
						target[col] = target[col].bfill()

		# Categorical
		if self.categorical_cols:
			categorical_cols_to_process = [c for c in self.categorical_cols if c not in exclude_features]
			if cat_strategy == "drop":
				if categorical_cols_to_process:
					target = target.dropna(subset=categorical_cols_to_process)
			elif cat_strategy == "mode":
				for col in categorical_cols_to_process:
					val = self.missing_cat_values.get(col, None)
					if val is None:
						mode = target[col].mode()
						val = mode.iloc[0] if not mode.empty else "Unknown"
					target[col] = target[col].fillna(val)
			elif cat_strategy == "ffill":
				if categorical_cols_to_process:
					for col in categorical_cols_to_process:
						target[col] = target[col].ffill()
			elif cat_strategy == "bfill":
				if categorical_cols_to_process:
					for col in categorical_cols_to_process:
						target[col] = target[col].bfill()
			elif cat_strategy == "constant":
				for col in categorical_cols_to_process:
					val = self.missing_cat_values.get(col, "Unknown")
					target[col] = target[col].fillna(val)		# Cập nhật lại phân loại cột chỉ khi làm việc với self.data
		if data is None:
			self.auto_detect_columns()
			return self
		else:
			return target
	def handle_outliers(self, data=None, exclude_features=None, outlier_strategy='drop'):
		"""
		Phát hiện và xử lý các ngoại lai (outliers) trong dữ liệu
		
		Sử dụng một trong ba phương pháp: IQR, Z-score hoặc Isolation Forest
		để phát hiện outliers, sau đó loại bỏ hàng (drop) hoặc cắt giá trị (clip).
		
		LƯU Ý: Chỉ nên áp dụng trên tập TRAIN, KHÔNG nên xử lý outlier trên tập test
		để tránh data leakage và đảm bảo đánh giá mô hình chính xác.

		Parameters
		----------
		data : DataFrame or None
			DataFrame cần xử lý outlier. 
			None -> dùng self.data như cũ.
		exclude_features : list of str, optional
			Danh sách tên các cột muốn bỏ qua không xử lý ngoại lai.
			Mặc định là None (xử lý tất cả các cột số)
		outlier_strategy : str, optional
			Chiến lược xử lý outliers.
			- 'drop': Xóa các hàng chứa outliers
			- 'clip': Cắt giá trị outliers về ngưỡng min/max (giữ lại tất cả dữ liệu)
			Mặc định là 'drop'
			
		Returns
		-------
		self hoặc DataFrame
			Nếu data=None: trả về self
			Nếu data được cung cấp: trả về DataFrame đã xử lý

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		NotImplementedError
			Nếu chọn chiến lược 'clip' với phương pháp Isolation Forest
	
		Notes
		-----
		Phương pháp phát hiện:
		- IQR: Phát hiện giá trị nằm ngoài [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
		- Z-score: Phát hiện giá trị nằm ngoài khoảng ±3σ
		- Isolation Forest: Sử dụng thuật toán ML (contamination=0.1)
		
		Chiến lược xử lý:
		- drop: Xóa hàng chứa outliers (giảm số lượng dữ liệu)
		- clip: Cắt giá trị về ngưỡng (giữ nguyên số lượng dữ liệu, thay đổi giá trị)
		"""
		if data is None:
			if self.data is None:
				raise ValueError("Data not loaded.")
			target = self.data
		else:
			target = data

		if exclude_features is None:
			exclude_features = []

		self._log(f"[Outlier] method='{self.outlier_method}', strategy='{outlier_strategy}'")
		initial_rows = len(target)
		clipped_values_count = 0

		# ====== IQR & Z-SCORE ====== #
		if self.outlier_method in ('iqr', 'zscore'):
			if outlier_strategy == 'drop':
				mask = pd.Series(True, index=target.index)
				for col in self.numeric_cols:
					if col in exclude_features:
						continue
					# Tính ngưỡng trực tiếp từ dữ liệu
					if self.outlier_method == 'iqr':
						Q1 = target[col].quantile(0.25)
						Q3 = target[col].quantile(0.75)
						IQR = Q3 - Q1
						lower_bound = Q1 - 1.5 * IQR
						upper_bound = Q3 + 1.5 * IQR
					else:  # zscore
						mean = target[col].mean()
						std = target[col].std()
						lower_bound = mean - 3 * std
						upper_bound = mean + 3 * std

					mask &= (target[col] >= lower_bound) & (target[col] <= upper_bound)

				target = target[mask]

			elif outlier_strategy == 'clip':
				for col in self.numeric_cols:
					if col in exclude_features:
						continue
					# Tính ngưỡng trực tiếp từ dữ liệu
					if self.outlier_method == 'iqr':
						Q1 = target[col].quantile(0.25)
						Q3 = target[col].quantile(0.75)
						IQR = Q3 - Q1
						lower_bound = Q1 - 1.5 * IQR
						upper_bound = Q3 + 1.5 * IQR
					else:  # zscore
						mean = target[col].mean()
						std = target[col].std()
						lower_bound = mean - 3 * std
						upper_bound = mean + 3 * std

					clipped_values_count += ((target[col] < lower_bound) | (target[col] > upper_bound)).sum()
					target[col] = target[col].clip(lower=lower_bound, upper=upper_bound)

		# ====== ISOLATION FOREST ====== #
		elif self.outlier_method == 'isolation_forest':
			numeric_cols_for_isolation = [col for col in self.numeric_cols if col not in exclude_features]

			if not numeric_cols_for_isolation:
				self._log("Isolation Forest: No numeric columns to process after excluding specified features.")
			else:
				# Tạo model mới và fit trực tiếp (không lưu lại)
				iso_model = IsolationForest(contamination=0.1, random_state=42)
				yhat = iso_model.fit_predict(target[numeric_cols_for_isolation])

				if outlier_strategy == 'drop':
					target = target[yhat == 1]
				elif outlier_strategy == 'clip':
					raise NotImplementedError("Clipping not implemented for Isolation Forest method.")

		# ====== LOG KẾT QUẢ ====== #
		if outlier_strategy == 'drop':
			rows_removed = initial_rows - len(target)
			self._log(f"Removed {rows_removed} rows ({rows_removed / initial_rows * 100:.2f}% of data) as outliers")
			target = target.reset_index(drop=True)
		elif outlier_strategy == 'clip':
			self._log(f"Clipped {clipped_values_count} outlier values to bounds (no rows removed)")

		# nếu đang xử lý self.data thì update self.data, ngược lại trả về DataFrame
		if data is None:
			self.data = target
			return self
		else:
			return target
		
	def encode_categorical(self, strategy='onehot'):
		"""
		Mã hóa các cột phân loại (categorical) thành dạng số
		
		Chuyển đổi các cột phân loại (string) thành số bằng Label Encoding 
		hoặc One-Hot Encoding để có thể sử dụng trong các thuật toán machine learning.

		Parameters
		----------
		strategy : str, optional
			Phương pháp mã hóa.
			- 'label': Chuyển đổi thành số thứ tự (0, 1, 2, ...)
			- 'onehot': Tạo các cột dummy với giá trị 0/1
			Mặc định là 'onehot'

		Returns
		-------
		self
			Trả về chính đối tượng để có thể chain methods

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp

		Notes
		-----
		- Label Encoding: Phù hợp cho dữ liệu có thứ tự (ordinal)
		- One-Hot Encoding: Phù hợp cho dữ liệu không có thứ tự (nominal)
		- Sau khi One-Hot Encoding, số lượng cột sẽ tăng lên
		"""
		if self.data is None:
			raise ValueError("Data not loaded.")

		self._log(f"Encoding categorical features using: '{strategy}'")

		if strategy == 'label':
			# Label Encoding: Chuyển 'A', 'B', 'C' thành 0, 1, 2
			for col in self.categorical_cols:
				le = LabelEncoder()  # Sử dụng LabelEncoder để mã hóa giá trị phân loại thành số
				self.data[col] = le.fit_transform(self.data[col])
				self.encoders[col] = le  # Lưu lại encoder để sử dụng sau
		elif strategy == 'onehot':
			# One-Hot Encoding: Tạo các cột dummy 0/1
			self.data = pd.get_dummies(self.data, columns=self.categorical_cols, drop_first=True)  # One-hot encoding
			# Cần chạy lại auto_detect vì tên cột và số lượng cột đã thay đổi
			self.auto_detect_columns()

		return self

	def scale_features(self, data=None, exclude_features=None, fit=False):
		"""
		Chuẩn hóa các giá trị trong các cột số (numeric columns)
		
		Áp dụng StandardScaler hoặc MinMaxScaler để đưa các giá trị số
		về cùng một thang đo, giúp cải thiện hiệu suất của các thuật toán ML.

		Parameters
		----------
		data : DataFrame or None
			DataFrame cần scale. 
			None -> dùng self.data như cũ.
		exclude_features : list of str, optional
			Danh sách tên các cột muốn bỏ qua không scale.
			Mặc định là None (scale tất cả các cột số)
		fit : bool
			True: fit scaler trên dữ liệu này.
			False: chỉ transform bằng scaler đã fit.

		Returns
		-------
		self
			Trả về chính đối tượng để có thể chain methods

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp

		Notes
		-----
		- StandardScaler: Chuẩn hóa về phân phối chuẩn (mean=0, std=1)
		- MinMaxScaler: Chuẩn hóa về khoảng [0, 1]
		"""
		if data is None:
			if self.data is None:
				raise ValueError("Data not loaded.")
			target = self.data
		else:
			target = data

		if exclude_features is None:
			exclude_features = []

		self._log(f"[Scale] strategy='{self.scaling_strategy}', fit={fit}")

		# Chọn chiến lược chuẩn hóa
		numeric_cols_to_scale = [c for c in self.numeric_cols if c not in exclude_features]
		
		if fit or self.scaler is None:
			if self.scaling_strategy == 'standard':
				self.scaler = StandardScaler()
			elif self.scaling_strategy == 'minmax':
				self.scaler = MinMaxScaler()
			else:
				self._log("Unknown scaling strategy. Defaulting to StandardScaler.")
				self.scaler = StandardScaler()

			# FIT scaler trên tập hiện tại (thường là train)
			if numeric_cols_to_scale:
				target[numeric_cols_to_scale] = self.scaler.fit_transform(target[numeric_cols_to_scale])
				# Lưu danh sách cột đã scale để dùng khi transform
				self.scaled_cols_ = numeric_cols_to_scale

			# Nếu là MinMaxScaler thì lưu min/max theo cột
			if isinstance(self.scaler, MinMaxScaler) and numeric_cols_to_scale:
				self.scale_min_ = dict(zip(numeric_cols_to_scale, self.scaler.data_min_))
				self.scale_max_ = dict(zip(numeric_cols_to_scale, self.scaler.data_max_))
		else:
			# Chỉ transform với scaler đã fit (cho test/val)
			# Sử dụng danh sách cột đã được fit
			cols_to_transform = [c for c in self.scaled_cols_ if c in target.columns and c not in exclude_features]
			if cols_to_transform:
				target[cols_to_transform] = self.scaler.transform(target[cols_to_transform])

		# Cập nhật lại phân loại cột chỉ khi làm trên self.data
		if data is None:
			self.data = target
			return self
		else:
			return target
	
	def drop_features(self, features_to_drop):
		"""
		Xóa các cột không cần thiết khỏi DataFrame
		
		Parameters
		----------
		features_to_drop : list of str
			Danh sách tên các cột cần xóa

		Returns
		-------
		self
			Trả về chính đối tượng để có thể chain methods

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		"""
		if self.data is None:
			raise ValueError("Data not loaded.")

		self._log(f"Dropping features: {features_to_drop}")
		self.data = self.data.drop(columns=features_to_drop, errors='ignore')  # Xóa các cột không cần thiết
		# Cập nhật lại phân loại cột sau khi xóa
		self.auto_detect_columns()
		return self

	def get_processed_data(self):
		"""
		Trả về DataFrame đã được xử lý và sẵn sàng sử dụng
		
		Returns
		-------
		DataFrame
			DataFrame đã được tiền xử lý hoàn chỉnh, sẵn sàng cho huấn luyện mô hình
		"""
		self._log("Data processing pipeline complete. Returning data.")
		return self.data

	def save_data(self, filepath):
		"""
		Lưu dữ liệu đã xử lý vào file CSV
		
		Parameters
		----------
		filepath : str
			Đường dẫn file CSV để lưu dữ liệu

		Raises
		------
		ValueError
			Nếu không có dữ liệu đã xử lý để lưu
		Exception
			Các lỗi khác trong quá trình lưu file
		"""
		if self.data is None:
			raise ValueError("No processed data to save.")

		self._log(f"Saving processed data to {filepath}...")
		try:
			self.data.to_csv(filepath, index=False)  # Lưu dữ liệu vào file CSV mà không kèm index của pandas
			self._log("Save successful.")
		except Exception as e:
			self._log(f"Error saving data: {e}")
			raise