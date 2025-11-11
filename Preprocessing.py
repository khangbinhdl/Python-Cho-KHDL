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

# Thiết lập Pandas để không hiển thị số khoa học
pd.set_option('display.float_format', '{:.6f}'.format)

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
	def handle_missing_values(self, num_strategy=None, cat_strategy=None, dt_strategy="drop"):
		"""
		Xử lý giá trị bị thiếu với phương pháp riêng cho từng loại cột
		
		Áp dụng các chiến lược khác nhau cho cột số, cột phân loại và cột datetime.
		Nếu không truyền tham số, sẽ sử dụng giá trị mặc định từ self.missing_strategy.

		Parameters
		----------
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

		# Mặc định kế thừa self.missing_strategy nếu không được truyền riêng
		num_strategy = num_strategy or self.missing_strategy
		cat_strategy = cat_strategy or self.missing_strategy

		self._log(
			f"Handling missing values | num: '{num_strategy}', cat: '{cat_strategy}', dt: '{dt_strategy}'"
		)

		# Đếm số giá trị thiếu trước khi xử lý
		initial_rows = len(self.data)
		missing_before = self.data.isnull().sum()
		total_missing_before = missing_before.sum()
		
		self._log(f"Total missing values before handling: {total_missing_before} across all columns")
		
		# Log chi tiết từng cột có giá trị thiếu
		if total_missing_before > 0:
			for col in missing_before[missing_before > 0].index:
				col_type = 'numeric' if col in self.numeric_cols else ('categorical' if col in self.categorical_cols else 'datetime')
				self._log(f"  - Column '{col}' ({col_type}): {missing_before[col]} missing values")

		# Xử lý cột Datetime
		if self.datetime_cols: # Kiểm tra xem có cột datetime nào không
			if dt_strategy == "drop":
				# Bỏ bất kỳ hàng nào có giá trị ngày giờ bị thiếu
				self.data = self.data.dropna(subset=self.datetime_cols)
			elif dt_strategy == "ffill":
				# Điền giá trị của hàng ngay trước đó
				self.data[self.datetime_cols] = self.data[self.datetime_cols].ffill()
			elif dt_strategy == "bfill":
				# Điền giá trị của hàng ngay sau đó
				self.data[self.datetime_cols] = self.data[self.datetime_cols].bfill()

		# Xử lý cột Numeric
		if self.numeric_cols: # Kiểm tra xem có cột số nào không
			if num_strategy == "drop":
				# Drop theo hàng nếu bất kỳ numeric col nào bị thiếu
				self.data = self.data.dropna(subset=self.numeric_cols)
			elif num_strategy in ("mean", "median", "mode"):
				for col in self.numeric_cols:
					# Tính toán giá trị điền
					if self.data[col].isna().any():
						if num_strategy == "mean":
							#tính giá trị trung bình cộng của cột
							val = self.data[col].mean()
						elif num_strategy == "median":
							#tính giá trị trung vị của cột
							val = self.data[col].median()
						else:
							# lấy giá trị xuất hiện nhiều nhất
							# .mode() trả về một Series (vì có thể có nhiều mode), .iloc[0] lấy mode đầu tiên
							# Phòng trường hợp cột không có mode (ví dụ: toàn NaN đã bị lọc), dùng median làm dự phòng
							val = self.data[col].mode().iloc[0] if not self.data[col].mode().empty else self.data[col].median()
						# Áp dụng điền
						self.data[col] = self.data[col].fillna(val)
			elif num_strategy == "ffill":
				#Điền giá trị của hàng ngay trước đó
				self.data[self.numeric_cols] = self.data[self.numeric_cols].ffill()
			elif num_strategy == "bfill":
				#Điền giá trị của hàng ngay sau đó
				self.data[self.numeric_cols] = self.data[self.numeric_cols].bfill()

		# Xử lý cột Categorical
		if self.categorical_cols: # Kiểm tra xem có cột phân loại nào không
			if cat_strategy == "drop":
				#Xóa bất kỳ hàng nào có giá trị NaN trong cột phân loại
				self.data = self.data.dropna(subset=self.categorical_cols)
			elif cat_strategy == "mode":
				for col in self.categorical_cols: # Lặp qua từng cột phân loại
					if self.data[col].isna().any(): # Chỉ xử lý nếu cột có giá trị thiếu
						mode = self.data[col].mode() # Tìm mode của cột đó (có thể trả về nhiều giá trị)
						# Điền bằng mode đầu tiên. Nếu cột không có mode (ví dụ: toàn NaN),
						# Điền bằng một giá trị mặc định là "Unknown".
						self.data[col] = self.data[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
			elif cat_strategy == "ffill":
				#Điền giá trị của hàng ngay trước đó
				self.data[self.categorical_cols] = self.data[self.categorical_cols].ffill()
			elif cat_strategy == "bfill":
				#Điền giá trị của hàng ngay sau đó
				self.data[self.categorical_cols] = self.data[self.categorical_cols].bfill()
			elif cat_strategy == "constant":
				#Chiến lược 'constant': Điền một giá trị hằng số (ở đây là "Unknown")
				self.data[self.categorical_cols] = self.data[self.categorical_cols].fillna("Unknown")

		# Đếm số giá trị thiếu sau khi xử lý
		missing_after = self.data.isnull().sum()
		total_missing_after = missing_after.sum()
		rows_after = len(self.data)
		
		# Báo cáo kết quả
		values_filled = total_missing_before - total_missing_after
		rows_dropped = initial_rows - rows_after
		
		self._log(f"Missing values handled: {values_filled} values filled")
		if rows_dropped > 0:
			self._log(f"Rows dropped due to missing values: {rows_dropped} ({rows_dropped/initial_rows*100:.2f}%)")
		self._log(f"Remaining missing values: {total_missing_after}")
		
		# Cập nhật lại phân loại cột vì sau xử lý khi số hàng/cột có thể thay đổi
		self.auto_detect_columns()
		return self

	def handle_outliers(self, exclude_features=None, outlier_strategy='drop'):
		"""
		Phát hiện và xử lý các ngoại lai (outliers) trong dữ liệu
		
		Sử dụng một trong ba phương pháp: IQR, Z-score hoặc Isolation Forest
		để phát hiện outliers, sau đó loại bỏ hàng (drop) hoặc cắt giá trị (clip).

		Parameters
		----------
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
		self
			Trả về chính đối tượng để có thể chain methods

		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp

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
		if self.data is None:
			raise ValueError("Data not loaded.")

		if exclude_features is None:
			exclude_features = []

		self._log(f"Handling outliers using method: '{self.outlier_method}', strategy: '{outlier_strategy}'")
		initial_rows = len(self.data)
		clipped_values_count = 0

		# XỬ LÝ CHO IQR VÀ Z-SCORE
		if self.outlier_method in ('iqr', 'zscore'):
			if outlier_strategy == 'drop':
				# Tạo mask để giữ lại các hàng không phải outlier
				mask = pd.Series([True] * len(self.data), index=self.data.index)
				
				for col in self.numeric_cols:
					if col in exclude_features:
						self._log(f"Skipping outlier handling for excluded feature: {col}")
						continue
					
					# Phát hiện và loại bỏ ngoại lai bằng phương pháp IQR (Interquartile Range)
					if self.outlier_method == 'iqr':
						Q1 = self.data[col].quantile(0.25)
						Q3 = self.data[col].quantile(0.75)
						IQR = Q3 - Q1
						lower_bound = Q1 - 1.5 * IQR
						upper_bound = Q3 + 1.5 * IQR
						# Cập nhật mask: giữ lại hàng nằm trong khoảng hợp lệ
						mask &= (self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)

					# Phát hiện và loại bỏ ngoại lai bằng Z-score
					elif self.outlier_method == 'zscore':
						mean = self.data[col].mean()
						std = self.data[col].std()
						lower_bound = mean - 3 * std
						upper_bound = mean + 3 * std
						# Cập nhật mask: giữ lại hàng nằm trong khoảng hợp lệ
						mask &= (self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)
				
				# Áp dụng mask để loại bỏ outliers
				self.data = self.data[mask]
			
			elif outlier_strategy == 'clip':
				# Cắt giá trị outliers về ngưỡng min/max
				for col in self.numeric_cols:
					if col in exclude_features:
						self._log(f"Skipping outlier handling for excluded feature: {col}")
						continue
					
					if self.outlier_method == 'iqr':
						Q1 = self.data[col].quantile(0.25)
						Q3 = self.data[col].quantile(0.75)
						IQR = Q3 - Q1
						lower_bound = Q1 - 1.5 * IQR
						upper_bound = Q3 + 1.5 * IQR
					
					elif self.outlier_method == 'zscore':
						mean = self.data[col].mean()
						std = self.data[col].std()
						lower_bound = mean - 3 * std
						upper_bound = mean + 3 * std
					
					# Đếm số giá trị bị cắt
					clipped_values_count += ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
					
					# Cắt giá trị về ngưỡng
					self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)

		# XỬ LÝ CHO ISOLATION FOREST
		elif self.outlier_method == 'isolation_forest':
			numeric_cols_for_isolation = [col for col in self.numeric_cols if col not in exclude_features]
			if numeric_cols_for_isolation:  # Đảm bảo chạy trên cột số
				iso = IsolationForest(contamination=0.1, random_state=42)
				# Fit trên tất cả các cột số cùng lúc
				yhat = iso.fit_predict(self.data[numeric_cols_for_isolation])

				if outlier_strategy == 'drop':
					# Giữ lại các hàng không phải là ngoại lai (yhat == 1, outlier là -1)
					self.data = self.data[yhat == 1]
				elif outlier_strategy == 'clip':
					# Với Isolation Forest + clip: cắt từng cột về ngưỡng IQR
					outlier_mask = yhat == -1
					for col in numeric_cols_for_isolation:
						Q1 = self.data[col].quantile(0.25)
						Q3 = self.data[col].quantile(0.75)
						IQR = Q3 - Q1
						lower_bound = Q1 - 1.5 * IQR
						upper_bound = Q3 + 1.5 * IQR
						
						# Chỉ cắt giá trị của các hàng bị đánh dấu là outlier
						clipped_values_count += ((self.data.loc[outlier_mask, col] < lower_bound) | 
												(self.data.loc[outlier_mask, col] > upper_bound)).sum()
						self.data.loc[outlier_mask, col] = self.data.loc[outlier_mask, col].clip(
							lower=lower_bound, upper=upper_bound
						)
			else:
				self._log("Isolation Forest: No numeric columns to process after excluding specified features.")

		# Báo cáo kết quả
		if outlier_strategy == 'drop':
			rows_removed = initial_rows - len(self.data)
			self._log(f"Removed {rows_removed} rows ({rows_removed/initial_rows*100:.2f}% of data) as outliers")
			# Reset index sau khi loại bỏ hàng
			self.data = self.data.reset_index(drop=True)
		elif outlier_strategy == 'clip':
			self._log(f"Clipped {clipped_values_count} outlier values to bounds (no rows removed)")

		return self

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

	def scale_features(self):
		"""
		Chuẩn hóa các giá trị trong các cột số (numeric columns)
		
		Áp dụng StandardScaler hoặc MinMaxScaler để đưa các giá trị số
		về cùng một thang đo, giúp cải thiện hiệu suất của các thuật toán ML.

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
		if self.data is None:
			raise ValueError("Data not loaded.")

		self._log(f"Scaling numeric features using: '{self.scaling_strategy}'")

		# Chọn chiến lược chuẩn hóa (StandardScaler hoặc MinMaxScaler)
		if self.scaling_strategy == 'standard':
			self.scaler = StandardScaler()  # Chuẩn hóa với StandardScaler (đưa dữ liệu về trung bình 0, độ lệch chuẩn 1)
		elif self.scaling_strategy == 'minmax':
			self.scaler = MinMaxScaler()  # Chuẩn hóa với MinMaxScaler (đưa dữ liệu về phạm vi [0, 1])
		else:
			self._log("Unknown scaling strategy. Defaulting to StandardScaler.")
			self.scaler = StandardScaler()  # Mặc định dùng StandardScaler nếu chiến lược không hợp lệ

		if self.numeric_cols:
			self.data[self.numeric_cols] = self.scaler.fit_transform(self.data[self.numeric_cols])  # Áp dụng chuẩn hóa cho các cột số

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