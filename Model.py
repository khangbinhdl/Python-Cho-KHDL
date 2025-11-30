import pandas as pd
import numpy as np
import joblib
import pickle
import logging
import os
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from Optimization import BayesianOptimizer

# Logger riêng
LOGGER = logging.getLogger("MODEL_TRAINER")
if not LOGGER.handlers:
	LOGGER.propagate = True
	LOGGER.setLevel(logging.INFO)

class ModelTrainer:
	"""
	Class quản lý toàn bộ quy trình huấn luyện và đánh giá mô hình Machine Learning
	
	Quản lý các bước: Chia dữ liệu, Chuẩn bị X/y, Huấn luyện, Tối ưu hóa tham số và Đánh giá.
	
	Attributes
	----------
	random_state : int
		Seed cho reproducibility
	data : DataFrame or None
		Dữ liệu gốc được nạp vào
	train_df : DataFrame or None
		Dữ liệu training sau khi split
	test_df : DataFrame or None
		Dữ liệu testing sau khi split
	X_train : DataFrame or None
		Features của tập training
	X_test : DataFrame or None
		Features của tập testing
	y_train : Series or None
		Target của tập training
	y_test : Series or None
		Target của tập testing
	models : dict
		Dictionary chứa các model templates (chưa train)
	trained_models : dict
		Dictionary chứa các models đã được huấn luyện
	results : list
		Danh sách kết quả đánh giá các models
	best_model : object or None
		Model có hiệu suất tốt nhất
	best_model_name : str or None
		Tên của model tốt nhất
	"""

	def __init__(self, random_state=42):
		"""
		Khởi tạo ModelTrainer
		
		Parameters
		----------
		random_state : int, optional
			Seed cho reproducibility. Mặc định là 42
		"""
		self.random_state = random_state
		
		# Các biến chứa dữ liệu
		self.data = None
		self.train_df = None
		self.test_df = None
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None

		# Các biến liên quan đến models
		self.models = {}  # Model templates (chưa train)
		self.trained_models = {}  # Models đã train
		self.results = []  # Kết quả đánh giá
		self.best_model = None  # Model tốt nhất
		self.best_model_name = None  # Tên model tốt nhất

		np.random.seed(random_state)
		self._log("ModelTrainer initialized with random_state={}".format(random_state))

	def __str__(self):
		"""
		Biểu diễn chuỗi thân thiện với người dùng
		
		Returns
		-------
		str
			Chuỗi mô tả trạng thái hiện tại của trainer
		"""
		trained_count = len(self.trained_models)
		if trained_count == 0:
			return "ModelTrainer (chưa huấn luyện model nào)"
		return f"ModelTrainer: {trained_count} models đã train, best={self.best_model_name}"

	def __repr__(self):
		"""
		Biểu diễn chuỗi dành cho developer
		
		Returns
		-------
		str
			Chuỗi mô tả chi tiết
		"""
		return f"ModelTrainer(random_state={self.random_state}, trained={len(self.trained_models)})"

	@staticmethod
	def _log(message):
		LOGGER.info(message)

	def load_data(self, data, target_column='calories'):
		"""
		Nạp dữ liệu đã được tiền xử lý vào ModelTrainer
		
		Parameters
		----------
		data : DataFrame
			Dữ liệu đã được tiền xử lý từ DataPreprocessor
		target_column : str, optional
			Tên cột target cần dự đoán.
			Mặc định là 'calories'
			
		Returns
		-------
		self
			Trả về chính đối tượng để có thể chain methods
			
		Raises
		------
		ValueError
			Nếu dữ liệu không hợp lệ hoặc không chứa cột target
		"""
		if not isinstance(data, pd.DataFrame):
			raise ValueError("Data must be a pandas DataFrame")
			
		if target_column not in data.columns:
			raise ValueError(f"Target column '{target_column}' not found in data")
			
		self.data = data.copy()
		self.target_column = target_column
		
		self._log(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
		self._log(f"Target column: '{target_column}'")
		
		return self

	def split_data(self, test_size=0.2, stratify=None):
		"""
		Chia dữ liệu thành tập train và test
		
		Parameters
		----------
		test_size : float, optional
			Tỷ lệ dữ liệu dùng cho test (0.0 - 1.0). Mặc định là 0.2
		stratify : array-like or None, optional
			Nếu không None, dữ liệu được chia theo stratified split.
			Mặc định là None
			
		Returns
		-------
		train_df : DataFrame
			Dữ liệu training
		test_df : DataFrame
			Dữ liệu testing
			
		Raises
		------
		ValueError
			Nếu dữ liệu chưa được nạp
		"""
		if self.data is None:
			raise ValueError("Data not loaded. Call load_data() first.")
		
		self._log("Splitting data into Train and Test sets...")
		self.train_df, self.test_df = train_test_split(
			self.data, test_size=test_size, 
			random_state=self.random_state, stratify=stratify
		)
		self._log(f"Split complete. Train shape: {self.train_df.shape}, Test shape: {self.test_df.shape}")
		return self.train_df, self.test_df

	def set_training_data(self, train_processed, test_processed, target_col):
		"""
		Thiết lập dữ liệu đã xử lý và tách thành X, y
		
		Parameters
		----------
		train_processed : DataFrame
			Dữ liệu training đã được tiền xử lý
		test_processed : DataFrame
			Dữ liệu testing đã được tiền xử lý
		target_col : str
			Tên cột target cần dự đoán
			
		Notes
		-----
		Method này tách features (X) và target (y) cho cả train và test set.
		"""
		self._log("Setting processed training data (separating X and y)...")
		
		self.X_train = train_processed.drop(columns=[target_col])
		self.y_train = train_processed[target_col]
		
		self.X_test = test_processed.drop(columns=[target_col])
		self.y_test = test_processed[target_col]
		
		self._log(f"Ready for training. X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")

	def initialize_models(self):
		"""
		Khởi tạo danh sách các mô hình Machine Learning
		
		Khởi tạo nhiều loại mô hình hồi quy với các tham số mặc định.

		Returns
		-------
		self
			Trả về chính đối tượng để có thể chain methods
			
		Notes
		-----
		Các mô hình được khởi tạo:
		- LinearRegression: Hồi quy tuyến tính cơ bản
		- Ridge: Hồi quy Ridge với regularization L2
		- Lasso: Hồi quy Lasso với regularization L1
		- ElasticNet: Kết hợp Ridge và Lasso
		- XGBoost: XGBoost Regressor
		"""
		self.models = {
			'LinearRegression': LinearRegression(),
			
			'Ridge': Ridge(
				random_state=self.random_state
			),
			
			'Lasso': Lasso(
				random_state=self.random_state,
			),
			
			'ElasticNet': ElasticNet(
				random_state=self.random_state,
			),
			
			'RandomForest': RandomForestRegressor(
				random_state=self.random_state,
				n_jobs=2
			),

			'LightGBM': LGBMRegressor(
				random_state=self.random_state,
				verbose=-1,
				n_jobs=2
			),
		}
		
		self._log(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
		return self

	def train_models(self, models_to_train=None):
		"""
		Huấn luyện tất cả hoặc một phần các mô hình đã khởi tạo
		
		Parameters
		----------
		models_to_train : list of str, optional
			Danh sách tên các mô hình cần huấn luyện.
			Nếu None, huấn luyện tất cả mô hình.
			Mặc định là None
			
		Returns
		-------
		self
			Trả về chính đối tượng để có thể chain methods
			
		Raises
		------
		ValueError
			Nếu dữ liệu train chưa được chuẩn bị hoặc mô hình chưa được khởi tạo
		"""
		if self.X_train is None or self.y_train is None:
			raise ValueError("Training data not available. Call split_data() first.")
			
		if not self.models:
			self.initialize_models()
			
		if models_to_train is None:
			models_to_train = list(self.models.keys())
		
		# Reset trained_models to avoid keeping old models from previous runs
		self.trained_models = {}
			
		self._log("Starting model training...")
		
		for name in models_to_train:
			if name not in self.models:
				self._log(f"Warning: Model '{name}' not found, skipping...")
				continue
				
			try:
				self._log(f"Training {name}...")
				model = self.models[name]
				model.fit(self.X_train, self.y_train)
				self.trained_models[name] = model
				self._log(f"✓ {name} trained successfully")
				
			except Exception as e:
				self._log(f"✗ Error training {name}: {str(e)}")
				
		self._log(f"Training completed. {len(self.trained_models)}/{len(models_to_train)} models trained successfully")
		return self
		

	def evaluate_models(self):
		"""
		Đánh giá tất cả các mô hình đã huấn luyện trên tập test
		
		Tính toán các metrics: MSE, RMSE, MAE, R² cho mỗi model
		và xác định model tốt nhất dựa trên R² score.
		
		Returns
		-------
		dict
			Dictionary chứa:
			- 'results': list of dict, mỗi dict chứa metrics của một model
			- 'best_model_name': str, tên model có R² cao nhất
			
		Raises
		------
		ValueError
			Nếu chưa có models được huấn luyện hoặc chưa có dữ liệu test
			
		Notes
		-----
		Metrics được tính:
		- MSE (Mean Squared Error): Trung bình bình phương sai số
		- RMSE (Root Mean Squared Error): Căn bậc hai của MSE
		- MAE (Mean Absolute Error): Trung bình sai số tuyệt đối
		- R² Score: Hệ số xác định (càng gần 1 càng tốt)
		"""
		if not self.trained_models:
			raise ValueError("No trained models found. Call train_models() first.")
			
		if self.X_test is None or self.y_test is None:
			raise ValueError("Test data not available. Call split_data() first.")
		
		self._log("Evaluating models...")
		# Reset results to avoid accumulating old results from previous evaluations
		self.results = []
		self.evaluation_results = []
		best_score = -np.inf # So sánh bằng R2
		
		self._log("Evaluating models on Test set...")
		for name, model in self.trained_models.items():
			try:
				y_pred = model.predict(self.X_test)
				
				mse = mean_squared_error(self.y_test, y_pred)
				rmse = np.sqrt(mse)
				mae = mean_absolute_error(self.y_test, y_pred)
				r2 = r2_score(self.y_test, y_pred)

				result = {
					'model_name': name,
					'mse': mse,
					'rmse': rmse, 
					'mae': mae,
					'r2_score': r2
				}
				
				self.results.append(result)
				self.evaluation_results.append(result)
				self._log(f"✓ {name} evaluated: RMSE={rmse:.4f}, MAE={mae:.4f} ,R2={r2:.4f}")
				
				if r2 > best_score:
					best_score = r2
					self.best_model = model
					self.best_model_name = name
					
			except Exception as e:
				self._log(f"✗ Error evaluating {name}: {str(e)}")

		return {'results': self.results, 'best_model_name': self.best_model_name}

	def get_feature_importance(self, model_name=None, top_n=None):
		"""
		Lấy độ quan trọng của các features từ mô hình
		
		Parameters
		----------
		model_name : str, optional
			Tên mô hình. Nếu None, sử dụng mô hình tốt nhất.
			Mặc định là None
		top_n : int, optional
			Số lượng features quan trọng nhất để trả về.
			Nếu None, trả về tất cả.
			Mặc định là None
			
		Returns
		-------
		DataFrame
			DataFrame chứa tên feature và độ quan trọng
			
		Notes
		-----
		- Tree-based models (RandomForest, LightGBM): sử dụng feature_importances_
		- Linear models (LinearRegression, Ridge, Lasso, ElasticNet): sử dụng giá trị tuyệt đối của coef_
		"""
		# Xác định mô hình
		if model_name is None:
			if self.best_model is None:
				raise ValueError("No best model found. Train and evaluate models first.")
			model = self.best_model
			model_name = self.best_model_name
		else:
			if model_name not in self.trained_models:
				raise ValueError(f"Trained model '{model_name}' not found")
			model = self.trained_models[model_name]
			
		feature_names = self.X_train.columns
		
		# Lấy feature importance dựa trên loại mô hình
		if hasattr(model, 'feature_importances_'):
			# Tree-based models: RandomForest, LightGBM
			importances = model.feature_importances_
			self._log(f"Using feature_importances_ for {model_name}")
			
		elif hasattr(model, 'coef_'):
			# Linear models: LinearRegression, Ridge, Lasso, ElasticNet
			# Sử dụng giá trị tuyệt đối của hệ số
			importances = np.abs(model.coef_)
			self._log(f"Using absolute coefficients for {model_name}")
			
		else:
			raise ValueError(f"Model '{model_name}' does not support feature importance extraction")
		
		# Tạo DataFrame và sắp xếp
		importance_df = pd.DataFrame({
			'feature': feature_names,
			'importance': importances
		}).sort_values('importance', ascending=False)
		
		if top_n:
			importance_df = importance_df.head(top_n)
			
		self._log(f"✓ Feature importance extracted for {model_name}")
		return importance_df
	
	def optimize_params(self, model_name, n_trials=20, cv=5, n_jobs=1):
		"""
		Tối ưu hóa siêu tham số của model bằng Bayesian Optimization (Optuna)
		
		Sử dụng Optuna để tìm kiếm tham số tối ưu cho model được chỉ định.
		Không gian tìm kiếm được định nghĩa sẵn trong BayesianOptimizer.
		
		Parameters
		----------
		model_name : str
			Tên mô hình cần tối ưu.
			Các giá trị hợp lệ: 'RandomForest', 'LightGBM', 'Ridge', 'Lasso', 'ElasticNet'
		n_trials : int, optional
			Số lần thử nghiệm (trials) trong quá trình tối ưu. Mặc định là 20
		cv : int, optional
			Số fold cho cross-validation. Mặc định là 5
		n_jobs : int, optional
			Số CPU cores sử dụng khi training model sau khi optimize.
			Mặc định là 1
			
		Returns
		-------
		dict or None
			Dictionary chứa best parameters nếu optimization thành công.
			Trả về None nếu:
			- Model không cần optimize (LinearRegression)
			- Optimization thất bại
			- Model chưa được định nghĩa search space
			
		Raises
		------
		ValueError
			Nếu training data chưa sẵn sàng hoặc model_name không hợp lệ
			
		Notes
		-----
		- Sử dụng R² score làm metric tối ưu hóa
		- Model được cập nhật với best params nhưng CHƯA được train
		- Cần gọi train_models() sau khi optimize để train với params mới
		- LinearRegression không có hyperparameters nên bị bỏ qua
		"""
		if self.X_train is None or self.y_train is None:
			raise ValueError("Training data not available. Call set_training_data() first.")
		
		# Khởi tạo models nếu chưa có
		if not self.models:
			self.initialize_models()
			
		if model_name not in self.models:
			raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")

		# Bỏ qua LinearRegression vì không có gì để tối ưu
		if model_name == 'LinearRegression':
			self._log("LinearRegression does not require optimization. Skipping.")
			return None

		self._log(f"Starting optimization for {model_name}...")
		
		try:
			# 1. Khởi tạo Optimizer
			optimizer = BayesianOptimizer(
				self.X_train, self.y_train, 
				random_state=self.random_state, 
				cv=cv
			)
			
			# 2. Chạy optimization để lấy best params
			best_params = optimizer.optimize(model_name, n_trials=n_trials)
			
			# 3. Xử lý trường hợp optimization thất bại
			if best_params is None:
				self._log(f"Optimization failed for {model_name}. Model will use default params.")
				return None

			# 4. Tạo model mới với best params
			model_class = self.models[model_name].__class__
			model_default_params = self.models[model_name].get_params()
			
			# Merge best params với các params cố định
			final_params = best_params.copy()
			
			# Thêm các params cố định nếu model hỗ trợ
			if 'random_state' in model_default_params:
				final_params['random_state'] = self.random_state
			if 'n_jobs' in model_default_params:
				final_params['n_jobs'] = n_jobs
			# Chỉ LightGBM mới có verbose=-1, RandomForest không có
			if model_name == 'LightGBM' and 'verbose' in model_default_params:
				final_params['verbose'] = -1

			# Khởi tạo model với params tối ưu (CHƯA TRAIN)
			optimized_model = model_class(**final_params)
			
			# 5. Cập nhật vào self.models (chưa train, chỉ update params)
			self.models[model_name] = optimized_model
			
			self._log(f"✓ {model_name} parameters optimized successfully")
			self._log(f"  Best params: {best_params}")
			self._log(f"  Note: Call train_models() to train all models with optimized params.")
			
			return best_params

		except Exception as e:
			self._log(f"✗ Error during optimization of {model_name}: {str(e)}")
			self._log(f"  Model will use default params.")
			return None


	def load_model(self, filepath, method='joblib'):
		"""
		Nạp mô hình đã lưu từ file
		
		Parameters
		----------
		filepath : str
			Đường dẫn tuyệt đối hoặc tương đối đến file mô hình
		method : str, optional
			Phương thức deserialization.
			Các giá trị hợp lệ: 'joblib', 'pickle'.
			Mặc định là 'joblib'
			
		Returns
		-------
		object
			Model object đã được nạp từ file
			
		Raises
		------
		FileNotFoundError
			Nếu filepath không tồn tại
		ValueError
			Nếu method không hợp lệ
			
		Notes
		-----
		Joblib được khuyến nghị cho scikit-learn models vì hiệu quả hơn với numpy arrays.
		"""
		try:
			if method == 'joblib':
				model = joblib.load(filepath)
			elif method == 'pickle':
				with open(filepath, 'rb') as f:
					model = pickle.load(f)
			else:
				raise ValueError("method must be 'joblib' or 'pickle'")
				
			self._log(f"✓ Model loaded from: {filepath}")
			return model
			
		except Exception as e:
			self._log(f"✗ Error loading model: {str(e)}")
			raise
			
	def save_results(self, filepath=None, format='csv'):
		"""
		Lưu kết quả đánh giá các mô hình vào file
		
		Parameters
		----------
		filepath : str or None, optional
			Đường dẫn file để lưu kết quả.
			Nếu None, tự động tạo tên file với timestamp trong thư mục 'results/'.
			Mặc định là None
		format : str, optional
			Định dạng file output.
			Các giá trị hợp lệ: 'csv', 'json'.
			Mặc định là 'csv'
			
		Returns
		-------
		str
			Đường dẫn tuyệt đối của file đã lưu
			
		Raises
		------
		ValueError
			Nếu chưa có kết quả đánh giá (chưa gọi evaluate_models())
			
		Notes
		-----
		Thư mục 'results/' sẽ được tự động tạo nếu chưa tồn tại.
		Format CSV dễ mở bằng Excel, JSON tốt cho lưu trữ metadata.
		"""
		if not self.results:
			raise ValueError("No evaluation results found. Call evaluate_models() first.")
			
		# Tạo thư mục results nếu chưa có
		os.makedirs("results", exist_ok=True)
		
		# Tạo tên file nếu chưa có
		if filepath is None:
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			filepath = f"results/model_results_{timestamp}.{format}"
			
		try:
			if format == 'csv':
				df = pd.DataFrame(self.results)
				df.to_csv(filepath, index=False)
			elif format == 'json':
				with open(filepath, 'w', encoding='utf-8') as f:
					json.dump(self.results, f, indent=2, ensure_ascii=False)
			else:
				raise ValueError("format must be 'csv' or 'json'")
				
			self._log(f"✓ Results saved to: {filepath}")
			return filepath
			
		except Exception as e:
			self._log(f"✗ Error saving results: {str(e)}")
			raise

	def save_model(self, model_name=None, filepath=None, method='joblib'):
		"""
		Lưu mô hình đã huấn luyện vào file để sử dụng sau
		
		Parameters
		----------
		model_name : str or None, optional
			Tên mô hình cần lưu.
			Nếu None, lưu mô hình tốt nhất (best_model).
			Mặc định là None
		filepath : str or None, optional
			Đường dẫn file để lưu model.
			Nếu None, tự động tạo tên file với timestamp trong thư mục 'models/'.
			Mặc định là None
		method : str, optional
			Phương thức serialization.
			Các giá trị hợp lệ: 'joblib', 'pickle'.
			Mặc định là 'joblib'
			
		Returns
		-------
		str
			Đường dẫn tuyệt đối của file model đã lưu
			
		Raises
		------
		ValueError
			Nếu model không tồn tại trong trained_models hoặc chưa có best_model
			
		Notes
		-----
		- Thư mục 'models/' sẽ được tự động tạo nếu chưa tồn tại
		- Joblib được khuyến nghị cho scikit-learn models
		- File extension: .pkl cho joblib, .pickle cho pickle
		"""
		# Xác định mô hình cần lưu
		if model_name is None:
			if self.best_model is None:
				raise ValueError("No best model found. Train and evaluate models first.")
			model_to_save = self.best_model
			model_name = self.best_model_name
		else:
			if model_name not in self.trained_models:
				raise ValueError(f"Trained model '{model_name}' not found")
			model_to_save = self.trained_models[model_name]
			
		# Tạo thư mục models nếu chưa có
		os.makedirs("models", exist_ok=True)
		
		# Tạo tên file nếu chưa có
		if filepath is None:
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			if method == 'joblib':
				filepath = f"models/{model_name}_{timestamp}.pkl"
			else:
				filepath = f"models/{model_name}_{timestamp}.pickle"
				
		try:
			if method == 'joblib':
				joblib.dump(model_to_save, filepath)
			elif method == 'pickle':
				with open(filepath, 'wb') as f:
					pickle.dump(model_to_save, f)
			else:
				raise ValueError("method must be 'joblib' or 'pickle'")
				
			self._log(f"✓ Model '{model_name}' saved to: {filepath}")
			return filepath
			
		except Exception as e:
			self._log(f"✗ Error saving model: {str(e)}")
			raise