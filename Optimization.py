import optuna
import logging
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from lightgbm import LGBMRegressor

# Tắt bớt log của Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

LOGGER = logging.getLogger("OPTIMIZATION")
if not LOGGER.handlers:
	LOGGER.propagate = True
	LOGGER.setLevel(logging.INFO)

class BayesianOptimizer:
	"""
	Class chuyên biệt để tối ưu hóa Hyperparams bằng Optuna.
	
	Định nghĩa sẵn không gian tìm kiếm (Search Space) cho từng loại model.
	Sử dụng Bayesian Optimization để tìm kiếm tham số tối ưu.
	
	Attributes
	----------
	X_train : array-like
		Dữ liệu features cho training
	y_train : array-like
		Dữ liệu target cho training
	random_state : int
		Seed cho reproducibility
	cv : int
		Số fold cho cross-validation
	"""

	def __init__(self, X_train, y_train, random_state=42, cv=5):
		"""
		Khởi tạo BayesianOptimizer.
		
		Parameters
		----------
		X_train : array-like of shape (n_samples, n_features)
			Dữ liệu features cho training
		y_train : array-like of shape (n_samples,)
			Dữ liệu target cho training
		random_state : int, optional
			Seed cho reproducibility. Mặc định là 42
		cv : int, optional
			Số fold cho cross-validation. Mặc định là 5
		"""
		self.X_train = X_train
		self.y_train = y_train
		self.random_state = random_state
		self.cv = cv

	def _get_search_space(self, model_name):
		"""
		Trả về search space cho từng model.
		
		Parameters
		----------
		model_name : str
			Tên model cần lấy search space
			
		Returns
		-------
		dict or None
			Dictionary chứa search space với format:
			- key: tên parameter
			- value: tuple (type, min, max, log_scale)
			Trả về None nếu model không cần optimize (LinearRegression)
			
		Notes
		-----
		Các model được hỗ trợ:
		- RandomForest: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
		- LightGBM: n_estimators, learning_rate, num_leaves, max_depth, min_child_samples,
		  subsample, colsample_bytree, reg_alpha, reg_lambda
		- Ridge: alpha
		- Lasso: alpha
		- ElasticNet: alpha, l1_ratio
		"""
		if model_name == 'LinearRegression':
			return None
			
		search_spaces = {
			'RandomForest': {
				'n_estimators': ('int', 50, 500),
				'max_depth': ('int', 3, 30),
				'min_samples_split': ('int', 2, 20),
				'min_samples_leaf': ('int', 1, 10),
				'max_features': ('categorical', ['sqrt', 'log2'])
			},
			'LightGBM': {
				'n_estimators': ('int', 100, 1000),
				'learning_rate': ('float', 0.005, 0.3, True),  # True = log scale
				'num_leaves': ('int', 20, 150),
				'max_depth': ('int', 3, 20),
				'min_child_samples': ('int', 5, 100),
				'subsample': ('float', 0.5, 1.0),
				'colsample_bytree': ('float', 0.5, 1.0),
				'reg_alpha': ('float', 1e-8, 10.0, True),
				'reg_lambda': ('float', 1e-8, 10.0, True)
			},
			'Ridge': {
				'alpha': ('float', 1e-3, 100.0, True)
			},
			'Lasso': {
				'alpha': ('float', 1e-4, 10.0, True)
			},
			'ElasticNet': {
				'alpha': ('float', 1e-4, 10.0, True),
				'l1_ratio': ('float', 0.0, 1.0)
			}
		}
		
		return search_spaces.get(model_name, None)

	def _create_model(self, model_name, trial):
		"""
		Tạo model instance với parameters từ Optuna trial.
		
		Parameters
		----------
		model_name : str
			Tên model cần tạo
		trial : optuna.trial.Trial
			Optuna trial object để suggest parameters
			
		Returns
		-------
		object or None
			Model instance với parameters được suggest từ trial.
			Trả về None nếu model không được hỗ trợ.
			
		Notes
		-----
		Model được tạo với random_state cố định và n_jobs=2 (nếu hỗ trợ)
		"""
		search_space = self._get_search_space(model_name)
		
		if search_space is None:
			return None
			
		# Suggest params từ trial
		params = {}
		for param_name, param_config in search_space.items():
			param_type = param_config[0]
			
			if param_type == 'int':
				params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
			elif param_type == 'float':
				log_scale = param_config[3] if len(param_config) > 3 else False
				params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=log_scale)
			elif param_type == 'categorical':
				params[param_name] = trial.suggest_categorical(param_name, param_config[1])
		
		# Thêm fixed params
		params['random_state'] = self.random_state
		
		# Tạo model instance
		if model_name == 'RandomForest':
			params['n_jobs'] = 2
			return RandomForestRegressor(**params)
		elif model_name == 'LightGBM':
			params['n_jobs'] = 2
			params['verbose'] = -1
			return LGBMRegressor(**params)
		elif model_name == 'Ridge':
			return Ridge(**params)
		elif model_name == 'Lasso':
			return Lasso(**params)
		elif model_name == 'ElasticNet':
			return ElasticNet(**params)
		else:
			return None

	def optimize(self, model_name, n_trials=20):
		"""
		Chạy Bayesian Optimization để tìm parameters tối ưu.
		
		Parameters
		----------
		model_name : str
			Tên model cần optimize (VD: 'RandomForest', 'LightGBM', 'Ridge')
		n_trials : int, optional
			Số lần thử nghiệm (trials). Mặc định là 20
			
		Returns
		-------
		dict or None
			Dictionary chứa best parameters nếu optimization thành công.
			Trả về None nếu:
			- Model không cần optimize (LinearRegression)
			- Model chưa được định nghĩa search space
			- Optimization thất bại
			
		Notes
		-----
		Sử dụng cross-validation với scoring='neg_mean_squared_error'.
		Optimization direction là 'minimize' để tìm MSE nhỏ nhất.
		"""
		# Skip LinearRegression
		if model_name == 'LinearRegression':
			LOGGER.info("LinearRegression does not require optimization. Skipping.")
			return None
		
		# Kiểm tra xem model có được support không
		search_space = self._get_search_space(model_name)
		if search_space is None:
			LOGGER.warning(f"Model '{model_name}' chưa được định nghĩa search space. Skipping.")
			return None
		
		LOGGER.info(f"--- Optuna: Optimizing {model_name} ({n_trials} trials) ---")

		def objective(trial):
			try:
				model = self._create_model(model_name, trial)
				if model is None:
					return float('inf')
				
				# Cross-validation scoring
				scores = cross_val_score(
					model, self.X_train, self.y_train, 
					cv=self.cv, 
					scoring='neg_mean_squared_error', 
					n_jobs=1
				)
				# Trả về MSE (dương)
				return -scores.mean()
				
			except Exception as e:
				LOGGER.error(f"Trial failed: {e}")
				return float('inf')

		# Chạy optimization
		study = optuna.create_study(direction='minimize')
		study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

		best_params = study.best_params
		best_score = study.best_value
		
		LOGGER.info(f"✓ Best MSE: {best_score:.4f}")
		LOGGER.info(f"✓ Best Params: {best_params}")

		return best_params