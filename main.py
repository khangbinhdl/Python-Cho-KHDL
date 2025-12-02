from DataPreprocessor.Preprocessor import DataPreprocessor
from Visualizer.EDA import EDA
from Visualizer.ModelVisualizer import ModelVisualizer
from ModelTrainer.ModelTrainer import ModelTrainer
from ModelTrainer.ModelIO import ModelIO

import logging
from logging import StreamHandler, FileHandler, Formatter
from datetime import datetime
import os
import pandas as pd
import argparse
import configparser

def setup_logging():
	os.makedirs("logs", exist_ok=True)
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	log_path = os.path.join("logs", f"pipeline_{timestamp}.log")
	
	root = logging.getLogger()
	for h in list(root.handlers): root.removeHandler(h)
	root.setLevel(logging.INFO)
	
	formatter = Formatter("%(asctime)s [%(name)s]: %(message)s", datefmt="%H:%M:%S")
	ch = StreamHandler()
	ch.setFormatter(formatter)
	fh = FileHandler(log_path, mode="w", encoding="utf-8")
	fh.setFormatter(formatter)
	
	root.addHandler(ch)
	root.addHandler(fh)
	return log_path

def parse_arguments():
	"""Phân tích các tham số dòng lệnh"""
	parser = argparse.ArgumentParser(description='ML Pipeline for Fast Food Nutrition Data')
	
	parser.add_argument('--config', type=str, default='default_config.ini',
						help='Đường dẫn tới file cấu hình (mặc định: default_config.ini)')
	parser.add_argument('--data', type=str, 
						help='Đường dẫn tới file dữ liệu (ghi đè config)')
	parser.add_argument('--target', type=str,
						help='Tên cột target (ghi đè config)')
	parser.add_argument('--optimize', action='store_true',
						help='Bật tối ưu hóa hyperparameter')
	parser.add_argument('--no-eda', action='store_true',
						help='Bỏ qua tạo EDA')
	parser.add_argument('--no-viz', action='store_true',
						help='Bỏ qua tạo biểu đồ visualization')
	parser.add_argument('--test-size', type=float,
						help='Tỷ lệ tập test (ghi đè config)')
	parser.add_argument('--random-state', type=int,
						help='Random state để tái tạo kết quả (ghi đè config)')
	parser.add_argument('--models', type=str,
						help='Các model để train (all/RandomForest,LightGBM,Ridge,Lasso,ElasticNet,LinearRegression) (ghi đè config)')
	
	return parser.parse_args()

def load_config(config_path, args):
	"""Tải cấu hình từ file và kết hợp với các tham số dòng lệnh"""
	config = configparser.ConfigParser()
	
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"File cấu hình {config_path} không tồn tại!")
	
	config.read(config_path, encoding='utf-8')
	
	# Ghi đè với các tham số dòng lệnh
	if args.data:
		config.set('PATHS', 'data_file', args.data)
	if args.target:
		config.set('DATA', 'target_column', args.target)
	if args.test_size:
		config.set('DATA', 'test_size', str(args.test_size))
	if args.random_state:
		config.set('DATA', 'random_state', str(args.random_state))
	if args.optimize:
		config.set('OPTIMIZATION', 'enable_optimization', 'true')
	if args.no_eda:
		config.set('VISUALIZATION', 'enable_eda', 'false')
	if args.no_viz:
		config.set('VISUALIZATION', 'enable_plots', 'false')
	if args.models:
		config.set('MODEL', 'selected_models', args.models)
	
	return config

if __name__ == "__main__":
	# Phân tích tham số và tải cấu hình
	args = parse_arguments()
	config = load_config(args.config, args)
	
	setup_logging()
	logger = logging.getLogger("MAIN")
	
	logger.info(f"Using configuration file: {args.config}")
	logger.info(f"Data file: {config.get('PATHS', 'data_file')}")
	logger.info(f"Target column: {config.get('DATA', 'target_column')}")
	logger.info(f"Selected models: {config.get('MODEL', 'selected_models')}")
	logger.info(f"Optimization enabled: {config.getboolean('OPTIMIZATION', 'enable_optimization')}")
	logger.info(f"EDA enabled: {config.getboolean('VISUALIZATION', 'enable_eda')}")
	logger.info(f"Visualization enabled: {config.getboolean('VISUALIZATION', 'enable_plots')}")
	
	file_path = config.get('PATHS', 'data_file')
	target_col = config.get('DATA', 'target_column')

	# =========================================================================
	# 1. TIỀN XỬ LÝ SƠ BỘ (SAFE PREPROCESSING)
	# =========================================================================

	preprocessor = DataPreprocessor(
		num_strategy=config.get('PREPROCESSING', 'num_strategy'),
		cat_strategy=config.get('PREPROCESSING', 'cat_strategy'), 
		dt_strategy=config.get('PREPROCESSING', 'dt_strategy'),
		scaling_strategy=config.get('PREPROCESSING', 'scaling_strategy'),
		outlier_method=config.get('PREPROCESSING', 'outlier_method'),
	)
	preprocessor.load_data(file_path, auto_convert_numeric=True)
	
	# Chuyển đổi datetime nếu có (tách riêng khỏi auto_detect_columns)
	preprocessor.convert_to_datetime()

	# =========================================================================
	# 2. EDA TRƯỚC KHI XỬ LÝ DỮ LIỆU
	# =========================================================================

	if config.getboolean('VISUALIZATION', 'enable_eda'):
		eda_before = EDA(preprocessor.get_processed_data(), show_plots=False)
		eda_before.perform_eda(save_path='plots/eda/before')

	# =========================================================================
	# 3. Chia dữ liệu TRAIN/TEST, TIẾN HÀNH XỬ LÝ DỮ LIỆU
	# =========================================================================

	# Drop cột rác và clean giá trị âm
	preprocessor.drop_features(['calories_from_fat', 'weight_watchers_pnts', 'company', 'item'])
	preprocessor.clean_negative_values()
	
	# One-hot encoding (Làm trước split để đảm bảo đồng bộ cột)
	preprocessor.encode_categorical(strategy='onehot')

	# Loại bỏ duplicate trước khi chia train/test
	preprocessor.remove_duplicates()

	
	# Lấy dữ liệu tạm thời (đã clean cơ bản)
	current_data = preprocessor.get_processed_data()
	preprocessor.save_data(config.get('PATHS', 'temp_data_file'))

	logger.info("Initializing ModelTrainer to split data...")
	trainer = ModelTrainer(random_state=config.getint('DATA', 'random_state'))
	
	# Nạp dữ liệu vào ModelTrainer
	trainer.load_data(current_data, target_column=target_col)
	
	# GỌI HÀM CỦA CLASS ĐỂ CHIA TRAIN/TEST (Thay vì dùng sklearn trực tiếp)
	train_df, test_df = trainer.split_data(test_size=config.getfloat('DATA', 'test_size'))

	logger.info("Processing Split Data (Preventing Leakage)...")
	
	# DEBUG: Kiểm tra missing values trước khi xử lý
	logger.info(f"Train NaN count before processing: {train_df.isna().sum().sum()}")
	logger.info(f"Test NaN count before processing: {test_df.isna().sum().sum()}")

	# --- Xử lý tập TRAIN (FIT & TRANSFORM) ---
	# 1. Missing: Học median từ train -> điền vào train
	train_processed = preprocessor.handle_missing_values(data=train_df, fit=True)
	logger.info(f"Train NaN count after missing handling: {train_processed.isna().sum().sum()}")
	
	# 2. Outliers: Chỉ loại bỏ trên tập TRAIN
	train_processed = preprocessor.handle_outliers(
		data=train_processed, 
		exclude_features=[target_col] 
	)
	
	# 3. Scaling: Học min/max/std từ train -> scale train
	train_processed = preprocessor.scale_features(
		data=train_processed, 
		exclude_features=[target_col], 
		fit=True
	)

	logger.info(f"Train NaN count after scaling: {train_processed.isna().sum().sum()}")
	
	# --- Xử lý tập TEST (CHỈ TRANSFORM) ---
	# 1. Missing: Dùng median đã học từ train -> điền vào test
	test_processed = preprocessor.handle_missing_values(data=test_df, fit=False)
	logger.info(f"Test NaN count after missing handling: {test_processed.isna().sum().sum()}")
	
	# 2. Scaling: Dùng tham số đã học từ train -> scale test
	test_processed = preprocessor.scale_features(
		data=test_processed, 
		exclude_features=[target_col], 
		fit=False
	)
	logger.info(f"Test NaN count after scaling: {test_processed.isna().sum().sum()}")

	# =========================================================================
	# 4. EDA SAU KHI XỬ LÝ DỮ LIỆU
	# =========================================================================

	merged_df = pd.concat([train_processed, test_processed], ignore_index=True)
	
	logger.info(f"Train set size: {len(train_processed)}")
	logger.info(f"Test set size: {len(test_processed)}")
	logger.info(f"Merged set size: {len(merged_df)}")
	
	if config.getboolean('VISUALIZATION', 'enable_eda'):
		eda_after = EDA(merged_df, show_plots=False)
		eda_after.perform_eda(save_path='plots/eda/after')

	# =========================================================================
	# 5. HUẤN LUYỆN & ĐÁNH GIÁ
	# =========================================================================
	
	# Nạp dữ liệu sạch ngược lại vào Trainer để tách X, y
	trainer.set_training_data(train_processed, test_processed, target_col=target_col)
	
	# Khởi tạo các mô hình
	trainer.initialize_models()
	
	# Chọn models cần train
	if config.get('MODEL', 'selected_models').lower() != 'all':
		# Parse selected models từ string
		selected_model_names = [m.strip() for m in config.get('MODEL', 'selected_models').split(',')]
		# Lọc chỉ giữ lại các models được chọn
		filtered_models = {name: model for name, model in trainer.models.items() if name in selected_model_names}
		trainer.models = filtered_models
		logger.info(f"Training only selected models: {list(trainer.models.keys())}")
	else:
		logger.info(f"Training all available models: {list(trainer.models.keys())}")

	# Optimize hyperparams cho các models được chọn (configurable trials)
	if config.getboolean('OPTIMIZATION', 'enable_optimization'):
		# Chỉ optimize các models có trong trainer.models
		models_to_optimize = [m.strip() for m in config.get('OPTIMIZATION', 'models_to_optimize').split(',')]
		models_to_opt = [m for m in models_to_optimize if m in trainer.models.keys()]
		for model_name in models_to_opt:
			logger.info(f"Optimizing {model_name}...")
			trainer.optimize_params(model_name, n_trials=config.getint('OPTIMIZATION', 'n_trials'), n_jobs=config.getint('OPTIMIZATION', 'n_jobs'))
	
	# Train tất cả models với params đã optimize
	trainer.train_models()
	
	# Đánh giá và so sánh tất cả models
	evaluation_output = trainer.evaluate_models()
	results_list = evaluation_output['results']
	
	# Lưu kết quả
	ModelIO.save_results(results_list, filepath=config.get('OUTPUT', 'results_csv'), format='csv')
	ModelIO.save_results(results_list, filepath=config.get('OUTPUT', 'results_json'), format='json')
	
	# Lưu mô hình tốt nhất
	if trainer.best_model:
		ModelIO.save_model(trainer.best_model, trainer.best_model_name)
	
	# =========================================================================
	# 6. VISUALIZE
	# =========================================================================
	if config.getboolean('VISUALIZATION', 'enable_plots'):
		vis = ModelVisualizer(evaluation_output)
		vis.plot_model_comparison(save_path=config.get('OUTPUT', 'comparison_plot'))
		
		# Feature importance (top_n đã được xử lý trong get_feature_importance)
		imp_df = trainer.get_feature_importance(top_n=config.getint('VISUALIZATION', 'feature_importance_top_n'))
		vis.plot_feature_importance(imp_df, save_path=config.get('OUTPUT', 'importance_plot'))

	logger.info("Process Completed.")