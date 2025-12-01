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
	
	parser.add_argument('--config', type=str, default='config.ini',
						help='Đường dẫn tới file cấu hình (mặc định: config.ini)')
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
	
	return parser.parse_args()

def load_config(config_path, args):
	"""Tải cấu hình từ file và kết hợp với các tham số dòng lệnh"""
	config = configparser.ConfigParser()
	
	# Giá trị mặc định nếu file config không tồn tại
	default_config = {
		'data_file': 'Data/FastFoodNutritionMenuV3.csv',
		'temp_data_file': 'Data/temp_processed_data.csv',
		'target_column': 'calories',
		'test_size': 0.2,
		'random_state': 42,
		'num_strategy': 'drop',
		'cat_strategy': 'mode',
		'dt_strategy': 'drop',
		'scaling_strategy': 'standard',
		'outlier_method': 'zscore',
		'enable_optimization': False,
		'n_trials': 50,
		'n_jobs': 3,
		'models_to_optimize': ['RandomForest', 'LightGBM', 'Ridge', 'Lasso', 'ElasticNet'],
		'enable_eda': True,
		'enable_plots': True,
		'feature_importance_top_n': 15,
		'results_csv': 'results/evaluation_results.csv',
		'results_json': 'results/evaluation_results.json',
		'comparison_plot': 'plots/comparison.png',
		'importance_plot': 'plots/importance.png'
	}
	
	if os.path.exists(config_path):
		config.read(config_path)
		
		# Phân tích các giá trị từ config
		result_config = {}
		result_config['data_file'] = config.get('PATHS', 'data_file', fallback=default_config['data_file'])
		result_config['temp_data_file'] = config.get('PATHS', 'temp_data_file', fallback=default_config['temp_data_file'])
		result_config['target_column'] = config.get('DATA', 'target_column', fallback=default_config['target_column'])
		result_config['test_size'] = config.getfloat('DATA', 'test_size', fallback=default_config['test_size'])
		result_config['random_state'] = config.getint('DATA', 'random_state', fallback=default_config['random_state'])
		
		result_config['num_strategy'] = config.get('PREPROCESSING', 'num_strategy', fallback=default_config['num_strategy'])
		result_config['cat_strategy'] = config.get('PREPROCESSING', 'cat_strategy', fallback=default_config['cat_strategy'])
		result_config['dt_strategy'] = config.get('PREPROCESSING', 'dt_strategy', fallback=default_config['dt_strategy'])
		result_config['scaling_strategy'] = config.get('PREPROCESSING', 'scaling_strategy', fallback=default_config['scaling_strategy'])
		result_config['outlier_method'] = config.get('PREPROCESSING', 'outlier_method', fallback=default_config['outlier_method'])
		
		result_config['enable_optimization'] = config.getboolean('OPTIMIZATION', 'enable_optimization', fallback=default_config['enable_optimization'])
		result_config['n_trials'] = config.getint('OPTIMIZATION', 'n_trials', fallback=default_config['n_trials'])
		result_config['n_jobs'] = config.getint('OPTIMIZATION', 'n_jobs', fallback=default_config['n_jobs'])
		models_str = config.get('OPTIMIZATION', 'models_to_optimize', fallback=','.join(default_config['models_to_optimize']))
		result_config['models_to_optimize'] = [m.strip() for m in models_str.split(',')]
		
		result_config['enable_eda'] = config.getboolean('VISUALIZATION', 'enable_eda', fallback=default_config['enable_eda'])
		result_config['enable_plots'] = config.getboolean('VISUALIZATION', 'enable_plots', fallback=default_config['enable_plots'])
		result_config['feature_importance_top_n'] = config.getint('VISUALIZATION', 'feature_importance_top_n', fallback=default_config['feature_importance_top_n'])
		
		result_config['results_csv'] = config.get('OUTPUT', 'results_csv', fallback=default_config['results_csv'])
		result_config['results_json'] = config.get('OUTPUT', 'results_json', fallback=default_config['results_json'])
		result_config['comparison_plot'] = config.get('OUTPUT', 'comparison_plot', fallback=default_config['comparison_plot'])
		result_config['importance_plot'] = config.get('OUTPUT', 'importance_plot', fallback=default_config['importance_plot'])
	else:
		result_config = default_config.copy()
	
	# Ghi đè với các tham số dòng lệnh
	if args.data:
		result_config['data_file'] = args.data
	if args.target:
		result_config['target_column'] = args.target
	if args.test_size:
		result_config['test_size'] = args.test_size
	if args.random_state:
		result_config['random_state'] = args.random_state
	if args.optimize:
		result_config['enable_optimization'] = True
	if args.no_eda:
		result_config['enable_eda'] = False
	if args.no_viz:
		result_config['enable_plots'] = False
	
	return result_config

if __name__ == "__main__":
	# Phân tích tham số và tải cấu hình
	args = parse_arguments()
	config = load_config(args.config, args)
	
	setup_logging()
	logger = logging.getLogger("MAIN")
	
	logger.info(f"Using configuration file: {args.config}")
	logger.info(f"Data file: {config['data_file']}")
	logger.info(f"Target column: {config['target_column']}")
	logger.info(f"Optimization enabled: {config['enable_optimization']}")
	logger.info(f"EDA enabled: {config['enable_eda']}")
	logger.info(f"Visualization enabled: {config['enable_plots']}")
	
	file_path = config['data_file']
	target_col = config['target_column']

	# =========================================================================
	# 1. TIỀN XỬ LÝ SƠ BỘ (SAFE PREPROCESSING)
	# =========================================================================

	preprocessor = DataPreprocessor(
		num_strategy=config['num_strategy'],
		cat_strategy=config['cat_strategy'], 
		dt_strategy=config['dt_strategy'],
		scaling_strategy=config['scaling_strategy'],
		outlier_method=config['outlier_method'],
	)
	preprocessor.load_data(file_path, auto_convert_numeric=True)
	
	# Chuyển đổi datetime nếu có (tách riêng khỏi auto_detect_columns)
	preprocessor.convert_to_datetime()

	# =========================================================================
	# 2. EDA TRƯỚC KHI XỬ LÝ DỮ LIỆU
	# =========================================================================

	if config['enable_eda']:
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
	preprocessor.save_data(config['temp_data_file'])

	logger.info("Initializing ModelTrainer to split data...")
	trainer = ModelTrainer(random_state=config['random_state'])
	
	# Nạp dữ liệu vào ModelTrainer
	trainer.load_data(current_data, target_column=target_col)
	
	# GỌI HÀM CỦA CLASS ĐỂ CHIA TRAIN/TEST (Thay vì dùng sklearn trực tiếp)
	train_df, test_df = trainer.split_data(test_size=config['test_size'])

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
	
	if config['enable_eda']:
		eda_after = EDA(merged_df, show_plots=False)
		eda_after.perform_eda(save_path='plots/eda/after')

	# =========================================================================
	# 5. HUẤN LUYỆN & ĐÁNH GIÁ
	# =========================================================================
	
	# Nạp dữ liệu sạch ngược lại vào Trainer để tách X, y
	trainer.set_training_data(train_processed, test_processed, target_col=target_col)
	
	# Khởi tạo các mô hình
	trainer.initialize_models()

	# Optimize hyperparams cho tất cả models (configurable trials)
	if config['enable_optimization']:
		for model_name in config['models_to_optimize']:
			logger.info(f"Optimizing {model_name}...")
			trainer.optimize_params(model_name, n_trials=config['n_trials'], n_jobs=config['n_jobs'])
	
	# Train tất cả models với params đã optimize
	trainer.train_models()
	
	# Đánh giá và so sánh tất cả models
	evaluation_output = trainer.evaluate_models()
	results_list = evaluation_output['results']
	
	# Lưu kết quả
	ModelIO.save_results(results_list, filepath=config['results_csv'], format='csv')
	ModelIO.save_results(results_list, filepath=config['results_json'], format='json')
	
	# Lưu mô hình tốt nhất
	if trainer.best_model:
		ModelIO.save_model(trainer.best_model, trainer.best_model_name)
	
	# =========================================================================
	# 6. VISUALIZE
	# =========================================================================
	if config['enable_plots']:
		vis = ModelVisualizer(evaluation_output)
		vis.plot_model_comparison(save_path=config['comparison_plot'])
		
		# Feature importance (top_n đã được xử lý trong get_feature_importance)
		imp_df = trainer.get_feature_importance(top_n=config['feature_importance_top_n'])
		vis.plot_feature_importance(imp_df, save_path=config['importance_plot'])

	logger.info("Process Completed.")