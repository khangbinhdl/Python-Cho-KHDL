from Preprocessing import DataPreprocessor
from Visualize import EDA, ModelVisualize
from Model import ModelTrainer

import logging
from logging import StreamHandler, FileHandler, Formatter
from datetime import datetime
import os
import pandas as pd

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

if __name__ == "__main__":
	setup_logging()
	logger = logging.getLogger("MAIN")
	
	file_path = './FastFoodNutritionMenuV3.csv'
	target_col = 'calories'

	# =========================================================================
	# 1. TIỀN XỬ LÝ SƠ BỘ (SAFE PREPROCESSING)
	# =========================================================================

	preprocessor = DataPreprocessor(
		num_strategy='mean',
		cat_strategy='mode', 
		dt_strategy='drop',
		scaling_strategy='robust',
		outlier_method='isolation_forest',
	)
	preprocessor.load_data(file_path)

	# =========================================================================
	# 2. EDA TRƯỚC KHI XỬ LÝ DỮ LIỆU
	# =========================================================================

	# eda_before = EDA(preprocessor.get_processed_data(), show_plots=False)
	# eda_before.perform_eda(save_path='plots/eda/before')

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

	logger.info("Initializing ModelTrainer to split data...")
	trainer = ModelTrainer(random_state=42)
	
	# Nạp dữ liệu vào ModelTrainer
	trainer.load_data(current_data, target_column=target_col)
	
	# GỌI HÀM CỦA CLASS ĐỂ CHIA TRAIN/TEST (Thay vì dùng sklearn trực tiếp)
	train_df, test_df = trainer.split_data(test_size=0.2)

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

	# merged_df = pd.concat([train_processed, test_processed], ignore_index=True)
	
	# logger.info(f"Train set size: {len(train_processed)}")
	# logger.info(f"Test set size: {len(test_processed)}")
	# logger.info(f"Merged set size: {len(merged_df)}")
	
	# eda_after = EDA(merged_df, show_plots=False)
	# eda_after.perform_eda(save_path='plots/eda/after')

	# # =========================================================================
	# # =========================================================================
	# # 5. HUẤN LUYỆN & ĐÁNH GIÁ
	# # =========================================================================
	
	# Nạp dữ liệu sạch ngược lại vào Trainer để tách X, y
	trainer.set_training_data(train_processed, test_processed, target_col=target_col)
	
	# Khởi tạo các mô hình
	trainer.initialize_models()

	# Optimize hyperparams cho tất cả models (50 trials)
	# models_to_optimize = ['RandomForest', 'LightGBM', 'Ridge', 'Lasso', 'ElasticNet']
	# for model_name in models_to_optimize:
	#     logger.info(f"Optimizing {model_name}...")
	#     trainer.optimize_params(model_name, n_trials=50, n_jobs=3)
	
	# Train tất cả models với params đã optimize
	trainer.train_models()
	
	# Đánh giá và so sánh tất cả models
	results = trainer.evaluate_models()
	
	# Lưu kết quả
	trainer.save_results(filepath="results/evaluation_results.csv", format='csv')
	trainer.save_results(filepath="results/evaluation_results.json", format='json')
	
	# Lưu mô hình tốt nhất
	# trainer.save_model()
	
	# =========================================================================
	# 6. VISUALIZE
	# =========================================================================
	vis = ModelVisualize(results)
	vis.plot_model_comparison(save_path='plots/comparison.png')
	
	# Feature importance (top_n đã được xử lý trong get_feature_importance)
	# imp_df = trainer.get_feature_importance(top_n=15)
	# vis.plot_feature_importance(imp_df, save_path='plots/importance.png')

	logger.info("Process Completed.")