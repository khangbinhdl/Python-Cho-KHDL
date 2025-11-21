from Preprocessing import DataPreprocessor
from EDA import EDAData
from model import ModelTrainer
import logging
from logging import StreamHandler, FileHandler, Formatter
from datetime import datetime
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def setup_logging():
	# Tạo thư mục logs nếu chưa có
	os.makedirs("logs", exist_ok=True)

	# Tạo tên file log theo ngày giờ, ví dụ: pipeline_2025-11-11_21-45-30.log
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	log_path = os.path.join("logs", f"pipeline_{timestamp}.log")

	# Xóa handler cũ nếu đã tồn tại (tránh nhân log khi rerun)
	root = logging.getLogger()
	for h in list(root.handlers):
		root.removeHandler(h)

	root.setLevel(logging.INFO)

	# === Console handler ===
	ch = StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(Formatter("%(asctime)s [%(name)s]: %(message)s", datefmt="%H:%M:%S"))

	# === File handler ===
	fh = FileHandler(log_path, mode="w", encoding="utf-8")
	fh.setLevel(logging.INFO)
	fh.setFormatter(Formatter("%(asctime)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

	root.addHandler(ch)
	root.addHandler(fh)

	logging.getLogger().info(f"Logging initialized → {log_path}")
	return log_path

if __name__ == "__main__":
	# 1. Thiết lập logging
	log_file = setup_logging()
	# file_path = './FastFoodNutritionMenuV3.csv'

	file_path = 'D:\\KHDL - HCMUS\\Năm 3\\Python KHDL\\Project2\\Python-Cho-KHDL\\FastFoodNutritionMenuV3.csv'

	# 2. Nạp và làm sạch dữ liệu cơ bản
	preprocessor = DataPreprocessor(missing_strategy='median',
									scaling_strategy='standard',
									outlier_method='iqr')
	preprocessor.load_data(file_path)

	# 3. EDA trước khi làm sạch (optional)
	# eda_before = EDAData(preprocessor.get_processed_data())
	# eda_before.perform_eda()

	# 4. Loại bỏ các cột không cần thiết
	# Dựa vào EDA: calories_from_fat có tương quan cao với total_fat_g
	# weight_watchers_pnts có tương quan cao với calories (target)
	preprocessor.drop_features(['calories_from_fat', 'weight_watchers_pnts', 'company', 'item'])
	preprocessor.clean_negative_values()

	clean_data = preprocessor.get_processed_data()
	
	# 5. EDA sau khi làm sạch (optional)
	# eda_after = EDAData(clean_data)
	# eda_after.summary_statistics()
	# eda_after.correlation_analysis()

	# 6. CHIA DỮ LIỆU TRƯỚC KHI XỬ LÝ (Tránh data leakage)
	train_data, test_data = train_test_split(clean_data, test_size=0.2, random_state=42)
	
	# 7. Áp dụng preprocessing trên train (FIT) và test (TRANSFORM)
	# Cập nhật lại danh sách cột cho preprocessor
	preprocessor.data = train_data.copy()
	preprocessor.auto_detect_columns()
	
	# Xử lý missing values (FIT trên train)
	train_processed = preprocessor.handle_missing_values(
		data=train_data,
		num_strategy='median',
		fit=True
	)
	
	# Xử lý outliers chỉ trên TRAIN (không áp dụng cho test)
	train_processed = preprocessor.handle_outliers(
		data=train_processed,
		exclude_features=['trans_fat_g', 'calories'],
		outlier_strategy='drop',  # Đổi từ isolation_forest sang drop
	)
	
	# Scaling features (FIT trên train)
	train_processed = preprocessor.scale_features(
		data=train_processed,
		exclude_features=['calories'],
		fit=True
	)

	# Xử lý test data (chỉ TRANSFORM)
	# Xử lý missing values (TRANSFORM trên test)
	test_processed = preprocessor.handle_missing_values(
		data=test_data,
		num_strategy='median',
		fit=False
	)
	
	# KHÔNG xử lý outliers trên test để tránh data leakage
	
	# Scale features (TRANSFORM trên test)  
	test_processed = preprocessor.scale_features(
		data=test_processed,
		exclude_features=['calories'],
		fit=False
	)

	# 8. Chuẩn bị dữ liệu cho ModelTrainer
	# Gộp lại train và test đã được xử lý
	processed_data = pd.concat([train_processed, test_processed], ignore_index=True)
	
	print(f"Original data shape: {clean_data.shape}")
	print(f"Processed data shape: {processed_data.shape}")
	print(f"Train processed shape: {train_processed.shape}")
	print(f"Test processed shape: {test_processed.shape}")

	# 9. Sử dụng ModelTrainer
	trainer = ModelTrainer(random_state=42)
	
	# Nạp dữ liệu đã xử lý
	trainer.load_data(processed_data, target_column='calories')
	
	# Chia lại dữ liệu với cùng tỷ lệ và random_state
	trainer.split_data(test_size=0.2)
	
	# Chạy full pipeline
	summary = trainer.run_full_pipeline(
		test_size=0.2,
		tune_best_model=True,  # Tối ưu siêu tham số cho model tốt nhất
		save_artifacts=True    # Lưu model và kết quả
	)
	
	# In kết quả tổng kết
	print("\n" + "="*50)
	print("PIPELINE SUMMARY")
	print("="*50)
	for key, value in summary.items():
		print(f"{key}: {value}")
	print("="*50)

