from Preprocessing import DataPreprocessor
from EDA import EDAData
import logging
from logging import StreamHandler, FileHandler, Formatter
from datetime import datetime
import os
import numpy as np

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
	# 1 Thiết lập logging
	log_file = setup_logging()

	file_path = './FastFoodNutritionMenuV3.csv'

	# 2 Tiền xử lý dữ liệu
	preprocessor = DataPreprocessor(missing_strategy='median',
									scaling_strategy='standard',
									outlier_method='iqr')
	preprocessor.load_data(file_path)
	processed_data = preprocessor.get_processed_data()

	# 3 EDA trước khi làm sạch
	# eda_before = EDAData(processed_data)
	# eda_before.perform_eda()

	# Dựa vào biểu có thể thấy cột calories_from_fat và total_fat_g có mối tương quan rất cao (1)
	# Nên ta sẽ loại bỏ một trong hai cột này để tránh đa cộng tuyến
	# Thực tế calcories_from_fat = total_fat_g * 9

	# Ngoài ra, còn có weight_watchers_pnts và calories (cột target) cũng có tương quan khá cao (1)
	# Nên ta cũng sẽ loại bỏ weight_watchers_pnts, vì calories là cột mục tiêu cần dự đoán
	# Thực tế weight_watchers_pnts được tính dựa trên calories

	# Ngoài ra ta sẽ bỏ 2 cột company, item vì không cần thiết cho huấn luyện mô hình

	# 4 Làm sạch cơ bản
	preprocessor.drop_features(['calories_from_fat', 'weight_watchers_pnts', 'company', 'item'])
	preprocessor.clean_negative_values()

	processed_data = preprocessor.get_processed_data()

	# 5 EDA sau khi làm sạch
	# eda_after = EDAData(processed_data)
	# eda_after.summary_statistics()
	# eda_after.correlation_analysis()

	# Khúc class Model
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
	
	train, test = train_test_split(processed_data, test_size=0.2, random_state=42)
	
	# 6 Áp dụng tiền xử lý lên tập huấn luyện (FIT)

	# Cập nhật lại danh sách cột cho preprocessor
	preprocessor.data = train.copy()
	preprocessor.auto_detect_columns()
	
	# Xử lý missing values (FIT trên train)
	train_processed = preprocessor.handle_missing_values(
		data=train,
		num_strategy='median',
		fit=True
	)
	
	# Xử lý outliers
	train_processed = preprocessor.handle_outliers(
		data=train_processed,
		exclude_features=['trans_fat_g', 'calories'],
		outlier_strategy='isolation_forest',
	)
	
	train_processed = preprocessor.scale_features(
		data=train_processed,
		exclude_features=['calories'],
		fit=True
	)

	X_train = train_processed.drop(columns=['calories'])
	y_train = train_processed['calories']
	
	# Xử lý missing values (TRANSFORM trên test)
	test = preprocessor.handle_missing_values(
		data=test,
		num_strategy='median',
		fit=False
	)
	
	# Scale features (TRANSFORM trên test)
	test = preprocessor.scale_features(
		data=test,
		exclude_features=['calories'],
		fit=False
	)
	
	X_test = test.drop(columns=['calories'])
	y_test = test['calories']


	print("train shape:", X_train.shape)
	print("test shape:", X_test.shape)

	model = LinearRegression()
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	mae = mean_absolute_error(y_test, y_pred)

	print("Mean Squared Error (MSE):", mse)
	print("R-squared (R2 ):", r2)
	print("Mean Absolute Error (MAE):", mae)

