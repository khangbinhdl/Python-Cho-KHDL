from Preprocessing import DataPreprocessor
from EDA import EDA
import logging
from logging import StreamHandler, FileHandler, Formatter
from datetime import datetime
import os

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
	# eda_before = EDA(processed_data)
	# eda_before.perform_eda()

	# Dựa vào biểu có thể thấy cột calories_from_fat và total_fat_g có mối tương quan rất cao (1)
	# Nên ta sẽ loại bỏ một trong hai cột này để tránh đa cộng tuyến
	# Thực tế calcories_from_fat = total_fat_g * 9

	# Ngoài ra, còn có weight_watchers_pnts và calories (cột target) cũng có tương quan khá cao (1)
	# Nên ta cũng sẽ loại bỏ weight_watchers_pnts, vì calories là cột mục tiêu cần dự đoán
	# Thực tế weight_watchers_pnts được tính dựa trên calories

	# Ngoài ra ta sẽ bỏ 2 cột company, item vì không cần thiết cho huấn luyện mô hình

	# 4 Làm sạch tiếp
	preprocessor.drop_features(['calories_from_fat', 'weight_watchers_pnts', 'company', 'item'])
	preprocessor.clean_negative_values()
	preprocessor.handle_missing_values(num_strategy='median', cat_strategy='drop', dt_strategy='drop')
	preprocessor.handle_outliers(exclude_features=['trans_fat_g'], outlier_strategy='clip')
	preprocessor.scale_features()
	processed_data = preprocessor.get_processed_data()

	# 5 EDA sau khi làm sạch
	eda_after = EDA(processed_data)
	eda_after.perform_eda()
	
    # 6 Xuất ra file đã xử lý
	processed_data.to_csv('cleaned_dataset.csv', index=False)
