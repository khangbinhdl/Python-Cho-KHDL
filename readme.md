# Link dataset: 
[Kaggle](https://www.kaggle.com/datasets/tan5577/nutritonal-fast-food-dataset)

# Mô tả dữ liệu
| Cột | Mô tả |
|:----|:-----|
| **Company** | Tên công ty/thương hiệu sản xuất mặt hàng. |
| **Item** | Tên của sản phẩm/món ăn cụ thể. |
| **Calories** | Tổng lượng calo (năng lượng) trong một khẩu phần sản phẩm. |
| **Calories from Fat** | Lượng calo đến từ chất béo trong một khẩu phần. |
| **Total Fat (g)** | Tổng lượng chất béo (gram) trong một khẩu phần. |
| **Saturated Fat (g)** | Lượng chất béo bão hòa (gram) trong một khẩu phần. |
| **Trans Fat (g)** | Lượng chất béo chuyển hóa (trans fat - gram) trong một khẩu phần. |
| **Cholesterol (mg)** | Lượng Cholesterol (miligram) trong một khẩu phần. |
| **Sodium (mg)** | Lượng Natri/Muối (miligram) trong một khẩu phần. |
| **Carbs (g)** | Tổng lượng Carbohydrate (gram) trong một khẩu phần. |
| **Fiber (g)** | Lượng chất xơ (gram) trong một khẩu phần. |
| **Sugars (g)** | Lượng đường (gram) trong một khẩu phần. |
| **Protein (g)** | Lượng Protein (gram) trong một khẩu phần. |
| **Weight Watchers Pnts** | Điểm số theo hệ thống tính điểm của chương trình ăn kiêng Weight Watchers (có thể đã lỗi thời hoặc chỉ áp dụng cho một số thị trường). |

# Setup và Installation

```bash
# Bước 1: Clone
git clone https://github.com/khangbinhdl/Python-Cho-KHDL.git   
cd "Python-Cho-KHDL"

# Bước 2: Setup environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

# Hướng dẫn sử dụng Pipeline

## Cấu trúc file config.ini

File `config.ini` chứa tất cả các tham số cấu hình cho pipeline:

```ini
[PATHS]
data_file = Data/FastFoodNutritionMenuV3.csv
temp_data_file = Data/temp_processed_data.csv

[DATA]
target_column = calories
test_size = 0.2
random_state = 42

[PREPROCESSING]
num_strategy = drop
cat_strategy = mode
dt_strategy = drop
scaling_strategy = standard
outlier_method = zscore

[MODEL]
selected_models = all
available_models = RandomForest,LightGBM,Ridge,Lasso,ElasticNet,LinearRegression

[OPTIMIZATION]
enable_optimization = false
n_trials = 50
n_jobs = 3
models_to_optimize = RandomForest,LightGBM,Ridge,Lasso,ElasticNet

[VISUALIZATION]
enable_eda = true
enable_plots = true
feature_importance_top_n = 15

[OUTPUT]
results_csv = results/evaluation_results.csv
results_json = results/evaluation_results.json
comparison_plot = plots/comparison.png
importance_plot = plots/importance.png
```

## Cách chạy pipeline

### 1. Chạy với cấu hình mặc định
```bash
python pipeline.py
```

### 2. Chạy với file config tùy chỉnh (nếu có)
```bash
# Nếu bạn tạo file config riêng
python pipeline.py --config custom_config.ini
```

### 3. Chạy với optimization enabled
```bash
python pipeline.py --optimize
```

### 4. Chạy không có EDA
```bash
python pipeline.py --no-eda
```

### 5. Chạy không có visualization
```bash
python pipeline.py --no-viz
```

### 6. Chạy với file dữ liệu khác
```bash
python pipeline.py --data "path/to/other/data.csv"
```

### 7. Chạy với target column khác
```bash
python pipeline.py --target "protein"
```

### 8. Chạy với test size khác
```bash
python pipeline.py --test-size 0.3
```

### 9. Chạy với random state khác
```bash
python pipeline.py --random-state 123
```

### 10. Chạy với model cụ thể
```bash
# Chạy chỉ RandomForest và LightGBM
python pipeline.py --models "RandomForest,LightGBM"

# Chạy chỉ Ridge regression
python pipeline.py --models "Ridge"

# Chạy tất cả models (mặc định)
python pipeline.py --models "all"
```

### 11. Kết hợp nhiều tham số
```bash
python pipeline.py --optimize --no-eda --test-size 0.25 --random-state 999
```

## Các tham số dòng lệnh

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--config` | Đường dẫn file config | `config.ini` |
| `--data` | Đường dẫn file dữ liệu | Từ config |
| `--target` | Tên cột target | Từ config |
| `--optimize` | Bật optimization | False |
| `--no-eda` | Tắt EDA | False |
| `--no-viz` | Tắt visualization | False |
| `--test-size` | Tỷ lệ test set | Từ config |
| `--random-state` | Random state | Từ config |
| `--models` | Models cần train | Từ config |

## Ưu tiên tham số

1. Tham số dòng lệnh (cao nhất)
2. File config
3. Giá trị mặc định trong code (thấp nhất)

## Ví dụ thực tế

### Chạy experiment nhanh (không EDA, không visualization)
```bash
python pipeline.py --no-eda --no-viz
```

### Chạy với optimization đầy đủ
```bash
python pipeline.py --optimize
```

### Thử nghiệm với dữ liệu khác
```bash
python pipeline.py --data "Data/new_dataset.csv" --target "new_target"
```

### So sánh hiệu suất các model
```bash
# Chạy chỉ các tree-based models
python pipeline.py --models "RandomForest,LightGBM" --optimize

# Chạy chỉ các linear models
python pipeline.py --models "Ridge,Lasso,ElasticNet,LinearRegression"

# Test nhanh với 1 model
python pipeline.py --models "RandomForest" --no-eda --no-viz
```

