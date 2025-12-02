# Fast Food Nutrition Prediction

Dự đoán lượng Calories của các món ăn nhanh dựa trên thông tin dinh dưỡng.

## Link dataset
[Kaggle - Nutritional Fast Food Dataset](https://www.kaggle.com/datasets/tan5577/nutritonal-fast-food-dataset)

## Cấu trúc Project

```
Python-Cho-KHDL/
│
├── src/                        # Source code chính
│   ├── data/                   # Data preprocessing
│   │   ├── io.py              # Load/save data
│   │   ├── transformer.py     # Data transformations
│   │   └── preprocessor.py    # Orchestrator
│   ├── models/                # Model training & evaluation
│   │   ├── trainer.py         # Main trainer
│   │   ├── evaluator.py       # Metrics calculation
│   │   ├── io.py              # Model I/O
│   │   └── optimizer.py       # Hyperparameter tuning
│   ├── visualization/         # Plotting
│   │   ├── eda.py             # Exploratory Data Analysis
│   │   └── model_plots.py     # Model visualization
│   └── utils/                 # Utilities
│       ├── logging.py         # Centralized logging
│       └── config.py          # Config loading
│
├── scripts/                   # Entry point scripts
│   ├── train.py              # Main training pipeline
│   └── experiment.py         # Grid search experiments
│
├── configs/                   # Configuration files
│   ├── default_config.ini    # Cấu hình mặc định
│   └── config.ini            # Cấu hình tùy chỉnh (optional)
│
├── data/                      # Data directory
│   ├── raw/                   # Original data
│   ├── processed/             # Cleaned data
│   └── interim/               # Intermediate files
│
├── outputs/                   # Generated outputs
│   ├── logs/                  # Log files
│   ├── models/                # Saved models (.joblib)
│   │   ├── LinearRegression.joblib
│   │   ├── Ridge.joblib
│   │   ├── Lasso.joblib
│   │   ├── ElasticNet.joblib
│   │   ├── RandomForest.joblib
│   │   └── LightGBM.joblib
│   ├── plots/                 # Visualizations
│   │   ├── eda/              # EDA plots
│   │   │   ├── before/       # Trước preprocessing
│   │   │   │   ├── boxplot_all.png
│   │   │   │   ├── correlation_heatmap.png
│   │   │   │   ├── distribution_all.png
│   │   │   │   └── missing_values.png
│   │   │   └── after/        # Sau preprocessing
│   │   │       ├── boxplot_all.png
│   │   │       ├── correlation_heatmap.png
│   │   │       └── distribution_all.png
│   │   ├── model_comparison.png
│   │   ├── model_comparison_all.png
│   │   ├── feature_importance.png
│   │   └── feature_importance_all_models.png
│   └── results/               # Evaluation results
│       ├── evaluation_results.csv
│       ├── evaluation_results.json
│       └── experiment_results.xlsx
│
├── docs/                      # Documentation
│   └── report/                # LaTeX report
│
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Setup và Installation

### Yêu cầu hệ thống
- Python 3.10+
- pip
- Git

### Bước 1: Clone repository
```bash
git clone https://github.com/khangbinhdl/Python-Cho-KHDL.git   
cd "Python-Cho-KHDL/Cuối kì"
```

### Bước 2: Setup virtual environment
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Bước 3: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

### 1. Chạy với cấu hình mặc định
```bash
python scripts/train.py
```

### 2. Chạy với file config tùy chỉnh
```bash
python scripts/train.py --config configs/my_config.ini
```

### 3. Chạy với hyperparameter optimization
```bash
python scripts/train.py --optimize
```

### 4. Chạy không có EDA visualization
```bash
python scripts/train.py --no-eda
```

### 5. Chạy với model cụ thể
```bash
# Chạy chỉ RandomForest và LightGBM
python scripts/train.py --models "RandomForest,LightGBM"

# Chạy chỉ Ridge regression
python scripts/train.py --models "Ridge"
```

### 6. Chạy grid search experiment
```bash
python scripts/experiment.py
```

### 7. Kết hợp nhiều tham số
```bash
python scripts/train.py \
    --optimize \
    --no-eda \
    --test-size 0.25 \
    --random-state 999 \
    --models "RandomForest,LightGBM"
```

## Các tham số dòng lệnh

| Tham số | Mô tả | Kiểu | Mặc định |
|---------|-------|------|----------|
| `--config` | Đường dẫn file config | str | `configs/default_config.ini` |
| `--data` | Đường dẫn file dữ liệu | str | Từ config |
| `--target` | Tên cột target | str | `calories` |
| `--test-size` | Tỷ lệ test set (0.0-1.0) | float | 0.2 |
| `--random-state` | Random seed | int | 42 |
| `--models` | Models cần train | str | `all` |
| `--optimize` | Bật optimization | flag | False |
| `--no-eda` | Tắt EDA plots | flag | False |
| `--no-viz` | Tắt tất cả visualization | flag | False |
| `--drop-features` | Các features cần loại bỏ | str | Từ config |
| `--clean-negative` | Xử lý giá trị âm | bool | True |
| `--categorical-encoding` | Phương pháp encoding | str | `onehot` |

## Outputs

### Logs
```
outputs/logs/pipeline_YYYYMMDD_HHMMSS.log
```
Chứa toàn bộ log của pipeline execution.

### Models
```
outputs/models/
├── LinearRegression.joblib
├── Ridge.joblib
├── Lasso.joblib
├── ElasticNet.joblib
├── RandomForest.joblib
└── LightGBM.joblib
```

### Plots
```
outputs/plots/
├── eda/
│   ├── before/
│   │   ├── boxplot_all.png
│   │   ├── correlation_heatmap.png
│   │   ├── distribution_all.png
│   │   └── missing_values.png
│   └── after/
│       ├── boxplot_all.png
│       ├── correlation_heatmap.png
│       └── distribution_all.png
├── model_comparison.png
├── model_comparison_all.png
├── feature_importance.png
└── feature_importance_all_models.png
```

### Results
```
outputs/results/
├── evaluation_results.csv      # CSV format
├── evaluation_results.json     # JSON format
└── experiment_results.xlsx     # Excel format (grid search)
```


## Mô tả dữ liệu

Dataset chứa **515 mẫu** với **14 features**:

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