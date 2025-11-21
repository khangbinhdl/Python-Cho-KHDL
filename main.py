from Preprocessing import DataPreprocessor
from Visualize import EDA, ModelVisualize
from Model import ModelTrainer
import logging
from logging import StreamHandler, FileHandler, Formatter
from datetime import datetime
import os

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
    # Những bước này không phụ thuộc vào phân phối dữ liệu nên làm gộp được
    preprocessor = DataPreprocessor(missing_strategy='median', outlier_method='isolation_forest')
    preprocessor.load_data(file_path)
    
    # Drop cột rác và clean giá trị âm
    preprocessor.drop_features(['calories_from_fat', 'weight_watchers_pnts', 'company', 'item'])
    preprocessor.clean_negative_values()
    
    # One-hot encoding (Làm trước split để đảm bảo đồng bộ cột)
    preprocessor.encode_categorical(strategy='onehot')
    
    # Lấy dữ liệu tạm thời (đã clean cơ bản)
    current_data = preprocessor.get_processed_data()

    # =========================================================================
    # 2. CHIA DỮ LIỆU (Sử dụng ModelTrainer Class)
    # =========================================================================
    logger.info("Initializing ModelTrainer to split data...")
    trainer = ModelTrainer(random_state=42)
    
    # GỌI HÀM CỦA CLASS ĐỂ CHIA TRAIN/TEST (Thay vì dùng sklearn trực tiếp)
    train_df, test_df = trainer.split_data(current_data, test_size=0.2)

    # =========================================================================
    # 3. XỬ LÝ CHUYÊN SÂU & CHỐNG DATA LEAKAGE
    # =========================================================================
    logger.info("Processing Split Data (Preventing Leakage)...")

    # --- Xử lý tập TRAIN (FIT & TRANSFORM) ---
    # 1. Missing: Học median từ train -> điền vào train
    train_processed = preprocessor.handle_missing_values(data=train_df, fit=True)
    
    # 2. Outliers: Chỉ loại bỏ trên tập TRAIN
    train_processed = preprocessor.handle_outliers(
        data=train_processed, 
        exclude_features=[target_col], # Không xóa nếu ngoại lai nằm ở target (tùy chọn)
        outlier_strategy='isolation_forest'
    )
    
    # 3. Scaling: Học min/max/std từ train -> scale train
    train_processed = preprocessor.scale_features(
        data=train_processed, 
        exclude_features=[target_col], 
        fit=True
    )

    # --- Xử lý tập TEST (CHỈ TRANSFORM) ---
    # 1. Missing: Dùng median đã học từ train -> điền vào test
    test_processed = preprocessor.handle_missing_values(data=test_df, fit=False)
    
    # 2. Outliers: KHÔNG XỬ LÝ TRÊN TEST (để test phản ánh thực tế)
    
    # 3. Scaling: Dùng tham số đã học từ train -> scale test
    test_processed = preprocessor.scale_features(
        data=test_processed, 
        exclude_features=[target_col], 
        fit=False
    )

    # =========================================================================
    # 4. HUẤN LUYỆN & ĐÁNH GIÁ
    # =========================================================================
    
    # Nạp dữ liệu sạch ngược lại vào Trainer để tách X, y
    trainer.set_training_data(train_processed, test_processed, target_col=target_col)
    
    # Tối ưu hóa tham số (Demo cho RandomForest)
    rf_params = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    trainer.optimize_params('RandomForest', rf_params)
    
    # Huấn luyện
    trainer.train_all_models()
    
    # Đánh giá
    results = trainer.evaluate_models()
    
    # Lưu kết quả
    trainer.save_results("results.csv")
    trainer.save_best_model("models")
    
    # =========================================================================
    # 5. VISUALIZE
    # =========================================================================
    vis = ModelVisualize(results)
    vis.plot_model_comparison(save_path='plots/comparison.png')
    
    imp_df = trainer.get_feature_importance()
    if imp_df is not None:
        vis.plot_feature_importance(imp_df, top_n=15, save_path='plots/importance.png')

    logger.info("Process Completed.")