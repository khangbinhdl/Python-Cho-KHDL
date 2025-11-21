"""
Minh họa cách sử dụng ModelTrainer.hyperparameter_tuning() method
"""

from Preprocessing import DataPreprocessor
from model import ModelTrainer
import logging
from logging import StreamHandler, FileHandler, Formatter
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def setup_logging():
    """Thiết lập cấu hình logging"""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("logs", f"hyperparameter_tuning_{timestamp}.log")

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(logging.INFO)

    ch = StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(Formatter("%(asctime)s [%(name)s]: %(message)s", datefmt="%H:%M:%S"))

    fh = FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(Formatter("%(asctime)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    root.addHandler(ch)
    root.addHandler(fh)

    logging.getLogger().info(f"Logging đã khởi tạo → {log_path}")
    return log_path

def main():
    # Thiết lập logging
    setup_logging()
    
    # Tải và tiền xử lý dữ liệu
    # file_path = './FastFoodNutritionMenuV3.csv'
    file_path = 'D:\\KHDL - HCMUS\\Năm 3\\Python KHDL\\Project2\\Python-Cho-KHDL\\FastFoodNutritionMenuV3.csv'
    
    preprocessor = DataPreprocessor(missing_strategy='median',
                                  scaling_strategy='standard',
                                  outlier_method='iqr')
    preprocessor.load_data(file_path)
    
    # Loại bỏ các đặc trưng không cần thiết
    preprocessor.drop_features(['calories_from_fat', 'weight_watchers_pnts', 'company', 'item'])
    preprocessor.clean_negative_values()
    
    clean_data = preprocessor.get_processed_data()
    
    # Chia dữ liệu thành train/val/test (60/20/20) để tránh data leakage
    # Bước 1: tách test set (20%)
    train_val_data, test_data = train_test_split(clean_data, test_size=0.2, random_state=42)
    
    # Bước 2: tách train và validation từ 80% còn lại
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2 của tổng
    
    print(f"Chia dữ liệu - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Áp dụng tiền xử lý cho dữ liệu train (FIT)
    preprocessor.data = train_data.copy()
    preprocessor.auto_detect_columns()
    
    train_processed = preprocessor.handle_missing_values(
        data=train_data,
        num_strategy='median',
        fit=True
    )
    
    train_processed = preprocessor.handle_outliers(
        data=train_processed,
        exclude_features=['trans_fat_g', 'calories'],
        outlier_strategy='drop',
    )
    
    train_processed = preprocessor.scale_features(
        data=train_processed,
        exclude_features=['calories'],
        fit=True
    )
    
    # Áp dụng tiền xử lý cho dữ liệu validation (chỉ TRANSFORM)
    val_processed = preprocessor.handle_missing_values(
        data=val_data,
        num_strategy='median',
        fit=False
    )
    
    val_processed = preprocessor.scale_features(
        data=val_processed,
        exclude_features=['calories'],
        fit=False
    )
    
    # Áp dụng tiền xử lý cho dữ liệu test (chỉ TRANSFORM)
    test_processed = preprocessor.handle_missing_values(
        data=test_data,
        num_strategy='median',
        fit=False
    )
    
    test_processed = preprocessor.scale_features(
        data=test_processed,
        exclude_features=['calories'],
        fit=False
    )
    
    # Kết hợp train+val cho việc huấn luyện model và tối ưu siêu tham số
    train_val_data = pd.concat([train_processed, val_processed], ignore_index=True)
    
    print(f"Kích thước Train+Val: {train_val_data.shape}, Test: {test_processed.shape}")
    
    # Khởi tạo ModelTrainer với dữ liệu train+val
    trainer = ModelTrainer(random_state=42)
    trainer.load_data(train_val_data, target_column='calories')
    
    # Tạo validation split thủ công cho tối ưu siêu tham số
    # Sử dụng dữ liệu validation đã tách riêng
    X_train = train_processed.drop(columns=['calories'])
    y_train = train_processed['calories']
    X_val = val_processed.drop(columns=['calories'])
    y_val = val_processed['calories']
    X_test = test_processed.drop(columns=['calories'])
    y_test = test_processed['calories']
    
    # Với ModelTrainer, sử dụng train+val kết hợp và để nó chia lại
    trainer.split_data(test_size=0.25)  # Tách validation set
    
    # Khởi tạo các models
    trainer.initialize_models()
    
    # Huấn luyện một số models nhanh trước để so sánh
    quick_models = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTree']
    trainer.train_models(quick_models)
    trainer.evaluate_models()
    
    print("\n" + "="*50)
    print("KẾT QUẢ CÁC MODELS NHANH")
    print("="*50)
    for result in trainer.results:
        print(f"{result['model_name']}: R² = {result['r2_score']:.4f}")
    
    # Bây giờ huấn luyện các ensemble models
    ensemble_models = ['RandomForest', 'ExtraTrees', 'GradientBoosting']
    trainer.train_models(ensemble_models)
    trainer.evaluate_models()
    
    print("\n" + "="*50)
    print("KẾT QUẢ TẤT CẢ MODELS")
    print("="*50)
    for result in trainer.results:
        print(f"{result['model_name']}: R² = {result['r2_score']:.4f}")
    
    # Tối ưu siêu tham số cho ExtraTrees (model hoạt động tốt nhất)
    print("\n" + "="*50)
    print("TỐI ƯU SIÊU THAM SỐ CHO EXTRA TREES")
    print("="*50)
    
    et_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Sử dụng RandomizedSearchCV để tối ưu nhanh hơn
    best_et = trainer.hyperparameter_tuning(
        model_name='ExtraTrees',
        param_grid=et_param_grid,
        cv=5,
        scoring='r2',
        search_type='random'  # Nhanh hơn grid search
    )
    
    if best_et:
        # Huấn luyện lại với tham số đã tối ưu
        trainer.train_models(['ExtraTrees'])
        trainer.evaluate_models()
        
        print("\n" + "="*50)
        print("KẾT QUẢ CUỐI CÙNG SAU TỐI ƯU")
        print("="*50)
        for result in trainer.results:
            print(f"{result['model_name']}: R² = {result['r2_score']:.4f}")
    
    # Vẽ biểu đồ so sánh
    trainer.plot_model_comparison(save_path='plots/hypertuned_model_comparison.png')
    
    # Hiển thị feature importance cho model tree-based tốt nhất
    try:
        trainer.plot_feature_importance(save_path='plots/hypertuned_feature_importance.png')
    except Exception as e:
        print(f"Không thể vẽ feature importance: {e}")
    
    # Lưu kết quả
    trainer.save_results(filepath='results/hypertuned_results.csv')
    trainer.save_model(filepath='models/best_hypertuned_model.pkl')
    
    print("\n" + "="*50)
    print("TỐI ƯU SIÊU THAM SỐ HOÀN THÀNH!")
    print(f"Model tốt nhất: {trainer.best_model_name}")
    print(f"Điểm R² tốt nhất: {max(trainer.results, key=lambda x: x['r2_score'])['r2_score']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()