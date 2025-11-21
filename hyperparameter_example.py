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
    
    # Khởi tạo ModelTrainer và split dữ liệu 1 lần duy nhất
    trainer = ModelTrainer(random_state=42)
    trainer.load_data(clean_data, target_column='calories')
    trainer.split_data(test_size=0.2)
    
    # Lấy train và test data từ ModelTrainer
    train_data = pd.concat([trainer.X_train, trainer.y_train], axis=1)
    test_data = pd.concat([trainer.X_test, trainer.y_test], axis=1)
    
    print(f"Chia dữ liệu - Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Áp dụng tiền xử lý cho dữ liệu train (FIT)
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
    
    print(f"Kích thước Train: {train_processed.shape}, Test: {test_processed.shape}")
    
    # Cập nhật dữ liệu đã xử lý vào trainer (không cần split lại)
    trainer.X_train = train_processed.drop(columns=['calories'])
    trainer.X_test = test_processed.drop(columns=['calories'])
    trainer.y_train = train_processed['calories']
    trainer.y_test = test_processed['calories']
    
    # Khởi tạo các models
    trainer.initialize_models()
    
    # Chỉ train ElasticNet với tham số mặc định để có baseline
    trainer.train_models(['ElasticNet'])
    trainer.evaluate_models()
    
    print("\n" + "="*50)
    print("KẾT QUẢ ELASTICNET VỚI THAM SỐ MẶC ĐỊNH")
    print("="*50)
    for result in trainer.results:
        print(f"{result['model_name']}: R² = {result['r2_score']:.4f}")
    
    # Tối ưu siêu tham số cho ElasticNet (model tốt nhất từ main2.py)
    print("\n" + "="*50)
    print("TỐI ƯU SIÊU THAM SỐ CHO ELASTICNET")
    print("="*50)
    
    elasticnet_param_grid = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_iter': [1000, 2000, 3000, 5000],
        'selection': ['cyclic', 'random'],
        'tol': [1e-4, 1e-3, 1e-2]
    }
    
    # Sử dụng GridSearchCV để tìm tham số tốt nhất
    best_elasticnet = trainer.hyperparameter_tuning(
        model_name='ElasticNet',
        param_grid=elasticnet_param_grid,
        cv=5,
        scoring='r2',
        search_type='grid'  # Dùng grid search để tìm chính xác
    )
    
    if best_elasticnet:
        # Huấn luyện lại với tham số đã tối ưu
        trainer.train_models(['ElasticNet'])
        trainer.evaluate_models()
        
        print("\n" + "="*50)
        print("SO SÁNH KẾT QUẢ TRƯỚC VÀ SAU TỐI ƯU")
        print("="*50)
        # In kết quả cuối cùng (chỉ có Lasso baseline và Lasso optimized)
        for result in trainer.results:
            print(f"{result['model_name']}: R² = {result['r2_score']:.4f}")
    
    # Lưu kết quả
    trainer.save_results(filepath='results/elasticnet_hypertuned_results.csv')
    trainer.save_model(filepath='models/best_elasticnet_model.pkl')
    
    print("\n" + "="*50)
    print("TỐI ƯU SIÊU THAM SỐ ELASTICNET HOÀN THÀNH!")
    print(f"Model tốt nhất: {trainer.best_model_name}")
    print(f"Điểm R² tốt nhất: {max(trainer.results, key=lambda x: x['r2_score'])['r2_score']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()