"""
Minh họa cách áp dụng Bayesian optimization cho Lasso (model tốt nhất)
"""

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import joblib
import logging
from datetime import datetime
import os

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from Preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class OptunaTuner:
    """
    Tối ưu hóa siêu tham số nâng cao sử dụng Optuna Bayesian Optimization cho ElasticNet
    
    Cung cấp tìm kiếm siêu tham số tinh vi với pruning và 
    khả năng thực thi song song.
    
    Thuộc tính
    ----------
    X_train : array-like
        Đặc trưng huấn luyện
    y_train : array-like  
        Mục tiêu huấn luyện
    random_state : int
        Seed ngẫu nhiên để tái tạo kết quả
    study : optuna.Study
        Đối tượng Optuna study
    best_params : dict
        Tham số tốt nhất được tìm thấy
    best_score : float
        Điểm số tốt nhất đạt được
    """
    
    def __init__(self, X_train, y_train, random_state=42):
        """
        Khởi tạo OptunaTuner với dữ liệu huấn luyện
        
        Tham số
        -------
        X_train : array-like
            Đặc trưng huấn luyện
        y_train : array-like
            Mục tiêu huấn luyện  
        random_state : int, tùy chọn
            Seed ngẫu nhiên. Mặc định là 42
        """
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.study = None
        self.best_params = None
        self.best_score = None
        
        # Thiết lập logging
        self.logger = logging.getLogger("OPTUNA_TUNER")
        
    def elasticnet_objective(self, trial):
        """
        Hàm mục tiêu cho tối ưu hóa ElasticNet
        
        Tham số
        -------
        trial : optuna.Trial
            Đối tượng Optuna trial
            
        Trả về
        ------
        float
            Điểm RMSE (để tối thiểu hóa)
        """
        # Đề xuất siêu tham số cho ElasticNet
        alpha = trial.suggest_float("alpha", 0.001, 100.0, log=True)  # Regularization strength
        l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)  # Mix ratio between L1 and L2 penalties
        max_iter = trial.suggest_int("max_iter", 1000, 10000)  # Maximum iterations
        tol = trial.suggest_float("tol", 1e-5, 1e-2, log=True)  # Tolerance for optimization
        selection = trial.suggest_categorical("selection", ['cyclic', 'random'])  # Selection method
        
        # Tạo model với tham số được đề xuất
        elasticnet = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            selection=selection,
            random_state=self.random_state
        )
        
        # Thực hiện cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = []
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(self.X_train)):
            X_fold_train = self.X_train.iloc[train_idx] if isinstance(self.X_train, pd.DataFrame) else self.X_train[train_idx]
            X_fold_valid = self.X_train.iloc[valid_idx] if isinstance(self.X_train, pd.DataFrame) else self.X_train[valid_idx]
            y_fold_train = self.y_train.iloc[train_idx] if isinstance(self.y_train, pd.Series) else self.y_train[train_idx]
            y_fold_valid = self.y_train.iloc[valid_idx] if isinstance(self.y_train, pd.Series) else self.y_train[valid_idx]
            
            # Huấn luyện và dự đoán
            elasticnet.fit(X_fold_train, y_fold_train)
            y_pred = elasticnet.predict(X_fold_valid)
            
            # Tính RMSE
            rmse = np.sqrt(mean_squared_error(y_fold_valid, y_pred))
            scores.append(rmse)
            
            # Báo cáo kết quả trung gian cho pruning
            trial.report(np.mean(scores), step=fold)
            
            # Kiểm tra xem trial có nên bị cắt tỉa không
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)  # Trả về RMSE trung bình (để tối thiểu hóa)
    
    def optimize_elasticnet(self, n_trials=100, timeout=3600):
        """
        Tối ưu hóa siêu tham số ElasticNet sử dụng Optuna
        
        Tham số
        -------
        n_trials : int, tùy chọn
            Số lượng trial tối đa. Mặc định là 100
        timeout : int, tùy chọn
            Giới hạn thời gian tính bằng giây. Mặc định là 3600 (1 giờ)
            
        Trả về
        ------
        dict
            Tham số tốt nhất được tìm thấy
        """
        self.logger.info(f"Bắt đầu tối ưu hóa Optuna với {n_trials} trials...")
        
        # Tạo study
        self.study = optuna.create_study(
            study_name=f"elasticnet_nutrition_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction="minimize",  # Tối thiểu hóa RMSE
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )
        
        # Tối ưu hóa
        self.study.optimize(
            self.elasticnet_objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=-1  # Sử dụng tất cả CPU cores
        )
        
        # Lưu trữ kết quả
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info(f"Tối ưu hóa hoàn thành!")
        self.logger.info(f"RMSE tốt nhất: {self.best_score:.4f}")
        self.logger.info(f"Tham số tốt nhất: {self.best_params}")
        
        return self.best_params
    
    def get_best_model(self):
        """
        Lấy model ElasticNet tốt nhất với tham số được tối ưu hóa
        
        Trả về
        ------
        ElasticNet
            Model được tối ưu hóa đã huấn luyện trên dữ liệu training
        """
        if self.best_params is None:
            raise ValueError("Chưa thực hiện tối ưu hóa. Gọi optimize_elasticnet() trước.")
            
        # Tạo và huấn luyện model tốt nhất
        best_elasticnet = ElasticNet(
            **self.best_params,
            random_state=self.random_state
        )
        
        best_elasticnet.fit(self.X_train, self.y_train)
        return best_elasticnet
    
    def save_study(self, filepath=None):
        """
        Lưu Optuna study để phân tích sau này
        
        Tham số
        -------
        filepath : str, tùy chọn
            Đường dẫn để lưu study. Nếu None, tự động tạo tên file
            
        Trả về
        ------
        str
            Đường dẫn đến study đã lưu
        """
        if self.study is None:
            raise ValueError("Không có study để lưu. Chạy tối ưu hóa trước.")
            
        os.makedirs("optuna_studies", exist_ok=True)
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"optuna_studies/elasticnet_study_{timestamp}.pkl"
            
        joblib.dump(self.study, filepath)
        self.logger.info(f"Study đã lưu vào: {filepath}")
        return filepath
    
    def plot_optimization_history(self, save_path=None):
        """
        Vẽ biểu đồ lịch sử tối ưu hóa sử dụng plotting tích hợp của Optuna
        
        Tham số
        -------
        save_path : str, tùy chọn
            Đường dẫn để lưu biểu đồ. Nếu None, chỉ hiển thị
        """
        if self.study is None:
            raise ValueError("Không có study để vẽ biểu đồ.")
            
        try:
            import optuna.visualization as vis
            import plotly.io as pio
            
            # Tạo biểu đồ lịch sử tối ưu hóa
            fig = vis.plot_optimization_history(self.study)
            fig.update_layout(title="Lịch Sử Tối Ưu Hóa Optuna")
            
            if save_path:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'plots', exist_ok=True)
                pio.write_image(fig, save_path)
                self.logger.info(f"Lịch sử tối ưu hóa đã lưu vào: {save_path}")
            
            fig.show()
            
        except ImportError:
            self.logger.warning("Plotly không khả dụng. Không thể tạo biểu đồ tối ưu hóa.")
    
    def get_param_importances(self):
        """
        Lấy độ quan trọng của tham số từ study
        
        Trả về
        ------
        dict
            Điểm số độ quan trọng của tham số
        """
        if self.study is None:
            raise ValueError("Không có study khả dụng.")
            
        try:
            from optuna.importance import get_param_importances
            importances = get_param_importances(self.study)
            
            self.logger.info("Độ quan trọng của tham số:")
            for param, importance in importances.items():
                self.logger.info(f"  {param}: {importance:.3f}")
                
            return importances
            
        except ImportError:
            self.logger.warning("Không thể tính toán độ quan trọng tham số.")
            return {}

def main_optuna_example():
    """
    Hàm chính minh họa tối ưu hóa Optuna cho dự đoán dinh dưỡng fast food
    Tập trung vào ElasticNet - model tốt nhất
    """
    # Thiết lập logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Thiết lập random state toàn cục để tái tạo kết quả
    np.random.seed(42)
    
    print("="*60)
    print("TỐI ƯU HÓA BAYESIAN OPTUNA CHO ELASTICNET")
    print("="*60)
    
    # Tải và tiền xử lý dữ liệu (giống main2.py)
    file_path = 'D:\\KHDL - HCMUS\\Năm 3\\Python KHDL\\Project2\\Python-Cho-KHDL\\FastFoodNutritionMenuV3.csv'
    
    preprocessor = DataPreprocessor(missing_strategy='median',
                                  scaling_strategy='standard',
                                  outlier_method='iqr')
    preprocessor.load_data(file_path)
    preprocessor.drop_features(['calories_from_fat', 'weight_watchers_pnts', 'company', 'item'])
    preprocessor.clean_negative_values()
    
    clean_data = preprocessor.get_processed_data()
    
    # Import ModelTrainer để sử dụng split_data
    from model import ModelTrainer
    
    # Khởi tạo ModelTrainer và split dữ liệu 1 lần duy nhất
    temp_trainer = ModelTrainer(random_state=42)
    temp_trainer.load_data(clean_data, target_column='calories')
    temp_trainer.split_data(test_size=0.2)
    
    # Lấy train và test data từ ModelTrainer
    train_val_data = pd.concat([temp_trainer.X_train, temp_trainer.y_train], axis=1)
    test_data = pd.concat([temp_trainer.X_test, temp_trainer.y_test], axis=1)
    
    print(f"Chia dữ liệu - Train: {len(train_val_data)}, Test: {len(test_data)}")
    
    # Áp dụng tiền xử lý (FIT chỉ trên train)
    preprocessor.data = train_val_data.copy()
    preprocessor.auto_detect_columns()
    
    # Xử lý missing values (FIT trên train)
    train_processed = preprocessor.handle_missing_values(
        data=train_val_data,
        num_strategy='median',
        fit=True
    )
    
    # Xử lý outliers chỉ trên TRAIN (không áp dụng cho test)
    train_processed = preprocessor.handle_outliers(
        data=train_processed,
        exclude_features=['trans_fat_g', 'calories'],
        outlier_strategy='drop'  # Đổi từ isolation_forest sang drop
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
    
    X_train_val = train_processed.drop(columns=['calories']).values
    y_train_val = train_processed['calories'].values
    
    X_test = test_processed.drop(columns=['calories']).values
    y_test = test_processed['calories'].values
    
    print(f"Kích thước Train: {X_train_val.shape}, Test: {X_test.shape}")
    
    # Khởi tạo Optuna tuner chỉ với dữ liệu TRAIN+VAL
    tuner = OptunaTuner(X_train_val, y_train_val, random_state=42)
    
    # Chạy tối ưu hóa với nhiều trials để ổn định
    print("Bắt đầu tối ưu hóa Optuna...")
    best_params = tuner.optimize_elasticnet(n_trials=100, timeout=3600)  # Tăng trials
    
    # Theo phương pháp notebook: Tạo model cuối cùng với tham số tốt nhất
    print("\n" + "="*60)
    print("TẠO MODEL CUỐI CÙNG VỚI THAM SỐ TỐT NHẤT")
    print("="*60)
    
    final_elasticnet = ElasticNet(
        **best_params,
        random_state=42
    )
    
    # Huấn luyện trên dataset TRAIN+VAL (bộ tối ưu siêu tham số)
    final_elasticnet.fit(X_train_val, y_train_val)
    
    # Đánh giá trên dataset TEST (dữ liệu thực sự chưa thấy)
    y_test_pred = final_elasticnet.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cũng đánh giá trên train+val để so sánh
    y_train_val_pred = final_elasticnet.predict(X_train_val)
    train_val_mae = mean_absolute_error(y_train_val, y_train_val_pred)
    train_val_mse = mean_squared_error(y_train_val, y_train_val_pred)
    train_val_r2 = r2_score(y_train_val, y_train_val_pred)
    
    print("\nHiệu suất model cuối cùng:")
    print(f"  RMSE tốt nhất (CV): {tuner.best_score:.4f}")
    print(f"\nBỘ TRAIN+VAL:")
    print(f"  MAE: {train_val_mae:.4f}")
    print(f"  MSE: {train_val_mse:.4f}")
    print(f"  Điểm R²: {train_val_r2:.4f}")
    print(f"\nBỘ TEST (Dữ liệu chưa thấy):")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  Điểm R²: {test_r2:.4f}")
    print(f"\nTham số tốt nhất: {best_params}")
    
    # Lấy độ quan trọng tham số
    tuner.get_param_importances()
    
    # Lưu study và model
    study_path = tuner.save_study()
    
    model_path = f"models/optuna_best_elasticnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_elasticnet, model_path)
    print(f"Model tốt nhất đã lưu vào: {model_path}")
    
    # Thử vẽ lịch sử tối ưu hóa
    try:
        tuner.plot_optimization_history('plots/optuna_optimization_history.png')
    except:
        print("Không thể tạo biểu đồ tối ưu hóa (có thể plotly chưa được cài đặt)")
    
    print("="*60)
    print("TỐI ƯU HÓA OPTUNA HOÀN THÀNH!")
    print("="*60)

if __name__ == "__main__":
    main_optuna_example()