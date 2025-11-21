import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import logging

# Logger riêng cho ModelTrainer
LOGGER = logging.getLogger("MODEL_TRAINER")
if not LOGGER.handlers:
    LOGGER.propagate = True
    LOGGER.setLevel(logging.INFO)

# Thiết lập style cho seaborn
sns.set_theme(style="whitegrid")

class ModelTrainer:
    """
    Class đóng gói quá trình huấn luyện và đánh giá mô hình Machine Learning
    
    Cung cấp các chức năng: nạp dữ liệu, chia train/test, huấn luyện nhiều mô hình,
    tối ưu siêu tham số, đánh giá, lưu mô hình và ghi kết quả thực nghiệm.

    Attributes
    ----------
    data : DataFrame or None
        Dữ liệu đã được tiền xử lý
    X_train : DataFrame or None
        Tập huấn luyện (features)
    X_test : DataFrame or None  
        Tập kiểm tra (features)
    y_train : Series or None
        Tập huấn luyện (target)
    y_test : Series or None
        Tập kiểm tra (target)
    models : dict
        Dictionary chứa các mô hình đã được khởi tạo
    trained_models : dict
        Dictionary chứa các mô hình đã được huấn luyện
    results : list
        Danh sách kết quả đánh giá các mô hình
    best_model : object or None
        Mô hình tốt nhất
    best_model_name : str or None
        Tên mô hình tốt nhất
    random_state : int
        Seed cho tính tái lập
    """
    
    def __init__(self, random_state=42):
        """
        Khởi tạo đối tượng ModelTrainer với random seed cố định
        
        Parameters
        ----------
        random_state : int, optional
            Seed cho tính tái lập của các thuật toán ML.
            Mặc định là 42
        """
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.models = {}
        self.trained_models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        
        self.random_state = random_state
        np.random.seed(random_state)
        
        self._log("ModelTrainer initialized with random_state={}".format(random_state))
        
    def _log(self, message):
        """
        Hàm tiện ích để ghi log với logger riêng của ModelTrainer
        
        Parameters
        ----------
        message : str
            Thông điệp cần ghi log
        """
        LOGGER.info(message)
    
    def load_data(self, data, target_column='calories'):
        """
        Nạp dữ liệu đã được tiền xử lý vào ModelTrainer
        
        Parameters
        ----------
        data : DataFrame
            Dữ liệu đã được tiền xử lý từ DataPreprocessor
        target_column : str, optional
            Tên cột target cần dự đoán.
            Mặc định là 'calories'
            
        Returns
        -------
        self
            Trả về chính đối tượng để có thể chain methods
            
        Raises
        ------
        ValueError
            Nếu dữ liệu không hợp lệ hoặc không chứa cột target
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        self.data = data.copy()
        self.target_column = target_column
        
        self._log(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        self._log(f"Target column: '{target_column}'")
        
        return self
    
    def split_data(self, test_size=0.2, stratify=None):
        """
        Chia dữ liệu thành tập huấn luyện và kiểm tra
        
        Parameters
        ----------
        test_size : float, optional
            Tỷ lệ dữ liệu dành cho tập kiểm tra (0.0-1.0).
            Mặc định là 0.2 (20%)
        stratify : array-like, optional
            Cột dùng để stratified split (cho classification).
            Mặc định là None (random split)
            
        Returns
        -------
        self
            Trả về chính đối tượng để có thể chain methods
            
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify
        )
        
        self._log(f"Data split completed:")
        self._log(f"  Train: {self.X_train.shape[0]} rows")
        self._log(f"  Test: {self.X_test.shape[0]} rows") 
        self._log(f"  Features: {self.X_train.shape[1]}")
        
        return self
        
    def initialize_models(self):
        """
        Khởi tạo danh sách các mô hình Machine Learning
        
        Khởi tạo nhiều loại mô hình hồi quy với các tham số mặc định.
        Không bao gồm SVM do thời gian huấn luyện lâu và khó tối ưu tham số.
        
        Returns
        -------
        self
            Trả về chính đối tượng để có thể chain methods
            
        Notes
        -----
        Các mô hình được khởi tạo:
        - LinearRegression: Hồi quy tuyến tính cơ bản
        - Ridge: Hồi quy Ridge với regularization L2
        - Lasso: Hồi quy Lasso với regularization L1
        - ElasticNet: Kết hợp Ridge và Lasso
        - DecisionTree: Cây quyết định
        - RandomForest: Rừng ngẫu nhiên  
        - ExtraTrees: Extremely Randomized Trees
        - GradientBoosting: Gradient Boosting
        - KNeighbors: K-Nearest Neighbors
        - MLP: Multi-layer Perceptron (Neural Network)
        """
        self.models = {
            'LinearRegression': LinearRegression(),
            
            'Ridge': Ridge(
                random_state=self.random_state
            ),
            
            'Lasso': Lasso(
                random_state=self.random_state,
                max_iter=2000
            ),
            
            'ElasticNet': ElasticNet(
                random_state=self.random_state,
                max_iter=2000
            ),
            
            'DecisionTree': DecisionTreeRegressor(
                random_state=self.random_state
            ),
            
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'GradientBoosting': GradientBoostingRegressor(
                random_state=self.random_state
            ),
            
            'KNeighbors': KNeighborsRegressor(
                n_jobs=-1
            ),
            
            'MLP': MLPRegressor(
                random_state=self.random_state,
                max_iter=500,
                early_stopping=True
            )
        }
        
        self._log(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        return self
        
    def train_models(self, models_to_train=None):
        """
        Huấn luyện tất cả hoặc một phần các mô hình đã khởi tạo
        
        Parameters
        ----------
        models_to_train : list of str, optional
            Danh sách tên các mô hình cần huấn luyện.
            Nếu None, huấn luyện tất cả mô hình.
            Mặc định là None
            
        Returns
        -------
        self
            Trả về chính đối tượng để có thể chain methods
            
        Raises
        ------
        ValueError
            Nếu dữ liệu train chưa được chuẩn bị hoặc mô hình chưa được khởi tạo
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not available. Call split_data() first.")
            
        if not self.models:
            self.initialize_models()
            
        if models_to_train is None:
            models_to_train = list(self.models.keys())
            
        self._log("Starting model training...")
        
        for name in models_to_train:
            if name not in self.models:
                self._log(f"Warning: Model '{name}' not found, skipping...")
                continue
                
            try:
                self._log(f"Training {name}...")
                model = self.models[name]
                model.fit(self.X_train, self.y_train)
                self.trained_models[name] = model
                self._log(f"✓ {name} trained successfully")
                
            except Exception as e:
                self._log(f"✗ Error training {name}: {str(e)}")
                
        self._log(f"Training completed. {len(self.trained_models)}/{len(models_to_train)} models trained successfully")
        return self
        
    def evaluate_models(self):
        """
        Đánh giá tất cả các mô hình đã được huấn luyện
        
        Tính toán các metric: MSE, MAE, R² cho từng mô hình và lưu kết quả.
        Tự động xác định mô hình tốt nhất dựa trên R² score.
        
        Returns
        -------
        self
            Trả về chính đối tượng để có thể chain methods
            
        Raises
        ------
        ValueError
            Nếu không có mô hình nào đã được huấn luyện
        """
        if not self.trained_models:
            raise ValueError("No trained models found. Call train_models() first.")
            
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not available. Call split_data() first.")
            
        self._log("Evaluating models...")
        self.results = []
        
        for name, model in self.trained_models.items():
            try:
                # Dự đoán
                y_pred = model.predict(self.X_test)
                
                # Tính các metric
                mse = mean_squared_error(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred) 
                r2 = r2_score(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Lưu kết quả
                result = {
                    'model_name': name,
                    'mse': mse,
                    'rmse': rmse, 
                    'mae': mae,
                    'r2_score': r2
                }
                
                self.results.append(result)
                self._log(f"✓ {name}: R²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")
                
            except Exception as e:
                self._log(f"✗ Error evaluating {name}: {str(e)}")
                
        # Tìm mô hình tốt nhất (highest R²)
        if self.results:
            best_result = max(self.results, key=lambda x: x['r2_score'])
            self.best_model_name = best_result['model_name']
            self.best_model = self.trained_models[self.best_model_name]
            
            self._log(f"Best model: {self.best_model_name} (R² = {best_result['r2_score']:.4f})")
            
        return self
        
    def hyperparameter_tuning(self, model_name, param_grid, cv=5, scoring='r2', search_type='grid'):
        """
        Tối ưu siêu tham số cho một mô hình cụ thể
        
        Parameters
        ----------
        model_name : str
            Tên mô hình cần tối ưu (phải có trong self.models)
        param_grid : dict
            Dictionary chứa các tham số và giá trị cần thử
        cv : int, optional
            Số fold cho cross-validation.
            Mặc định là 5
        scoring : str, optional
            Metric để đánh giá ('r2', 'neg_mean_squared_error', etc.).
            Mặc định là 'r2'
        search_type : str, optional
            Loại tìm kiếm: 'grid' (GridSearchCV) hoặc 'random' (RandomizedSearchCV).
            Mặc định là 'grid'
            
        Returns
        -------
        best_estimator : object
            Mô hình với tham số tốt nhất
            
        Raises
        ------
        ValueError
            Nếu mô hình không tồn tại hoặc dữ liệu chưa được chuẩn bị
            
        Notes
        -----
        Kết quả tối ưu sẽ tự động thay thế mô hình cũ trong self.models
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in initialized models")
            
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not available. Call split_data() first.")
            
        self._log(f"Starting hyperparameter tuning for {model_name}...")
        self._log(f"Search type: {search_type}, CV folds: {cv}, Scoring: {scoring}")
        
        base_model = self.models[model_name]
        
        try:
            if search_type == 'grid':
                searcher = GridSearchCV(
                    base_model, param_grid, 
                    cv=cv, scoring=scoring, 
                    n_jobs=-1, verbose=1
                )
            elif search_type == 'random':
                searcher = RandomizedSearchCV(
                    base_model, param_grid,
                    cv=cv, scoring=scoring,
                    n_jobs=-1, verbose=1,
                    n_iter=100,  # Số iteration cho random search
                    random_state=self.random_state
                )
            else:
                raise ValueError("search_type must be 'grid' or 'random'")
                
            # Thực hiện tìm kiếm
            searcher.fit(self.X_train, self.y_train)
            
            # Lưu kết quả
            self.models[model_name] = searcher.best_estimator_
            
            self._log(f"✓ Hyperparameter tuning completed for {model_name}")
            self._log(f"Best score: {searcher.best_score_:.4f}")
            self._log(f"Best params: {searcher.best_params_}")
            
            return searcher.best_estimator_
            
        except Exception as e:
            self._log(f"✗ Error during hyperparameter tuning: {str(e)}")
            return None
            
    def save_model(self, model_name=None, filepath=None, method='joblib'):
        """
        Lưu mô hình đã huấn luyện vào file
        
        Parameters
        ----------
        model_name : str, optional
            Tên mô hình cần lưu. Nếu None, lưu mô hình tốt nhất.
            Mặc định là None
        filepath : str, optional
            Đường dẫn file để lưu. Nếu None, tự động tạo tên file.
            Mặc định là None
        method : str, optional
            Phương thức lưu: 'joblib' hoặc 'pickle'.
            Mặc định là 'joblib'
            
        Returns
        -------
        str
            Đường dẫn file đã lưu
            
        Raises
        ------
        ValueError
            Nếu mô hình không tồn tại hoặc chưa được huấn luyện
        """
        # Xác định mô hình cần lưu
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model found. Train and evaluate models first.")
            model_to_save = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Trained model '{model_name}' not found")
            model_to_save = self.trained_models[model_name]
            
        # Tạo thư mục models nếu chưa có
        os.makedirs("models", exist_ok=True)
        
        # Tạo tên file nếu chưa có
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if method == 'joblib':
                filepath = f"models/{model_name}_{timestamp}.pkl"
            else:
                filepath = f"models/{model_name}_{timestamp}.pickle"
                
        try:
            if method == 'joblib':
                joblib.dump(model_to_save, filepath)
            elif method == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(model_to_save, f)
            else:
                raise ValueError("method must be 'joblib' or 'pickle'")
                
            self._log(f"✓ Model '{model_name}' saved to: {filepath}")
            return filepath
            
        except Exception as e:
            self._log(f"✗ Error saving model: {str(e)}")
            raise
            
    def load_model(self, filepath, method='joblib'):
        """
        Nạp mô hình đã lưu từ file
        
        Parameters
        ----------
        filepath : str
            Đường dẫn đến file mô hình
        method : str, optional
            Phương thức nạp: 'joblib' hoặc 'pickle'.
            Mặc định là 'joblib'
            
        Returns
        -------
        object
            Mô hình đã được nạp
            
        Raises
        ------
        FileNotFoundError
            Nếu file không tồn tại
        """
        try:
            if method == 'joblib':
                model = joblib.load(filepath)
            elif method == 'pickle':
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
            else:
                raise ValueError("method must be 'joblib' or 'pickle'")
                
            self._log(f"✓ Model loaded from: {filepath}")
            return model
            
        except Exception as e:
            self._log(f"✗ Error loading model: {str(e)}")
            raise
            
    def save_results(self, filepath=None, format='csv'):
        """
        Lưu kết quả đánh giá các mô hình vào file
        
        Parameters
        ----------
        filepath : str, optional
            Đường dẫn file để lưu. Nếu None, tự động tạo tên file.
            Mặc định là None
        format : str, optional
            Định dạng file: 'csv' hoặc 'json'.
            Mặc định là 'csv'
            
        Returns
        -------
        str
            Đường dẫn file đã lưu
            
        Raises
        ------
        ValueError
            Nếu chưa có kết quả đánh giá
        """
        if not self.results:
            raise ValueError("No evaluation results found. Call evaluate_models() first.")
            
        # Tạo thư mục results nếu chưa có
        os.makedirs("results", exist_ok=True)
        
        # Tạo tên file nếu chưa có
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"results/model_results_{timestamp}.{format}"
            
        try:
            if format == 'csv':
                df = pd.DataFrame(self.results)
                df.to_csv(filepath, index=False)
            elif format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError("format must be 'csv' or 'json'")
                
            self._log(f"✓ Results saved to: {filepath}")
            return filepath
            
        except Exception as e:
            self._log(f"✗ Error saving results: {str(e)}")
            raise
            
    def plot_model_comparison(self, metric='r2_score', figsize=(12, 8), save_path=None):
        """
        Vẽ biểu đồ so sánh hiệu suất các mô hình sử dụng seaborn
        
        Parameters
        ----------
        metric : str, optional
            Metric để vẽ biểu đồ ('r2_score', 'mse', 'mae', 'rmse').
            Mặc định là 'r2_score'
        figsize : tuple, optional
            Kích thước figure (width, height).
            Mặc định là (12, 8)
        save_path : str, optional
            Đường dẫn để lưu biểu đồ. Nếu None, chỉ hiển thị.
            Mặc định là None
            
        Raises
        ------
        ValueError
            Nếu chưa có kết quả đánh giá hoặc metric không hợp lệ
        """
        if not self.results:
            raise ValueError("No evaluation results found. Call evaluate_models() first.")
            
        valid_metrics = ['r2_score', 'mse', 'mae', 'rmse']
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of: {valid_metrics}")
            
        # Chuẩn bị dữ liệu
        df = pd.DataFrame(self.results)
        df_sorted = df.sort_values(by=metric, ascending=(metric not in ['r2_score']))
        
        # Tạo color palette với highlight cho mô hình tốt nhất
        palette = ['crimson' if name == self.best_model_name else 'steelblue' 
                  for name in df_sorted['model_name']]
        
        # Vẽ biểu đồ với seaborn
        plt.figure(figsize=figsize)
        ax = sns.barplot(
            data=df_sorted, 
            x='model_name', 
            y=metric,
            palette=palette,
            edgecolor='black',
            linewidth=0.8
        )
        
        # Thêm giá trị trên mỗi cột
        for i, (patch, value) in enumerate(zip(ax.patches, df_sorted[metric])):
            ax.text(patch.get_x() + patch.get_width()/2, 
                   patch.get_height() + abs(patch.get_height()) * 0.01,
                   f'{value:.4f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Định dạng biểu đồ
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'Model Performance Comparison\n({metric.replace("_", " ").title()})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Xoay labels và cải thiện hiển thị
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Thêm grid với seaborn style
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Tight layout
        plt.tight_layout()
        
        # Lưu biểu đồ nếu có đường dẫn
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'plots', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            self._log(f"✓ Plot saved to: {save_path}")
            
        plt.show()
        
    def get_feature_importance(self, model_name=None, top_n=None):
        """
        Lấy độ quan trọng của các features từ mô hình
        
        Parameters
        ----------
        model_name : str, optional
            Tên mô hình. Nếu None, sử dụng mô hình tốt nhất.
            Mặc định là None
        top_n : int, optional
            Số lượng features quan trọng nhất để trả về.
            Nếu None, trả về tất cả.
            Mặc định là None
            
        Returns
        -------
        DataFrame or None
            DataFrame chứa tên feature và độ quan trọng, None nếu mô hình không hỗ trợ
            
        Notes
        -----
        Chỉ áp dụng cho mô hình có thuộc tính feature_importances_ 
        (Tree-based models, ensemble methods)
        """
        # Xác định mô hình
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model found. Train and evaluate models first.")
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Trained model '{model_name}' not found")
            model = self.trained_models[model_name]
            
        # Kiểm tra xem mô hình có hỗ trợ feature importance không
        if not hasattr(model, 'feature_importances_'):
            self._log(f"Model '{model_name}' does not support feature importance")
            return None
            
        # Lấy feature importance
        importances = model.feature_importances_
        feature_names = self.X_train.columns
        
        # Tạo DataFrame và sắp xếp
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
            
        self._log(f"✓ Feature importance extracted for {model_name}")
        return importance_df
        
    def plot_feature_importance(self, model_name=None, top_n=10, figsize=(10, 6), save_path=None):
        """
        Vẽ biểu đồ độ quan trọng của features sử dụng seaborn
        
        Parameters
        ----------
        model_name : str, optional
            Tên mô hình. Nếu None, sử dụng mô hình tốt nhất.
            Mặc định là None
        top_n : int, optional
            Số lượng features quan trọng nhất để hiển thị.
            Mặc định là 10
        figsize : tuple, optional
            Kích thước figure (width, height).
            Mặc định là (10, 6)
        save_path : str, optional
            Đường dẫn để lưu biểu đồ. Nếu None, chỉ hiển thị.
            Mặc định là None
        """
        importance_df = self.get_feature_importance(model_name, top_n)
        
        if importance_df is None:
            return
            
        # Sắp xếp để feature quan trọng nhất ở trên (cho horizontal plot)
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # Vẽ biểu đồ với seaborn
        plt.figure(figsize=figsize)
        
        # Tạo color palette gradient
        colors = sns.color_palette("viridis", n_colors=len(importance_df))
        
        ax = sns.barplot(
            data=importance_df,
            y='feature',
            x='importance',
            palette=colors,
            edgecolor='black',
            linewidth=0.6
        )
        
        # Thêm giá trị
        for i, (patch, value) in enumerate(zip(ax.patches, importance_df['importance'])):
            ax.text(patch.get_width() + max(importance_df['importance']) * 0.01, 
                   patch.get_y() + patch.get_height()/2,
                   f'{value:.4f}', 
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Định dạng biểu đồ
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title(f'Feature Importance Analysis\n{model_name or self.best_model_name}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Cải thiện hiển thị
        ax.tick_params(axis='both', labelsize=10)
        
        # Thêm grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Lưu biểu đồ nếu có đường dẫn
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'plots', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            self._log(f"✓ Feature importance plot saved to: {save_path}")
            
        plt.show()
        
    def run_full_pipeline(self, test_size=0.2, tune_best_model=False, save_artifacts=True, skip_split=False):
        """
        Chạy toàn bộ pipeline training từ đầu đến cuối
        
        Parameters
        ----------
        test_size : float, optional
            Tỷ lệ dữ liệu test.
            Mặc định là 0.2
        tune_best_model : bool, optional
            Có tối ưu siêu tham số cho mô hình tốt nhất hay không.
            Mặc định là True
        save_artifacts : bool, optional
            Có lưu mô hình và kết quả hay không.
            Mặc định là True
        skip_split : bool, optional
            Có bỏ qua việc chia dữ liệu hay không (khi đã có train/test data).
            Mặc định là False
            
        Returns
        -------
        dict
            Dictionary chứa tóm tắt kết quả pipeline
            
        Raises
        ------
        ValueError
            Nếu dữ liệu chưa được nạp
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        self._log("="*50)
        self._log("Starting Full ML Pipeline")
        self._log("="*50)
        
        # 1. Chia dữ liệu (chỉ khi chưa có train/test data)
        if not skip_split:
            self.split_data(test_size=test_size)
        else:
            self._log("Skipping data split - using existing train/test data")
        
        # 2. Khởi tạo và huấn luyện mô hình
        self.initialize_models()
        self.train_models()
        
        # 3. Đánh giá mô hình
        self.evaluate_models()
        
        # 4. Tối ưu mô hình tốt nhất 
        if tune_best_model and self.best_model_name:
            self._log(f"Attempting hyperparameter tuning for {self.best_model_name}...")
            
            # Định nghĩa param_grids cho tất cả các mô hình
            param_grids = {
                'Lasso': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'max_iter': [1000, 2000, 3000, 5000],
                    'tol': [1e-4, 1e-3, 1e-2],
                    'selection': ['cyclic', 'random']
                },
                'ElasticNet': {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_iter': [1000, 2000, 3000]
                },
                'RandomForest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                'GradientBoosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7]
                },
                'ExtraTrees': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            }
            
            if self.best_model_name in param_grids:
                search_type = 'random' if self.best_model_name in ['Lasso', 'ElasticNet'] else 'grid'
                self.hyperparameter_tuning(
                    self.best_model_name, 
                    param_grids[self.best_model_name],
                    cv=5 if self.best_model_name in ['Lasso', 'ElasticNet'] else 3,  # Lasso và ElasticNet dùng CV=5
                    search_type=search_type
                )
                # Huấn luyện lại với tham số tối ưu
                self.train_models([self.best_model_name])
                self.evaluate_models()
        
        # 5. Vẽ biểu đồ so sánh (comment để tránh trùng với main2.py)
        # self.plot_model_comparison(save_path='plots/model_comparison.png')
        
        # 6. Vẽ feature importance nếu có thể (comment để tránh trùng với main2.py)  
        # try:
        #     self.plot_feature_importance(save_path='plots/feature_importance.png')
        # except:
        #     pass
        
        # 7. Lưu artifacts (comment để tránh trùng với main2.py)
        if save_artifacts:
            model_path = self.save_model()
            results_path = self.save_results()
            
            summary = {
                'best_model': self.best_model_name,
                'best_score': max(self.results, key=lambda x: x['r2_score'])['r2_score'],
                'total_models': len(self.results),
                'model_saved_to': model_path,
                'results_saved_to': results_path
            }
        else:
            summary = {
                'best_model': self.best_model_name,
                'best_score': max(self.results, key=lambda x: x['r2_score'])['r2_score'],
                'total_models': len(self.results)
            }
            
        self._log("="*50)
        self._log("Pipeline completed successfully!")
        self._log(f"Best model: {summary['best_model']} (R² = {summary['best_score']:.4f})")
        self._log("="*50)
        
        return summary
