from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from src.models.evaluator import ModelEvaluator
from src.models.optimizer import BayesianOptimizer
from src.utils.logging import get_logger

# Logger riêng
LOGGER = get_logger("MODEL_TRAINER")

class ModelTrainer:
    """
    Class quản lý toàn bộ quy trình huấn luyện mô hình Machine Learning.
    
    Chỉ tập trung vào logic cốt lõi: Quản lý dữ liệu, Huấn luyện và Tối ưu hóa tham số.
    Việc đánh giá và IO được tách ra các module khác.
    
    Attributes
    ----------
    random_state : int
        Seed cho reproducibility.
    data : DataFrame or None
        Dữ liệu gốc được nạp vào.
    train_df : DataFrame or None
        Dữ liệu training sau khi split.
    test_df : DataFrame or None
        Dữ liệu testing sau khi split.
    X_train : DataFrame or None
        Features của tập training.
    X_test : DataFrame or None
        Features của tập testing.
    y_train : Series or None
        Target của tập training.
    y_test : Series or None
        Target của tập testing.
    models : dict
        Dictionary chứa các model templates (chưa train).
    trained_models : dict
        Dictionary chứa các models đã được huấn luyện.
    results : list
        Danh sách kết quả đánh giá các models.
    best_model : object or None
        Model có hiệu suất tốt nhất.
    best_model_name : str or None
        Tên của model tốt nhất.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Khởi tạo ModelTrainer.
        
        Parameters
        ----------
        random_state : int, optional
            Seed cho reproducibility. Mặc định là 42.
        """
        self.random_state: int = random_state
        
        # Các biến chứa dữ liệu
        self.data: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.target_column: Optional[str] = None

        # Các biến liên quan đến models
        self.models: dict[str, Any] = {}  # Model templates (chưa train)
        self.trained_models: dict[str, Any] = {}  # Models đã train
        self.results: list[dict[str, Any]] = []  # Kết quả đánh giá
        self.best_model: Optional[Any] = None  # Model tốt nhất
        self.best_model_name: Optional[str] = None  # Tên model tốt nhất

        np.random.seed(random_state)
        self._log("ModelTrainer initialized with random_state={}".format(random_state))

    def __str__(self) -> str:
        """Biểu diễn chuỗi thân thiện với người dùng."""
        trained_count = len(self.trained_models)
        if trained_count == 0:
            return "ModelTrainer (chưa huấn luyện model nào)"
        return f"ModelTrainer: {trained_count} models đã train, best={self.best_model_name}"

    def __repr__(self) -> str:
        """Biểu diễn chuỗi dành cho developer."""
        return f"ModelTrainer(random_state={self.random_state}, trained={len(self.trained_models)})"

    @staticmethod
    def _log(message: str) -> None:
        LOGGER.info(message)

    def load_data(self, data: pd.DataFrame, target_column: str = 'calories') -> ModelTrainer:
        """
        Nạp dữ liệu đã được tiền xử lý vào ModelTrainer.
        
        Parameters
        ----------
        data : DataFrame
            Dữ liệu đã được tiền xử lý.
        target_column : str, optional
            Tên cột target cần dự đoán. Mặc định là 'calories'.
            
        Returns
        -------
        self
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

    def split_data(
        self,
        test_size: float = 0.2,
        stratify: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chia dữ liệu thành tập train và test.
        
        Parameters
        ----------
        test_size : float, optional
            Tỷ lệ dữ liệu dùng cho test. Mặc định là 0.2.
        stratify : array-like or None, optional
            Stratified split. Mặc định là None.
            
        Returns
        -------
        train_df, test_df
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self._log("Splitting data into Train and Test sets...")
        self.train_df, self.test_df = train_test_split(
            self.data, test_size=test_size, 
            random_state=self.random_state, stratify=stratify
        )
        self._log(f"Split complete. Train shape: {self.train_df.shape}, Test shape: {self.test_df.shape}")
        return self.train_df, self.test_df

    def set_training_data(
        self,
        train_processed: pd.DataFrame,
        test_processed: pd.DataFrame,
        target_col: str
    ) -> None:
        """
        Thiết lập dữ liệu đã xử lý và tách thành X, y.
        
        Parameters
        ----------
        train_processed : DataFrame
            Dữ liệu training đã được tiền xử lý.
        test_processed : DataFrame
            Dữ liệu testing đã được tiền xử lý.
        target_col : str
            Tên cột target cần dự đoán.
        """
        self._log("Setting processed training data (separating X and y)...")
        
        self.X_train = train_processed.drop(columns=[target_col])
        self.y_train = train_processed[target_col]
        
        self.X_test = test_processed.drop(columns=[target_col])
        self.y_test = test_processed[target_col]
        
        self._log(f"Ready for training. X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")

    def initialize_models(self) -> ModelTrainer:
        """
        Khởi tạo danh sách các mô hình Machine Learning.
        
        Khởi tạo các mô hình regression bao gồm:
        LinearRegression, Ridge, Lasso, ElasticNet, RandomForest, LightGBM.
        
        Returns
        -------
        self
            Trả về instance để cho phép method chaining.
        """
        self.models = {
            'LinearRegression': LinearRegression(),
            
            'Ridge': Ridge(random_state=self.random_state),
            
            'Lasso': Lasso(random_state=self.random_state),
            
            'ElasticNet': ElasticNet(random_state=self.random_state),
            
            'RandomForest': RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=2
            ),

            'LightGBM': LGBMRegressor(
                random_state=self.random_state,
                verbose=-1,
                n_jobs=2
            ),
        }
        
        self._log(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        return self

    def train_models(self, models_to_train: Optional[list[str]] = None) -> ModelTrainer:
        """
        Huấn luyện các mô hình.
        
        Parameters
        ----------
        models_to_train : list of str or None, optional
            Danh sách tên các mô hình cần huấn luyện.
            Nếu None, huấn luyện tất cả các models đã được khởi tạo.
            Mặc định là None.
        
        Returns
        -------
        self
            Trả về instance để cho phép method chaining.
        
        Raises
        ------
        ValueError
            Nếu chưa có dữ liệu training. Cần gọi split_data() trước.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not available. Call split_data() first.")
            
        if not self.models:
            self.initialize_models()
            
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        self.trained_models = {}
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

    def optimize_params(
        self,
        model_name: str,
        n_trials: int = 20,
        cv: int = 5,
        n_jobs: int = 1
    ) -> Optional[dict[str, Any]]:
        """
        Tối ưu hóa siêu tham số của model bằng Bayesian Optimization.
        
        Parameters
        ----------
        model_name : str
            Tên của model cần tối ưu (ví dụ: 'RandomForest', 'LightGBM').
        n_trials : int, optional
            Số lần thử nghiệm tối ưu. Mặc định là 20.
        cv : int, optional
            Số fold cho cross-validation. Mặc định là 5.
        n_jobs : int, optional
            Số jobs song song khi training. Mặc định là 1.
        
        Returns
        -------
        dict or None
            Dictionary chứa best parameters nếu thành công.
            Trả về None nếu model không cần tối ưu hoặc tối ưu thất bại.
        
        Raises
        ------
        ValueError
            Nếu chưa có dữ liệu training hoặc model không tồn tại.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not available. Call set_training_data() first.")
        
        if not self.models:
            self.initialize_models()
            
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")

        if model_name == 'LinearRegression':
            self._log("LinearRegression does not require optimization. Skipping.")
            return None

        self._log(f"Starting optimization for {model_name}...")
        
        try:
            optimizer = BayesianOptimizer(
                self.X_train, self.y_train, 
                random_state=self.random_state, 
                cv=cv
            )
            
            best_params = optimizer.optimize(model_name, n_trials=n_trials)
            
            if best_params is None:
                self._log(f"Optimization failed for {model_name}. Model will use default params.")
                return None

            model_class = self.models[model_name].__class__
            model_default_params = self.models[model_name].get_params()
            
            final_params = best_params.copy()
            
            if 'random_state' in model_default_params:
                final_params['random_state'] = self.random_state
            if 'n_jobs' in model_default_params:
                final_params['n_jobs'] = n_jobs
            if model_name == 'LightGBM' and 'verbose' in model_default_params:
                final_params['verbose'] = -1

            optimized_model = model_class(**final_params)
            self.models[model_name] = optimized_model
            
            self._log(f"✓ {model_name} parameters optimized successfully")
            self._log(f"  Best params: {best_params}")
            
            return best_params

        except Exception as e:
            self._log(f"✗ Error during optimization of {model_name}: {str(e)}")
            return None

    def evaluate_models(self) -> dict[str, Union[list[dict[str, Any]], Optional[str]]]:
        """
        Đánh giá tất cả các mô hình đã huấn luyện trên tập test.
        
        Sử dụng ModelEvaluator để tính toán các metrics đánh giá
        (MSE, RMSE, MAE, R²) cho từng model.
        
        Returns
        -------
        dict
            Dictionary chứa:
            - 'results': list các dict kết quả đánh giá của từng model
            - 'best_model_name': tên model có R² cao nhất
        
        Raises
        ------
        ValueError
            Nếu chưa có model đã huấn luyện hoặc chưa có dữ liệu test.
        """
        if not self.trained_models:
            raise ValueError("No trained models found. Call train_models() first.")
            
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not available. Call split_data() first.")
        
        self._log("Evaluating models...")
        self.results = []
        best_score = -np.inf
        
        for name, model in self.trained_models.items():
            result = ModelEvaluator.evaluate_model(model, self.X_test, self.y_test, name)
            
            if result:
                self.results.append(result)
                
                # Cập nhật best model dựa trên R2
                if result['r2_score'] > best_score:
                    best_score = result['r2_score']
                    self.best_model = model
                    self.best_model_name = name
        
        return {'results': self.results, 'best_model_name': self.best_model_name}

    def get_feature_importance(
        self,
        model_name: Optional[str] = None,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Lấy độ quan trọng của các features từ mô hình.
        
        Parameters
        ----------
        model_name : str or None, optional
            Tên model cần lấy feature importance.
            Nếu None, sử dụng best_model. Mặc định là None.
        top_n : int or None, optional
            Số lượng features quan trọng nhất cần lấy.
            Nếu None, lấy tất cả. Mặc định là None.
        
        Returns
        -------
        DataFrame
            DataFrame chứa tên feature và độ quan trọng,
            đã sắp xếp theo thứ tự giảm dần.
        
        Raises
        ------
        ValueError
            Nếu chưa có best_model hoặc model_name không tồn tại.
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model found. Train and evaluate models first.")
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Trained model '{model_name}' not found")
            model = self.trained_models[model_name]
            
        return ModelEvaluator.get_feature_importance(model, self.X_train.columns, model_name, top_n)
