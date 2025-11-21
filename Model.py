import pandas as pd
import numpy as np
import joblib
import logging
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Logger riêng
LOGGER = logging.getLogger("MODEL_TRAINER")
if not LOGGER.handlers:
    LOGGER.propagate = True
    LOGGER.setLevel(logging.INFO)

class ModelTrainer:
    """
    Class quản lý toàn bộ quy trình: Chia dữ liệu, Chuẩn bị X/y, Huấn luyện, Tối ưu và Đánh giá.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=self.random_state),
            'Lasso': Lasso(random_state=self.random_state),
            'RandomForest': RandomForestRegressor(random_state=self.random_state),
            'GradientBoosting': GradientBoostingRegressor(random_state=self.random_state)
        }
        self.best_model = None
        self.best_model_name = None
        self.evaluation_results = []
        self.trained_models = {}
        
        # Các biến chứa dữ liệu
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _log(self, message):
        LOGGER.info(message)

    def split_data(self, data, test_size=0.2):
        """
        Bước 1: Chia DataFrame gốc thành 2 DataFrame con (Train/Test).
        Hàm này thay thế việc gọi train_test_split trực tiếp ở main.
        
        Parameters
        ----------
        data : DataFrame
            Dữ liệu đã được xử lý sơ bộ (clean basic, encode).
        """
        self._log("Splitting data into Train and Test sets...")
        self.train_df, self.test_df = train_test_split(
            data, test_size=test_size, random_state=self.random_state
        )
        self._log(f"Split complete. Train shape: {self.train_df.shape}, Test shape: {self.test_df.shape}")
        return self.train_df, self.test_df

    def set_training_data(self, train_processed, test_processed, target_col):
        """
        Bước 2: Sau khi Main đã xử lý (Outlier/Missing/Scale) trên từng tập,
        hàm này nhận lại dữ liệu sạch để tách X và y.
        """
        self._log("Setting processed training data (separating X and y)...")
        
        self.X_train = train_processed.drop(columns=[target_col])
        self.y_train = train_processed[target_col]
        
        self.X_test = test_processed.drop(columns=[target_col])
        self.y_test = test_processed[target_col]
        
        self._log(f"Ready for training. X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")

    def optimize_params(self, model_name, param_grid, cv=5):
        """Tối ưu tham số bằng GridSearchCV."""
        if model_name not in self.models:
            self._log(f"Model {model_name} not found.")
            return

        self._log(f"Optimizing {model_name}...")
        grid = GridSearchCV(
            self.models[model_name], 
            param_grid, 
            cv=cv, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        grid.fit(self.X_train, self.y_train)
        self.models[model_name] = grid.best_estimator_
        self._log(f"Best params for {model_name}: {grid.best_params_}")

    def train_all_models(self):
        """Huấn luyện tất cả các model trong danh sách."""
        self._log("Starting training...")
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
            self._log(f"Trained {name}")

    def evaluate_models(self):
        """Đánh giá mô hình trên tập Test."""
        self.evaluation_results = []
        best_score = -np.inf # So sánh bằng R2
        
        self._log("Evaluating models on Test set...")
        for name, model in self.trained_models.items():
            y_pred = model.predict(self.X_test)
            
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            self.evaluation_results.append({
                'model_name': name, 'mse': mse, 'rmse': rmse, 'mae': mae, 'r2_score': r2
            })
            
            if r2 > best_score:
                best_score = r2
                self.best_model = model
                self.best_model_name = name

        return {'results': self.evaluation_results, 'best_model_name': self.best_model_name}

    def get_feature_importance(self):
        """Lấy độ quan trọng feature."""
        if self.best_model is None: return None
        
        importances = None
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_)
            
        if importances is not None:
            return pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
        return None

    def save_results(self, path):
        if self.evaluation_results:
            pd.DataFrame(self.evaluation_results).to_csv(path, index=False)
            self._log(f"Saved results to {path}")

    def save_best_model(self, path_dir):
        if self.best_model:
            os.makedirs(path_dir, exist_ok=True)
            path = os.path.join(path_dir, f"{self.best_model_name}_best.pkl")
            joblib.dump(self.best_model, path)
            self._log(f"Saved best model to {path}")
            return path