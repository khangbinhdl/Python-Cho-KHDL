"""
Minh họa cách áp dụng Bayesian optimization cho ExtraTreesRegressor (model tốt nhất)
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from Preprocessing import DataPreprocessor
import warnings
warnings.filterwarnings('ignore')

class OptunaTuner:
    """
    Advanced hyperparameter tuning using Optuna Bayesian Optimization for ExtraTrees
    
    Provides sophisticated hyperparameter search with pruning and 
    parallel execution capabilities.
    
    Attributes
    ----------
    X_train : array-like
        Training features
    y_train : array-like  
        Training target
    random_state : int
        Random seed for reproducibility
    study : optuna.Study
        Optuna study object
    best_params : dict
        Best parameters found
    best_score : float
        Best score achieved
    """
    
    def __init__(self, X_train, y_train, random_state=42):
        """
        Initialize OptunaTuner with training data
        
        Parameters
        ----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target  
        random_state : int, optional
            Random seed. Default is 42
        """
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.study = None
        self.best_params = None
        self.best_score = None
        
        # Setup logging
        self.logger = logging.getLogger("OPTUNA_TUNER")
        
    def et_objective(self, trial):
        """
        Objective function for ExtraTrees optimization
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
            
        Returns
        -------
        float
            Negative RMSE score (to minimize)
        """
        # Suggest hyperparameters with focus on stability
        n_estimators = trial.suggest_int("n_estimators", 100, 300, step=50)  # Narrower range
        max_depth = trial.suggest_int("max_depth", 15, 35)  # Focus on deeper trees
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        max_features = trial.suggest_categorical("max_features", ['sqrt', 'log2', None])  # Categorical instead of float
        
        # Create model with suggested parameters
        et = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=self.random_state
        )
        
        # Perform cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = []
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(self.X_train)):
            X_fold_train = self.X_train.iloc[train_idx] if isinstance(self.X_train, pd.DataFrame) else self.X_train[train_idx]
            X_fold_valid = self.X_train.iloc[valid_idx] if isinstance(self.X_train, pd.DataFrame) else self.X_train[valid_idx]
            y_fold_train = self.y_train.iloc[train_idx] if isinstance(self.y_train, pd.Series) else self.y_train[train_idx]
            y_fold_valid = self.y_train.iloc[valid_idx] if isinstance(self.y_train, pd.Series) else self.y_train[valid_idx]
            
            # Fit and predict
            et.fit(X_fold_train, y_fold_train)
            y_pred = et.predict(X_fold_valid)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_fold_valid, y_pred))
            scores.append(rmse)
            
            # Report intermediate result for pruning
            trial.report(np.mean(scores), step=fold)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return np.mean(scores)  # Return mean RMSE (to minimize)
    
    def optimize_et(self, n_trials=100, timeout=3600):
        """
        Optimize ExtraTrees hyperparameters using Optuna
        
        Parameters
        ----------
        n_trials : int, optional
            Maximum number of trials. Default is 100
        timeout : int, optional
            Time limit in seconds. Default is 3600 (1 hour)
            
        Returns
        -------
        dict
            Best parameters found
        """
        self.logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        
        # Create study
        self.study = optuna.create_study(
            study_name=f"et_nutrition_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction="minimize",  # Minimize RMSE
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )
        
        # Optimize
        self.study.optimize(
            self.et_objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        self.logger.info(f"Optimization completed!")
        self.logger.info(f"Best RMSE: {self.best_score:.4f}")
        self.logger.info(f"Best params: {self.best_params}")
        
        return self.best_params
    
    def get_best_model(self):
        """
        Get the best ExtraTrees model with optimized parameters
        
        Returns
        -------
        ExtraTreesRegressor
            Optimized model fitted on training data
        """
        if self.best_params is None:
            raise ValueError("No optimization performed yet. Call optimize_et() first.")
            
        # Create and train best model
        best_et = ExtraTreesRegressor(
            **self.best_params,
            n_jobs=-1,
            random_state=self.random_state
        )
        
        best_et.fit(self.X_train, self.y_train)
        return best_et
    
    def save_study(self, filepath=None):
        """
        Save Optuna study for later analysis
        
        Parameters
        ----------
        filepath : str, optional
            Path to save study. If None, auto-generate filename
            
        Returns
        -------
        str
            Path to saved study
        """
        if self.study is None:
            raise ValueError("No study to save. Run optimization first.")
            
        os.makedirs("optuna_studies", exist_ok=True)
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"optuna_studies/et_study_{timestamp}.pkl"
            
        joblib.dump(self.study, filepath)
        self.logger.info(f"Study saved to: {filepath}")
        return filepath
    
    def plot_optimization_history(self, save_path=None):
        """
        Plot optimization history using Optuna's built-in plotting
        
        Parameters
        ----------
        save_path : str, optional
            Path to save plot. If None, just display
        """
        if self.study is None:
            raise ValueError("No study available for plotting.")
            
        try:
            import optuna.visualization as vis
            import plotly.io as pio
            
            # Create optimization history plot
            fig = vis.plot_optimization_history(self.study)
            fig.update_layout(title="Optuna Optimization History")
            
            if save_path:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'plots', exist_ok=True)
                pio.write_image(fig, save_path)
                self.logger.info(f"Optimization history saved to: {save_path}")
            
            fig.show()
            
        except ImportError:
            self.logger.warning("Plotly not available. Cannot create optimization plots.")
    
    def get_param_importances(self):
        """
        Get parameter importance from the study
        
        Returns
        -------
        dict
            Parameter importance scores
        """
        if self.study is None:
            raise ValueError("No study available.")
            
        try:
            from optuna.importance import get_param_importances
            importances = get_param_importances(self.study)
            
            self.logger.info("Parameter importances:")
            for param, importance in importances.items():
                self.logger.info(f"  {param}: {importance:.3f}")
                
            return importances
            
        except ImportError:
            self.logger.warning("Cannot calculate parameter importances.")
            return {}

def main_optuna_example():
    """
    Main function demonstrating Optuna optimization for fast food nutrition prediction
    Following the exact approach from [Bayesian]_Hyperparameters_tuning.ipynb
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set global random state for reproducibility
    np.random.seed(42)
    
    print("="*60)
    print("OPTUNA BAYESIAN OPTIMIZATION FOR EXTRATREES")
    print("="*60)
    
    # Load and preprocess data (same as main2.py)
    file_path = 'D:\\KHDL - HCMUS\\Năm 3\\Python KHDL\\Project2\\Python-Cho-KHDL\\FastFoodNutritionMenuV3.csv'
    
    preprocessor = DataPreprocessor(missing_strategy='median',
                                  scaling_strategy='standard',
                                  outlier_method='iqr')
    preprocessor.load_data(file_path)
    preprocessor.drop_features(['calories_from_fat', 'weight_watchers_pnts', 'company', 'item'])
    preprocessor.clean_negative_values()
    
    clean_data = preprocessor.get_processed_data()
    
    # Split and preprocess
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(clean_data, test_size=0.2, random_state=42)
    
    # Apply preprocessing
    preprocessor.data = train_data.copy()
    preprocessor.auto_detect_columns()
    
    train_processed = preprocessor.handle_missing_values(data=train_data, num_strategy='median', fit=True)
    train_processed = preprocessor.handle_outliers(data=train_processed, exclude_features=['trans_fat_g', 'calories'], outlier_strategy='drop')
    train_processed = preprocessor.scale_features(data=train_processed, exclude_features=['calories'], fit=True)
    
    # Prepare data for Optuna - Use FULL PROCESSED DATA like notebook
    test_processed = preprocessor.handle_missing_values(data=test_data, num_strategy='median', fit=False)
    test_processed = preprocessor.scale_features(data=test_processed, exclude_features=['calories'], fit=False)
    
    processed_data = pd.concat([train_processed, test_processed], ignore_index=True)
    
    X = processed_data.drop(columns=['calories']).values
    y = processed_data['calories'].values
    
    print(f"Full dataset shape: {X.shape}")
    
    # Initialize Optuna tuner with FULL dataset (like notebook)
    tuner = OptunaTuner(X, y, random_state=42)
    
    # Run optimization with more trials for stability
    print("Starting Optuna optimization...")
    best_params = tuner.optimize_et(n_trials=100, timeout=3600)  # Increase trials
    
    # Following notebook approach: Create final model with best params
    print("\n" + "="*60)
    print("CREATING FINAL MODEL WITH BEST PARAMS")
    print("="*60)
    
    final_et = ExtraTreesRegressor(
        **best_params,
        n_jobs=-1,
        random_state=42
    )
    
    # Train on ENTIRE dataset (like notebook cell 6)
    final_et.fit(X, y)
    
    # Evaluate on ENTIRE dataset (like notebook cell 7-8)  
    y_pred = final_et.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print("\nFinal model performance on the entire dataset:")
    print(f"  Best RMSE (CV): {tuner.best_score:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Best parameters: {best_params}")
    
    # Get parameter importances
    tuner.get_param_importances()
    
    # Save study and model
    study_path = tuner.save_study()
    
    model_path = f"models/optuna_best_et_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_et, model_path)
    print(f"Best model saved to: {model_path}")
    
    # Try to plot optimization history
    try:
        tuner.plot_optimization_history('plots/optuna_optimization_history.png')
    except:
        print("Could not create optimization plots (plotly may not be installed)")
    
    print("="*60)
    print("OPTUNA OPTIMIZATION COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main_optuna_example()