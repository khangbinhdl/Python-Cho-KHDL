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
    """Setup logging configuration"""
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

    logging.getLogger().info(f"Logging initialized → {log_path}")
    return log_path

def main():
    # Setup logging
    setup_logging()
    
    # Load and preprocess data
    # file_path = './FastFoodNutritionMenuV3.csv'
    file_path = 'D:\\KHDL - HCMUS\\Năm 3\\Python KHDL\\Project2\\Python-Cho-KHDL\\FastFoodNutritionMenuV3.csv'
    
    preprocessor = DataPreprocessor(missing_strategy='median',
                                  scaling_strategy='standard',
                                  outlier_method='iqr')
    preprocessor.load_data(file_path)
    
    # Remove unnecessary features
    preprocessor.drop_features(['calories_from_fat', 'weight_watchers_pnts', 'company', 'item'])
    preprocessor.clean_negative_values()
    
    clean_data = preprocessor.get_processed_data()
    
    # Split into train/val/test (60/20/20) to avoid data leakage
    # First split: separate test set (20%)
    train_val_data, test_data = train_test_split(clean_data, test_size=0.2, random_state=42)
    
    # Second split: separate train and validation from remaining 80%
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2 of total
    
    print(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Apply preprocessing to train data (FIT)
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
    
    # Apply preprocessing to validation data (TRANSFORM only)
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
    
    # Apply preprocessing to test data (TRANSFORM only)
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
    
    # Combine train+val for model training and hyperparameter tuning
    train_val_data = pd.concat([train_processed, val_processed], ignore_index=True)
    
    print(f"Train+Val shape: {train_val_data.shape}, Test shape: {test_processed.shape}")
    
    # Initialize ModelTrainer with train+val data
    trainer = ModelTrainer(random_state=42)
    trainer.load_data(train_val_data, target_column='calories')
    
    # Create manual validation split for hyperparameter tuning
    # We'll use the validation data we already separated
    X_train = train_processed.drop(columns=['calories'])
    y_train = train_processed['calories']
    X_val = val_processed.drop(columns=['calories'])
    y_val = val_processed['calories']
    X_test = test_processed.drop(columns=['calories'])
    y_test = test_processed['calories']
    
    # For ModelTrainer, we'll use the combined train+val and let it split again
    trainer.split_data(test_size=0.25)  # This will separate our validation set
    
    # Initialize models
    trainer.initialize_models()
    
    # Train a subset of fast models first for comparison
    quick_models = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTree']
    trainer.train_models(quick_models)
    trainer.evaluate_models()
    
    print("\n" + "="*50)
    print("QUICK MODELS RESULTS")
    print("="*50)
    for result in trainer.results:
        print(f"{result['model_name']}: R² = {result['r2_score']:.4f}")
    
    # Now train ensemble models
    ensemble_models = ['RandomForest', 'ExtraTrees', 'GradientBoosting']
    trainer.train_models(ensemble_models)
    trainer.evaluate_models()
    
    print("\n" + "="*50)
    print("ALL MODELS RESULTS")
    print("="*50)
    for result in trainer.results:
        print(f"{result['model_name']}: R² = {result['r2_score']:.4f}")
    
    # Hyperparameter tuning for ExtraTrees (best performing model)
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING FOR EXTRA TREES")
    print("="*50)
    
    et_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Use RandomizedSearchCV for faster tuning
    best_et = trainer.hyperparameter_tuning(
        model_name='ExtraTrees',
        param_grid=et_param_grid,
        cv=5,
        scoring='r2',
        search_type='random'  # Faster than grid search
    )
    
    if best_et:
        # Retrain with optimized parameters
        trainer.train_models(['ExtraTrees'])
        trainer.evaluate_models()
        
        print("\n" + "="*50)
        print("FINAL RESULTS AFTER TUNING")
        print("="*50)
        for result in trainer.results:
            print(f"{result['model_name']}: R² = {result['r2_score']:.4f}")
    
    # Plot comparisons
    trainer.plot_model_comparison(save_path='plots/hypertuned_model_comparison.png')
    
    # Show feature importance for the best tree-based model
    try:
        trainer.plot_feature_importance(save_path='plots/hypertuned_feature_importance.png')
    except Exception as e:
        print(f"Could not plot feature importance: {e}")
    
    # Save results
    trainer.save_results(filepath='results/hypertuned_results.csv')
    trainer.save_model(filepath='models/best_hypertuned_model.pkl')
    
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING COMPLETED!")
    print(f"Best model: {trainer.best_model_name}")
    print(f"Best R² score: {max(trainer.results, key=lambda x: x['r2_score'])['r2_score']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()