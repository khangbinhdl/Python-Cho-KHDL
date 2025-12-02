import sys
import os
from pathlib import Path

# Tự động thêm thư mục gốc project vào sys.path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import os
import itertools
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.utils.logging import setup_logging, get_logger

# Setup logging to file only to avoid cluttering terminal
def run_experiment():
    log_path = setup_logging(log_dir="outputs/logs", log_name="experiment")
    logger = get_logger("EXPERIMENT")
    logger.info(f"Log file: {log_path}")
    
    file_path = 'data/raw/FastFoodNutritionMenuV3.csv'
    target_col = 'calories'
    
    # Parameters to iterate
    num_strategies = ['mean', 'median', 'mode', 'drop']
    scaling_strategies = ['robust', 'standard']
    outlier_methods = ['isolation_forest', 'iqr', 'zscore']
    
    # Constant parameters from main.py
    cat_strategy = 'mode'
    dt_strategy = 'drop'
    
    all_results = []
    
    combinations = list(itertools.product(num_strategies, scaling_strategies, outlier_methods))
    total_runs = len(combinations)
    
    print(f"Starting experiment with {total_runs} configurations...")
    
    for i, (num_strat, scale_strat, outlier_method) in enumerate(combinations, 1):
        print(f"Run {i}/{total_runs}: num={num_strat}, scale={scale_strat}, outlier={outlier_method}")
        logger.info(f"=== Run {i}/{total_runs}: num={num_strat}, scale={scale_strat}, outlier={outlier_method} ===")
        
        try:
            # =========================================================================
            # 1. TIỀN XỬ LÝ SƠ BỘ (SAFE PREPROCESSING) - Giống train.py
            # =========================================================================
            preprocessor = DataPreprocessor(
                num_strategy=num_strat,
                cat_strategy=cat_strategy, 
                dt_strategy=dt_strategy,
                scaling_strategy=scale_strat,
                outlier_method=outlier_method,
            )
            preprocessor.load_data(file_path, auto_convert_numeric=True)
            
            # Chuyển đổi datetime nếu có
            preprocessor.convert_to_datetime()
            
            # Drop cột rác và clean giá trị âm
            preprocessor.drop_features(['calories_from_fat', 'weight_watchers_pnts', 'company', 'item'])
            preprocessor.clean_negative_values()
            
            # Categorical encoding (Làm trước split để đảm bảo đồng bộ cột)
            preprocessor.encode_categorical(strategy='onehot')
            
            # Loại bỏ duplicate trước khi chia train/test
            preprocessor.remove_duplicates()
            
            # =========================================================================
            # 2. CHIA DỮ LIỆU TRAIN/TEST
            # =========================================================================
            current_data = preprocessor.get_processed_data()
            trainer = ModelTrainer(random_state=42)
            trainer.load_data(current_data, target_column=target_col)
            train_df, test_df = trainer.split_data(test_size=0.2)
            
            # =========================================================================
            # 3. XỬ LÝ TẬP TRAIN (FIT & TRANSFORM)
            # =========================================================================
            # Missing: Học từ train -> điền vào train
            train_processed = preprocessor.handle_missing_values(data=train_df, fit=True)
            
            # Outliers: Chỉ loại bỏ trên tập TRAIN
            train_processed = preprocessor.handle_outliers(
                data=train_processed, 
                exclude_features=[target_col] 
            )
            
            # Scaling: Học từ train -> scale train
            train_processed = preprocessor.scale_features(
                data=train_processed, 
                exclude_features=[target_col], 
                fit=True
            )
            
            # =========================================================================
            # 4. XỬ LÝ TẬP TEST (CHỈ TRANSFORM)
            # =========================================================================
            # Missing: Dùng giá trị đã học từ train -> điền vào test
            test_processed = preprocessor.handle_missing_values(data=test_df, fit=False)
            
            # Scaling: Dùng tham số đã học từ train -> scale test
            test_processed = preprocessor.scale_features(
                data=test_processed, 
                exclude_features=[target_col], 
                fit=False
            )
            
            # =========================================================================
            # 5. HUẤN LUYỆN & ĐÁNH GIÁ
            # =========================================================================
            trainer.set_training_data(train_processed, test_processed, target_col=target_col)
            trainer.initialize_models()
            trainer.train_models()
            results_dict = trainer.evaluate_models()
            results = results_dict['results'] # List of dicts
            
            for res in results:
                res['num_strategy'] = num_strat
                res['scaling_strategy'] = scale_strat
                res['outlier_method'] = outlier_method
                all_results.append(res)
                
        except Exception as e:
            logger.error(f"Error in run {i}: {str(e)}")
            print(f"Error in run {i}: {str(e)}")

    # Create DataFrame
    df_results = pd.DataFrame(all_results)

    if df_results.empty:
        logger.error("No experiment results were recorded. Check earlier errors and raw data (target column present?).")
        print("No experiment results were recorded. Please inspect logs for earlier errors before trying again.")
        return
    
    # Sheet 1: Average metrics for all 6 models within each run configuration
    summary_by_config = df_results.groupby(['num_strategy', 'scaling_strategy', 'outlier_method'])[['mse', 'rmse', 'mae', 'r2_score']].mean().reset_index()
    
    # Rename and Reorder Sheet 1
    summary_by_config = summary_by_config.rename(columns={
        'r2_score': 'R2',
        'mse': 'MSE',
        'mae': 'MAE',
        'rmse': 'RMSE'
    })
    summary_by_config = summary_by_config[['num_strategy', 'scaling_strategy', 'outlier_method', 'R2', 'MSE', 'MAE', 'RMSE']]

    # Sheet 4 uses Sheet 1 data sorted by R2 descending
    summary_sorted_by_r2 = summary_by_config.sort_values(by='R2', ascending=False).reset_index(drop=True)
    
    # Sheet 2: Best model for each configuration based on R2 score
    idx = df_results.groupby(['num_strategy', 'scaling_strategy', 'outlier_method'])['r2_score'].idxmax()
    best_model_df = df_results.loc[idx].reset_index(drop=True)
    
    # Rename and Reorder Sheet 2
    best_model_df = best_model_df.rename(columns={
        'model_name': 'Best model',
        'r2_score': 'R2',
        'mse': 'MSE',
        'mae': 'MAE',
        'rmse': 'RMSE'
    })
    best_model_df = best_model_df[['num_strategy', 'scaling_strategy', 'outlier_method', 'Best model', 'R2', 'MSE', 'MAE', 'RMSE']]
    
    # Save to Excel
    output_file = "outputs/results/experiment_results.xlsx"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_by_config.to_excel(writer, sheet_name='Kết quả trung bình', index=False)
        best_model_df.to_excel(writer, sheet_name='Model tốt nhất', index=False)
        df_results.to_excel(writer, sheet_name='Tất cả kết quả', index=False)
        summary_sorted_by_r2.to_excel(writer, sheet_name='Sắp xếp theo R2', index=False)
    print(f"Experiment completed. Results saved to {output_file}")

if __name__ == "__main__":
    run_experiment()
