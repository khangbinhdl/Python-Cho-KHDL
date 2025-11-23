import pandas as pd
import logging
import os
import itertools
from Preprocessing import DataPreprocessor
from Model import ModelTrainer

# Setup logging to file only to avoid cluttering terminal
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    # Clear existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    logging.basicConfig(
        filename="logs/experiment.log",
        level=logging.INFO,
        format="%(asctime)s [%(name)s]: %(message)s",
        filemode="w"
    )

def run_experiment():
    setup_logging()
    logger = logging.getLogger("EXPERIMENT")
    
    file_path = './FastFoodNutritionMenuV3.csv'
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
            # 1. TIỀN XỬ LÝ SƠ BỘ (SAFE PREPROCESSING)
            # =========================================================================
            preprocessor = DataPreprocessor(
                num_strategy=num_strat,
                cat_strategy=cat_strategy, 
                dt_strategy=dt_strategy,
                scaling_strategy=scale_strat,
                outlier_method=outlier_method,
            )
            preprocessor.load_data(file_path)

            # =========================================================================
            # 3. Chia dữ liệu TRAIN/TEST, TIẾN HÀNH XỬ LÝ DỮ LIỆU
            # =========================================================================

            # Drop cột rác và clean giá trị âm
            preprocessor.drop_features(['calories_from_fat', 'weight_watchers_pnts', 'company', 'item'])
            preprocessor.clean_negative_values()
            
            # One-hot encoding (Làm trước split để đảm bảo đồng bộ cột)
            preprocessor.encode_categorical(strategy='onehot')

            # Loại bỏ duplicate trước khi chia train/test
            preprocessor.remove_duplicates()
            
            # Lấy dữ liệu tạm thời (đã clean cơ bản)
            current_data = preprocessor.get_processed_data()

            trainer = ModelTrainer(random_state=42)
            
            # Nạp dữ liệu vào ModelTrainer
            trainer.load_data(current_data, target_column=target_col)
            
            # GỌI HÀM CỦA CLASS ĐỂ CHIA TRAIN/TEST
            train_df, test_df = trainer.split_data(test_size=0.2)

            # --- Xử lý tập TRAIN (FIT & TRANSFORM) ---
            # 1. Missing
            train_processed = preprocessor.handle_missing_values(data=train_df, fit=True)
            
            # 2. Outliers
            train_processed = preprocessor.handle_outliers(
                data=train_processed, 
                exclude_features=[target_col] 
            )
            
            # 3. Scaling
            train_processed = preprocessor.scale_features(
                data=train_processed, 
                exclude_features=[target_col], 
                fit=True
            )

            # --- Xử lý tập TEST (CHỈ TRANSFORM) ---
            # 1. Missing
            test_processed = preprocessor.handle_missing_values(data=test_df, fit=False)
            
            # 2. Scaling
            test_processed = preprocessor.scale_features(
                data=test_processed, 
                exclude_features=[target_col], 
                fit=False
            )

            # =========================================================================
            # 5. HUẤN LUYỆN & ĐÁNH GIÁ
            # =========================================================================
            
            # Nạp dữ liệu sạch ngược lại vào Trainer để tách X, y
            trainer.set_training_data(train_processed, test_processed, target_col=target_col)
            
            # Khởi tạo các mô hình
            trainer.initialize_models()
            
            # Train tất cả models
            trainer.train_models()
            
            # Đánh giá và so sánh tất cả models
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
    
    # Sheet 1: Average metrics for all 6 models (grouped by configuration)
    summary_by_config = df_results.groupby(['num_strategy', 'scaling_strategy', 'outlier_method'])[['mse', 'rmse', 'mae', 'r2_score']].mean().reset_index()
    
    # Rename and Reorder Sheet 1
    summary_by_config = summary_by_config.rename(columns={
        'r2_score': 'R2',
        'mse': 'MSE',
        'mae': 'MAE',
        'rmse': 'RMSE'
    })
    summary_by_config = summary_by_config[['num_strategy', 'scaling_strategy', 'outlier_method', 'R2', 'MSE', 'MAE', 'RMSE']]
    
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
    output_file = "results/experiment_results.xlsx"
    os.makedirs("results", exist_ok=True)
    
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            summary_by_config.to_excel(writer, sheet_name='Average_Metrics', index=False)
            best_model_df.to_excel(writer, sheet_name='Best_Model', index=False)
            df_results.to_excel(writer, sheet_name='All_Results', index=False)
        print(f"Experiment completed. Results saved to {output_file}")
    except ImportError:
        print("Error: openpyxl is not installed. Saving to CSVs instead.")
        summary_by_config.to_csv("results/experiment_average_metrics.csv", index=False)
        best_model_df.to_csv("results/experiment_best_model.csv", index=False)
        df_results.to_csv("results/experiment_all_results.csv", index=False)
        print("Results saved to CSV files in results/ directory.")

if __name__ == "__main__":
    run_experiment()
