"""
Machine Learning Pipeline for Fast Food Nutrition Data
"""
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.models.io import ModelIO
from src.visualization.eda import EDA
from src.visualization.model_plots import ModelVisualizer

__all__ = [
    "DataPreprocessor",
    "ModelTrainer",
    "ModelIO",
    "EDA",
    "ModelVisualizer",
]

__version__ = "1.0.0"
