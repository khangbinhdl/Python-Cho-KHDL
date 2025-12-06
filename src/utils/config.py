"""
Configuration loading utilities
"""
from __future__ import annotations

import configparser
import os
from argparse import Namespace
from typing import Any, Optional


def load_config(
    config_path: str,
    args: Optional[Namespace] = None
) -> configparser.ConfigParser:
    """
    Tải cấu hình từ file .ini và kết hợp với arguments từ command line.
    
    Parameters
    ----------
    config_path : str
        Đường dẫn tới file cấu hình .ini.
    args : argparse.Namespace or None, optional
        Arguments từ command line parser. Nếu có, sẽ ghi đè các giá trị trong config.
        Mặc định là None.
    
    Returns
    -------
    configparser.ConfigParser
        Object ConfigParser đã được load và merge với args.
    
    Raises
    ------
    FileNotFoundError
        Nếu file cấu hình không tồn tại.
    
    Examples
    --------
    >>> from src.utils.config import load_config
    >>> config = load_config("configs/default.ini")
    >>> data_file = config.get('PATHS', 'data_file')
    """
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found!")
    
    config.read(config_path, encoding='utf-8')
    
    # Merge with command line arguments if provided
    if args:
        # Ghi đè với các tham số dòng lệnh
        if hasattr(args, 'data') and args.data:
            config.set('PATHS', 'data_file', args.data)
        if hasattr(args, 'target') and args.target:
            config.set('DATA', 'target_column', args.target)
        if hasattr(args, 'test_size') and args.test_size:
            config.set('DATA', 'test_size', str(args.test_size))
        if hasattr(args, 'random_state') and args.random_state:
            config.set('DATA', 'random_state', str(args.random_state))
        if hasattr(args, 'optimize') and args.optimize is not None:
            config.set('OPTIMIZATION', 'enable_optimization', str(args.optimize).lower())
        if hasattr(args, 'eda') and args.eda is not None:
            config.set('VISUALIZATION', 'enable_eda', str(args.eda).lower())
        if hasattr(args, 'plot') and args.plot is not None:
            config.set('VISUALIZATION', 'enable_plots', str(args.plot).lower())
        if hasattr(args, 'models') and args.models:
            config.set('MODEL', 'selected_models', args.models)
        
        # Preprocessing arguments
        if hasattr(args, 'num_strategy') and args.num_strategy:
            config.set('PREPROCESSING', 'num_strategy', args.num_strategy)
        if hasattr(args, 'cat_strategy') and args.cat_strategy:
            config.set('PREPROCESSING', 'cat_strategy', args.cat_strategy)
        if hasattr(args, 'dt_strategy') and args.dt_strategy:
            config.set('PREPROCESSING', 'dt_strategy', args.dt_strategy)
        if hasattr(args, 'scaler') and args.scaler:
            config.set('PREPROCESSING', 'scaler', args.scaler)
        if hasattr(args, 'outlier') and args.outlier:
            config.set('PREPROCESSING', 'outlier', args.outlier)
        if hasattr(args, 'encoder') and args.encoder:
            config.set('PREPROCESSING', 'encoder', args.encoder)
        if hasattr(args, 'drop_features') and args.drop_features:
            config.set('PREPROCESSING', 'drop_features', args.drop_features)
        if hasattr(args, 'clean_negative') and args.clean_negative is not None:
            config.set('PREPROCESSING', 'clean_negative_values', str(args.clean_negative).lower())
    
    return config


def get_config_value(
    config: configparser.ConfigParser,
    section: str,
    key: str,
    default: Optional[Any] = None
) -> Any:
    """
    Lấy giá trị từ config với xử lý exception và giá trị mặc định.
    
    Parameters
    ----------
    config : configparser.ConfigParser
        Config object.
    section : str
        Tên section trong file config.
    key : str
        Tên key cần lấy giá trị.
    default : any, optional
        Giá trị mặc định nếu không tìm thấy. Mặc định là None.
    
    Returns
    -------
    str or default
        Giá trị từ config, hoặc default nếu không tìm thấy.
    
    Examples
    --------
    >>> data_file = get_config_value(config, 'PATHS', 'data_file', 'data/raw/default.csv')
    """
    try:
        return config.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return default
