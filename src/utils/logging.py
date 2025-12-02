"""
Centralized logging setup utilities
"""
import logging
import os
from datetime import datetime
from logging import StreamHandler, FileHandler, Formatter


def setup_logging(log_dir="outputs/logs", log_name="pipeline"):
    """
    Thiết lập hệ thống logging tập trung cho toàn bộ project.
    
    Tạo file log với timestamp và cấu hình handlers cho console và file output.
    
    Parameters
    ----------
    log_dir : str, optional
        Thư mục chứa file log. Mặc định là "outputs/logs".
    log_name : str, optional
        Tên prefix cho file log. Mặc định là "pipeline".
    
    Returns
    -------
    str
        Đường dẫn tới file log đã tạo.
    
    Examples
    --------
    >>> from src.utils.logging import setup_logging
    >>> log_path = setup_logging(log_name="experiment")
    >>> logger = logging.getLogger("MY_MODULE")
    >>> logger.info("Starting process...")
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    
    root = logging.getLogger()
    # Clear existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    
    formatter = Formatter("%(asctime)s [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    
    # Console handler
    ch = StreamHandler()
    ch.setFormatter(formatter)
    
    # File handler
    fh = FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    
    root.addHandler(ch)
    root.addHandler(fh)
    
    return log_path


def get_logger(name):
    """
    Lấy logger với tên chỉ định, cấu hình sẵn để sử dụng.
    
    Parameters
    ----------
    name : str
        Tên của logger (thường là tên module).
    
    Returns
    -------
    logging.Logger
        Logger instance đã được cấu hình.
    
    Examples
    --------
    >>> from src.utils.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing data...")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = True
        logger.setLevel(logging.INFO)
    return logger
