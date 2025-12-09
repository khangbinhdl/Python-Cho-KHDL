"""
DataTransformer module - Backward compatible wrapper.

Module này giữ lại interface cũ để backward compatible với code hiện tại.
Logic đã được tách ra thành các class con trong src/data/transformers/.

Cấu trúc mới:
- TypeConverter: Phát hiện và chuyển đổi kiểu dữ liệu
- MissingValueHandler: Xử lý giá trị thiếu
- OutlierHandler: Xử lý ngoại lai
- FeatureEncoder: Mã hóa categorical
- FeatureScaler: Chuẩn hóa features
- DataCleaner: Các thao tác làm sạch dữ liệu
- DataTransformer: Facade class tổng hợp
"""

# Re-export tất cả từ module mới để backward compatible
from src.data.transformers import (
	TypeConverter,
	MissingValueHandler,
	OutlierHandler,
	FeatureEncoder,
	FeatureScaler,
	DataCleaner,
	DataTransformer,
)

__all__ = [
	'TypeConverter',
	'MissingValueHandler',
	'OutlierHandler',
	'FeatureEncoder',
	'FeatureScaler',
	'DataCleaner',
	'DataTransformer',
]
