# Yêu cầu
- Tất cả hàm phải sử dụng docstring numpy format.
- Sử dụng seaborn để trực quan hóa.

# Link dataset: 
[Kaggle](https://www.kaggle.com/datasets/tan5577/nutritonal-fast-food-dataset)

# Mô tả dữ liệu
| Cột | Mô tả |
|:----|:-----|
| **Company** | Tên công ty/thương hiệu sản xuất mặt hàng. |
| **Item** | Tên của sản phẩm/món ăn cụ thể. |
| **Calories** | Tổng lượng calo (năng lượng) trong một khẩu phần sản phẩm. |
| **Calories from Fat** | Lượng calo đến từ chất béo trong một khẩu phần. |
| **Total Fat (g)** | Tổng lượng chất béo (gram) trong một khẩu phần. |
| **Saturated Fat (g)** | Lượng chất béo bão hòa (gram) trong một khẩu phần. |
| **Trans Fat (g)** | Lượng chất béo chuyển hóa (trans fat - gram) trong một khẩu phần. |
| **Cholesterol (mg)** | Lượng Cholesterol (miligram) trong một khẩu phần. |
| **Sodium (mg)** | Lượng Natri/Muối (miligram) trong một khẩu phần. |
| **Carbs (g)** | Tổng lượng Carbohydrate (gram) trong một khẩu phần. |
| **Fiber (g)** | Lượng chất xơ (gram) trong một khẩu phần. |
| **Sugars (g)** | Lượng đường (gram) trong một khẩu phần. |
| **Protein (g)** | Lượng Protein (gram) trong một khẩu phần. |
| **Weight Watchers Pnts** | Điểm số theo hệ thống tính điểm của chương trình ăn kiêng Weight Watchers (có thể đã lỗi thời hoặc chỉ áp dụng cho một số thị trường). |

# Tiến độ công việc
## Preprocessing
- [x] Đọc dữ liệu từ file khác nhau bằng pandas, (bổ sung thêm xóa các cột trùng lặp, chuẩn hóa tên cột).
- [x] Kiểm tra và xử lí dữ liệu thiếu, ngoại lai (bổ sung thêm cách chọn clip hoặc xóa).
- [x] Chuẩn hóa dữ liệu.
- [x] Mã hóa các biến phân loại.
- [ ] Tạo đặc trưng mới (đéo biết tạo cc gì).
- [x] Tự động phát hiện kiểu dữ liệu và áp dụng phương pháp xử lí phù hợp.
- [x] Ghi dữ liệu ra file mới.


## ModelTrainer
- [ ] Nạp dữ liệu đã chuẩn hóa.
- [ ] Chia train/test bằng train_test_split.
- [ ] Huấn luyện mô hình. 
- [ ] Tối ưu siêu tham số. (Sử dụng file notebook ở 'Bayesian_Hyperparameters_tuning.ipynb' chạy local mới nhanh do colab có 2 nhân CPU).
- [ ] Đánh giá mô hình theo thước đo phù hợp.
- [ ] Lưu và nạp mô hình bằng joblib hoặc pickle.
- [ ] Ghi lại các kết quả thực nghiệm (MSE, MAE, R2) vào file CSV hoặc JSON.
- [ ] So sánh kết quả giữa các mô hình và lưu biểu đồ đánh giá (barplot performance).
- [ ] Có cơ chế tự động thử nhiều mô hình và chọn mô hình tốt nhất. (Hạn chế sử dụng SVM tại vì tao thấy nó train hơi lâu, sử dụng cái tìm tham số mệt).
- [ ] Cố định random seed.

## EDA
- [x] Thực hiện trực quan dữ liệu đầu vào (và sau khi xử lí).
- [ ] Thực hiện trực quan kết quả mô hình.
- [ ] Phân tích, nhận xét, và trình bày ý nghĩa kết quả theo hướng giải thích được mô hình.
- [ ] Có ít nhất một phần thể hiện việc phân tích đặc trưng quan trọng (Feature Importance, SHAP, hoặc Partial Dependence Plot). (Sử dụng Decision Tree hoặc Random Forest [thamkhảo](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)).

## Parser và hướng dẫn chạy
- Chưa làm, để tao nghiên cứu sau.

# Colab version
[Colab](https://colab.research.google.com/drive/1wnFwnhGz468_KIYh5ZvwIaOxmUbZdGJc?usp=sharing) (cái này tao để gemini nó sửa khúc logging để chạy cho đúng nên không biết nó có sửa bậy gì không)
