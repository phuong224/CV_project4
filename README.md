# Hệ thống Phân tích Giao thông bằng Thị giác Máy tính

Dự án này ứng dụng các kỹ thuật xử lý ảnh và học sâu để phân tích hình ảnh giao thông. Mục tiêu chính là xây dựng một hệ thống có khả năng "hiểu" được cấu trúc không gian của con đường (làn đường, lề đường) và nhận diện các vật thể đang tham gia giao thông (xe cộ, người đi bộ).

Đây là bước tiền đề để trong tương lai có thể phát triển các hệ thống hỗ trợ lái xe thông minh (ADAS), có khả năng đưa ra gợi ý hướng đi hoặc cảnh báo dựa trên môi trường xung quanh.

## 🚀 Quy trình xử lý

Hệ thống hoạt động theo một quy trình gồm 3 bước chính, được triển khai thông qua các module trong thư mục `src/` và minh họa trong các file `notebooks/`.

### 1. Tiền xử lý dữ liệu (`data_handling.py`)
Chuẩn bị và làm sạch dữ liệu ảnh đầu vào để tối ưu hóa hiệu suất cho các bước phân tích sau.
* **Đọc và Chuẩn hóa**: Đồng bộ kích thước đầu vào.
* **Tăng cường tương phản (CLAHE)**: Làm nổi bật vạch kẻ đường trong điều kiện ánh sáng không đồng đều.
* **Giảm nhiễu (Gaussian Blur)**: Loại bỏ các chi tiết nhiễu giúp phát hiện cạnh chính xác hơn.
* **Chuyển đổi Grayscale**: Giảm độ phức tạp tính toán từ ảnh RGB về 1 kênh màu.

### 2. Phân tích Cấu trúc Không gian (`geometry_analysis.py`)
Trích xuất các đặc trưng hình học của con đường để mô tả không gian di chuyển.
* **Vùng quan tâm (ROI)**: Sử dụng mặt nạ (mask) để tập trung xử lý phần đường đi, loại bỏ bầu trời và cảnh quan không liên quan.
* **Phát hiện cạnh (Canny)**: Xác định các biên độ sáng thay đổi đột ngột.
* **Biến đổi Hough (Probabilistic Hough Transform)**: Tìm kiếm các đoạn thẳng từ ảnh biên.
* **Lọc và Tinh chỉnh**: Sử dụng các bộ lọc dựa trên **độ dài** và **góc nghiêng (Angle Filter)** để xác định chính xác làn đường.

### 3. Nhận diện Vật thể (`object_analysis.py`)
Sử dụng học sâu để định nghĩa ngữ nghĩa cho các đối tượng trong cảnh.
* **YOLOv8**: Phát hiện và phân loại các vật thể thời gian thực như `car`, `person`, `truck`, `traffic light`,...
* **Tích hợp**: Sử dụng tọa độ vật thể từ YOLO để loại bỏ các biên giả trong quá trình phân tích hình học.

## 🛠 Công nghệ sử dụng

* **OpenCV-Python**: Xử lý ảnh và thị giác máy tính truyền thống.
* **NumPy**: Phép toán ma trận và tính toán số học hiệu suất cao.
* **Ultralytics YOLOv8**: Framework nhận diện vật thể State-of-the-art.
* **Matplotlib**: Trực quan hóa dữ liệu và kết quả thực nghiệm.

## 📂 Cấu trúc thư mục

```text
.
├── data/               # Dữ liệu ảnh/video đầu vào
├── notebooks/          # Jupyter Notebooks thực thi quy trình chính
├── src/                # Mã nguồn xử lý lõi
│   ├── data_handling.py     # Tiền xử lý ảnh
│   ├── geometry_analysis.py # Phân tích hình học & làn đường
│   └── object_analysis.py   # Nhận diện vật thể YOLO
├── weights/            # Lưu trữ file trọng số mô hình (.pt)
├── results/            # Kết quả đầu ra (Images, Plots)
└── requirements.txt    # Danh sách thư viện phụ thuộc
```

## 🛠 Cài đặt và Sử dụng

### 1. Chuẩn bị môi trường
Dự án yêu cầu **Python 3.8+**. Khuyến khích sử dụng môi trường ảo:

* Tạo và kích hoạt môi trường ảo
```bash
python -m venv venv
```
* Windows:
```bash
venv\Scripts\activate
```
* Linux/MacOS:
```bash
source venv/bin/activate
```
### 2. Cài đặt thư viện
Cài đặt các thư viện cần thiết qua file requirements.txt:
```bash
pip install -r requirements.txt
```

### 3. Thực thi quy trình
- Để thực thi mở các notebooks và chạy chúng (theo thứ tự 01, 02, 03).
- Kết quả sẽ được lưu tại resutls/.
- Các file trong src/ chỉ mang tính chất lưu trữ cho tái sử dụng.

## 📈 Hướng phát triển tương lai
- Dự án hiện tại đã xây dựng được nền tảng vững chắc. Các hướng phát triển tiếp theo bao gồm:
- Hợp nhất dữ liệu nâng cao (Sensor Fusion): Kết hợp tọa độ YOLO và đường thẳng Hough để xác định chính xác vật thể đang nằm ở làn đường nào (Làn trái, phải hay lấn làn).
- Ước tính khoảng cách (Distance Estimation): Sử dụng điểm tụ (vanishing point) và kích thước thực tế của vật thể để tính toán khoảng cách an toàn.
- Theo dõi vật thể (Tracking): Tích hợp DeepSORT để theo dõi quỹ đạo di chuyển của các phương tiện theo thời gian thực.
- Semantic Segmentation: Áp dụng phân đoạn ngữ nghĩa để nhận diện bề mặt đường ở những nơi không có vạch kẻ rõ ràng.

## 🛠 Công nghệ sử dụng
OpenCV-Python, NumPy, Ultralytics YOLOv8, Matplotlib.