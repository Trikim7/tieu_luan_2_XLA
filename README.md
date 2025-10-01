# Tiểu luận 2: Image Filters - Xử lý ảnh với Python

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)
[![PyQt5](https://img.shields.io/badge/PyQt5-GUI-orange.svg)](https://pypi.org/project/PyQt5/)

## 🎯 Nội dung các bài tập

### Bài 1 - So sánh và phân tích các bộ lọc làm mờ

**Yêu cầu:**

1. Cài đặt các bộ lọc làm mờ: Mean, Gaussian, Median, Bilateral
2. Thử nghiệm trên nhiều loại nhiễu: Gaussian noise, Salt & Pepper
3. Đánh giá bằng các chỉ số: PSNR, SSIM

**Kết quả mong đợi:**

- Báo cáo so sánh ưu/nhược điểm từng bộ lọc
- Biểu đồ trực quan (histogram trước/sau lọc)

### Bài 2 - Edge Detection (Phát hiện cạnh)

**Yêu cầu:**

1. Cài đặt Sobel, Prewitt, Laplacian từ đầu (không dùng hàm cv2 có sẵn)
2. So sánh kết quả với bộ lọc Canny
3. Ứng dụng pipeline vào ảnh thực tế (ảnh đường phố, ảnh văn bản)

**Kết quả mong đợi:**

- Bộ ảnh minh họa các bước (gradient X, Y, magnitude, threshold)
- Đánh giá độ nhạy của tham số ngưỡng

### Bài 3 - Tăng cường ảnh (Image Enhancement)

**Yêu cầu:**

1. Áp dụng bộ lọc Sharpen (Laplacian, Unsharp Masking)
2. Kết hợp với histogram equalization để cải thiện ảnh mờ/thiếu sáng
3. Đề xuất một workflow tăng cường ảnh chụp từ camera điện thoại

**Kết quả mong đợi:**

- Demo ảnh trước → ảnh sắc nét sau
- Giải thích tại sao filter + histogram equalization hiệu quả

### Bài 4 - Bộ lọc trong xử lý ảnh y tế

**Yêu cầu:**

1. Tìm dataset ảnh X-quang hoặc MRI công khai (Kaggle, NIH)
2. Áp dụng Gaussian smoothing để khử nhiễu
3. Áp dụng Sobel/Canny để phát hiện biên vùng bất thường
4. Báo cáo thảo luận ưu/nhược điểm của các bộ lọc trong ảnh y tế

**Kết quả mong đợi:**

- Ảnh minh họa trước/sau lọc
- Nhận xét độ rõ nét của vùng biên

### Bài 5 - Ứng dụng thực tế: Mini Photo Editor

**Yêu cầu:**

1. Xây dựng ứng dụng Python (CLI hoặc GUI) cho phép người dùng:
   - Làm mờ (Blur, Gaussian, Median)
   - Làm sắc nét (Sharpen)
   - Phát hiện cạnh (Sobel, Laplacian, Canny)
2. Cho phép điều chỉnh tham số kernel, sigma, threshold
3. Xuất kết quả

**Kết quả mong đợi:**

- "Mini Photo Editor bằng Python"

## 🚀 Cài đặt và sử dụng

### Yêu cầu hệ thống

```
Python 3.10+
OpenCV 4.x
NumPy
Matplotlib
scikit-image
PyQt5
```

### Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/Trikim7/tieu_luan_2_XLA.git
cd tieu_luan_2

# Cài đặt packages
pip install -r requirements.txt
```

### Chạy ứng dụng GUI

```bash
# Chạy Photo Editor
python -m src.app
```

### Chạy Jupyter Notebook

```bash
jupyter notebook notebooks/tieu_luan_2.ipynb
```

## 📁 Cấu trúc tiểu luận

```
tieu_luan_2/
├── README.md                    # Hướng dẫn
├── requirements.txt             # Dependencies
├── data/                        # Thư mục chứa ảnh test
├── notebooks/
│   └── tieu_luan_2.ipynb       # Notebook phân tích chi tiết
└── src/                        # Source code
    ├── app.py                  # GUI chính (PyQt5)
    ├── filters.py              # Các bộ lọc (Mean, Gaussian, Sobel, etc.)
    ├── enhancement.py          # Tăng cường ảnh (Sharpen, Histogram)
    ├── metrics.py              # Đánh giá chất lượng (PSNR, SSIM)
    └── utils.py                # Utilities
```

## 🔬 Kết quả nghiên cứu chính

### 1. So sánh bộ lọc làm mờ

| Bộ lọc                   | Nhiễu Gaussian | Nhiễu Salt & Pepper | Tốc độ  |
| -------------------------- | --------------- | -------------------- | ---------- |
| **Mean Filter**      | ⭐⭐⭐          | ⭐⭐                 | ⭐⭐⭐⭐⭐ |
| **Gaussian Filter**  | ⭐⭐⭐⭐        | ⭐⭐                 | ⭐⭐⭐⭐   |
| **Median Filter**    | ⭐⭐            | ⭐⭐⭐⭐⭐           | ⭐⭐⭐     |
| **Bilateral Filter** | ⭐⭐⭐⭐⭐      | ⭐                   | ⭐⭐       |

### 2. Nhận định quan trọng

- **Median Filter** hiệu quả nhất với nhiễu Salt & Pepper (PSNR: 31.45 dB)
- **Bilateral Filter** bảo toàn cạnh tốt nhất nhưng chậm
- **Gaussian Filter** cân bằng giữa chất lượng và tốc độ

### 3. Workflow tối ưu cho Image Enhancement

```
Ảnh mờ/tối → Histogram Equalization → Unsharp Masking → Ảnh sắc nét

```

### 4. Bộ lọc trong xử lý ảnh y tế

- Gaussian smoothing giúp giảm nhiễu hạt, Sobel cung cấp biên kém mượt hơn Canny, Canny tạo đường biên liên tục cho vùng nghi ngờ.
- Ảnh minh họa trước/sau lọc được lưu trữ trong các ô mã Python tương ứng.

## 🎨 Giao diện ứng dụng

Ứng dụng Mini Photo Editor cung cấp:

- **🖼️ Hiển thị ảnh**: So sánh trước/sau xử lý
- **📊 Histogram**: Phân tích phân bố pixel
- **🎛️ Điều khiển tham số**: Kernel size, sigma, threshold
- **📈 Metrics**: PSNR, SSIM real-time
- **💾 Export**: Lưu kết quả

## 🎓 Kiến thức đạt được

- Hiểu sâu về **toán học bộ lọc**: convolution, correlation
- Phân biệt **tuyến tính vs phi tuyến tính**: Mean/Gaussian vs Median
- **Edge detection**: gradient, magnitude, thresholding
- **Image enhancement**: sharpening, histogram equalization
- **Đánh giá định lượng**: PSNR, SSIM
- **Ứng dụng thực tế**: xử lý ảnh y tế, photo editing

## 📚 Tài liệu tham khảo

1. Digital Image Processing - Gonzalez & Woods
2. Computer Vision: Algorithms and Applications - Richard Szeliski
3. OpenCV Documentation
4. scikit-image Documentation

---
