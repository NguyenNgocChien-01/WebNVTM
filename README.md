# KIE Invoice Extraction

Dự án của này là một hệ thống `Django + OCR + Graph Neural Network` dùng để trích xuất thông tin từ hóa đơn/chứng từ. Ứng dụng cho phép tải ảnh lên để dự đoán theo thời gian thực, đồng thời hỗ trợ chạy thử trên tập dữ liệu test đã xử lý sẵn.
Báo cáo: [Tại đây]([https://drive.google.com/drive/u/0/folders/1ZhYQm2iUe5e2tNM-z4fLEkglrPAOzDWg](https://drive.google.com/file/d/1S3e0TNrxIuMnMEU0tFkAFsZ35YM5J2au/view?usp=drive_link))

## Mục tiêu

Hệ thống nhận diện và gán nhãn cho các đoạn văn bản trên hóa đơn thành 6 nhóm:

- `company`
- `address`
- `date`
- `total`
- `item`
- `other`

Kết quả cuối cùng gồm:

- ảnh đã vẽ bounding box và nhãn dự đoán
- thông tin trích xuất theo từng trường
- file log theo lần chạy
- file CSV/JSON kết quả phục vụ kiểm tra

## Pipeline xử lý

Project đang kết hợp 3 lớp xử lý chính:

1. `EasyOCR` để đọc text và tọa độ từ ảnh.
2. `SentenceTransformer` để tạo embedding ngữ nghĩa cho từng text box.
3. `InvoiceGCN` trong `torch-geometric` để phân loại mỗi node trên đồ thị chứng từ.

Ngoài mô hình GCN, project còn áp dụng thêm một số rule-based heuristic để sửa nhãn cho:

- ngày tháng
- tổng tiền
- danh sách mặt hàng

## Chức năng hiện có

### 1. Chạy với dữ liệu test

Trang `/test/` cho phép nhập `image_index` để chạy dự đoán trên tập `test_data.dataset` đã được xử lý trước.

Luồng này dùng cho:

- kiểm tra nhanh model hiện tại
- xem ảnh đã annotate
- đối chiếu thông tin trích xuất với dữ liệu mẫu

### 2. Chạy với ảnh thực tế

Trang `/real/` cho phép upload ảnh hóa đơn để:

- OCR ảnh
- tạo graph đặc trưng từ kết quả OCR
- suy luận bằng model đã huấn luyện
- áp dụng rule-based post-processing
- lưu ảnh kết quả vào `media/results`
- lưu chi tiết box vào `media/box`
- ghi log thống kê vào `logs`

## Cấu trúc thư mục chính

```text
kie/
|-- home/                 # Django app chứa views và routes
|-- kie/                  # Cấu hình project Django
|-- src/                  # Pipeline ML: config, data_processing, model, train, predict
|-- templates/            # Giao diện web cho home, test, real
|-- assets/               # CSS/JS/static files
|-- data/
|   |-- raw/              # Ảnh, box OCR, labels gốc
|   `-- processed/        # train_data.dataset, test_data.dataset
|-- outputs/
|   |-- models/           # Model .pth đã huấn luyện
|   `-- results/          # JSON/ảnh kết quả offline
|-- media/
|   |-- uploads/          # Ảnh upload từ web
|   |-- results/          # Ảnh annotate sinh ra trên web
|   `-- box/              # CSV chi tiết từng bounding box
|-- logs/                 # Log trích xuất / train / predict
|-- requirements.txt
`-- manage.py
