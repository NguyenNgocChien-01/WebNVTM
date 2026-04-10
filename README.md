# KIE Invoice Extraction

Du an cua ban la mot he thong `Django + OCR + Graph Neural Network` dung de trich xuat thong tin tu hoa don/chung tu. Ung dung cho phep tai anh len de du doan theo thoi gian thuc, dong thoi ho tro chay thu tren tap du lieu test da xu ly san.
Bao cao: [https://drive.google.com/drive/u/0/folders/1ZhYQm2iUe5e2tNM-z4fLEkglrPAOzDWg](Tại đây)
## Muc tieu

He thong nhan dien va gan nhan cho cac doan van ban tren hoa don thanh 6 nhom:

- `company`
- `address`
- `date`
- `total`
- `item`
- `other`

Ket qua cuoi cung gom:

- anh da ve bounding box va nhan du doan
- thong tin trich xuat theo tung truong
- file log theo lan chay
- file CSV/JSON ket qua phuc vu kiem tra

## Pipeline xu ly

Project dang ket hop 3 lop xu ly chinh:

1. `EasyOCR` de doc text va toa do tu anh.
2. `SentenceTransformer` de tao embedding ngu nghia cho tung text box.
3. `InvoiceGCN` trong `torch-geometric` de phan loai moi node tren do thi chung tu.

Ngoai mo hinh GCN, project con ap dung them mot so rule-based heuristic de sua nhan cho:

- ngay thang
- tong tien
- danh sach mat hang

## Chuc nang hien co

### 1. Chay voi du lieu test

Trang `/test/` cho phep nhap `image_index` de chay du doan tren tap `test_data.dataset` da duoc xu ly truoc.

Luong nay dung cho:

- kiem tra nhanh model hien tai
- xem anh da annotate
- doi chieu thong tin trich xuat voi du lieu mau

### 2. Chay voi anh thuc te

Trang `/real/` cho phep upload anh hoa don de:

- OCR anh
- tao graph dac trung tu ket qua OCR
- suy luan bang model da huan luyen
- ap dung rule-based post-processing
- luu anh ket qua vao `media/results`
- luu chi tiet box vao `media/box`
- ghi log thong ke vao `logs`

## Cau truc thu muc chinh

```text
kie/
|-- home/                 # Django app chua views va routes
|-- kie/                  # Cau hinh project Django
|-- src/                  # Pipeline ML: config, data_processing, model, train, predict
|-- templates/            # Giao dien web cho home, test, real
|-- assets/               # CSS/JS/static files
|-- data/
|   |-- raw/              # Anh, box OCR, labels goc
|   `-- processed/        # train_data.dataset, test_data.dataset
|-- outputs/
|   |-- models/           # Model .pth da huan luyen
|   `-- results/          # JSON/anh ket qua offline
|-- media/
|   |-- uploads/          # Anh upload tu web
|   |-- results/          # Anh annotate sinh ra tren web
|   `-- box/              # CSV chi tiet tung bounding box
|-- logs/                 # Log trich xuat / train / predict
|-- requirements.txt
`-- manage.py
```

## Thanh phan quan trong

- `home/views.py`: xu ly upload anh, OCR, suy luan, hien thi ket qua.
- `src/data_processing.py`: tao do thi tu OCR boxes, tinh feature hinh hoc va text.
- `src/model.py`: dinh nghia mo hinh `InvoiceGCN`/`ChebConv`.
- `src/train.py`: huan luyen model, early stopping, ghi log.
- `src/predict.py`: tai model, suy luan, xuat JSON/anh ket qua.
- `src/config.py`: cau hinh du lieu, labels, tham so model.

## Cau hinh hien tai

Theo `src/config.py`, project dang dung:

- labels: `company`, `address`, `date`, `total`, `item`, `other`
- model save path: `outputs/models/kie_gcn_model_best.pth`
- processed data: `data/processed/train_data.dataset`, `data/processed/test_data.dataset`
- hidden dims: `[512, 256, 128]`
- dropout: `0.3`
- epochs cau hinh: `2000`
- chebnet: `True`
- `K = 3`

## Cach chay

### 1. Cai thu vien

```bash
pip install -r requirements.txt
```

Luu y: code hien tai con import `sentence_transformers`, nhung goi nay chua co trong `requirements.txt`. Neu chay web/inference that bai vi thieu thu vien, can cai them:

```bash
pip install sentence-transformers
```

Neu dung `pytesseract`, may cung can cai Tesseract OCR ben ngoai he thong.

### 2. Chay Django

```bash
python manage.py runserver
```

Sau do truy cap:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/test/`
- `http://127.0.0.1:8000/real/`

## Dieu kien de project hoat dong dung

Ban can dam bao da co san:

- model tai `outputs/models/kie_gcn_model_best.pth`
- dataset da xu ly trong `data/processed`
- du lieu goc trong `data/raw`

Neu thieu model hoac dataset, cac man hinh suy luan se khong load duoc day du.

## Dau ra sinh ra sau khi chay

### Tu web upload anh

- Anh upload: `media/uploads/`
- Anh da annotate: `media/results/`
- CSV box + nhan du doan: `media/box/`
- Log trich xuat: `logs/`

### Tu inference/offline

- JSON ket qua: `outputs/results/`
- Anh annotate: `outputs/results/`
- Model tot nhat: `outputs/models/`

## Diem manh cua du an

- Ket hop OCR, embedding ngu nghia va graph learning.
- Ho tro ca web demo va pipeline ML.
- Co luu log va file trung gian de debug.
- Co them rule-based post-processing de cai thien mot so truong quan trong.

## Han che hien tai

- `requirements.txt` chua day du dependency thuc te.
- Trong repo chua co huong dan dong goi/deploy production.
- Logic tai model trong `home/views.py` con trung lap.
- README cu chua phan anh dung cau truc va cach chay hien tai.

## Tom tat ngan

Day la mot do an trich xuat thong tin hoa don bang KIE, trong do Django cung cap giao dien thao tac, EasyOCR + SentenceTransformer + GCN dam nhiem suy luan, va ket qua duoc xuat ra thanh text co cau truc cung anh da gan nhan de kiem tra truc quan.
