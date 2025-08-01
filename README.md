clone:
```bash
git clone https://github.com/NguyenNgocChien-01/WebNVTM.git
```

tải: 
```bash
pip install -r requirements.txt
```
Chạy: 
Nếu dự án có giao diện web, bạn có thể khởi động Web Server bằng lệnh sau:
```bash
python manage.py runserver
```

/home để vào trang home


dòng lệnh:

để train 
```bash

train: python thamso.py --mode train --epochs 500
python thamso.py --mode train --epochs 500 --lr 0.005
python thamso.py --mode train --epochs 10000 --dropout_rate 0.5
python thamso.py --mode train --hidden_dims 256 128
python main.py --mode train --hidden_dims 256 128 --chebnet --K 4
```
