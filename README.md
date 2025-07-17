tải: pip install -r requirements.txt

Chạy: python manage.py runserver

/home để vào trang home


dòng lệnh:
trích: python thamso.py --mode predict --input 1
train: python thamso.py --mode train --epochs 500

python thamso.py --mode train --epochs 500 --lr 0.005
python thamso.py --mode train --epochs 10000 --dropout_rate 0.5
python thamso.py --mode train --hidden_dims 256 128

python main.py --mode train --hidden_dims 256 128 --chebnet --K 4