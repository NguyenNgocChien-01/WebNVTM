import os
import time
import datetime
import json
import cv2
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from django.shortcuts import render
from django.conf import settings
from django.contrib import messages
import torch.nn.functional as F
from django.core.files.storage import FileSystemStorage
import uuid
import pytesseract
from src.config import CONFIG
from src.data_processing import Grapher, load_data
from src.model import InvoiceGCN
from src.predict import load_inference_model, write_evaluation_log 
from torch_geometric.data import Data
import easyocr
import re
import traceback 

def trangchu(request):
    return render(request,"index.html")

def run_real(request):
    return render(request,"real.html")

def run_test(request):
    return render(request,"test.html")

from sentence_transformers import SentenceTransformer

# --- 1. IMPORT CÁC THÀNH PHẦN TỪ DỰ ÁN ---
from src.config import CONFIG
from src.data_processing import Grapher
from src.model import InvoiceGCN
from src.predict import load_inference_model

# --- 2. KHỞI TẠO CÁC MODEL (Tải 1 lần duy nhất) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OCR_READER = None
SENT_MODEL = None
MODEL = None
TEST_DATA = None

try:
    _, TEST_DATA = load_data(CONFIG)
    # Tải Model EasyOCR
    OCR_READER = easyocr.Reader(['vi', 'en'], gpu=torch.cuda.is_available())
    print("✅ EasyOCR đã sẵn sàng!")

    # Tải mô hình ngôn ngữ SentenceTransformer
    SENT_MODEL = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device=DEVICE)
    print("✅ SentenceTransformer đã sẵn sàng!")

    # Tải Model GCN với đúng input_dim
    input_dim = 778
    MODEL = load_inference_model(input_dim, CONFIG, DEVICE)
    print(f"✅ Django: Model KIE đã được tải thành công với input_dim={input_dim}!")

except Exception as e:
    print(f"❌ LỖI KHỞI TẠO MODEL: {e}. Chức năng dự đoán có thể không hoạt động.")



# --- 2. TẢI MODEL VÀ DỮ LIỆU CHUNG (Tải 1 lần khi server khởi động) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None


try:
    _, TEST_DATA = load_data(CONFIG)
    
    model_filename = os.path.basename(CONFIG['model_save_path'])
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'outputs', 'models', model_filename)
    
    input_dim = TEST_DATA.num_node_features if TEST_DATA else 9
    MODEL = load_inference_model(input_dim, CONFIG, DEVICE)
    
    if MODEL and TEST_DATA:
        print(f"✅ Django: Model KIE và Test Data đã được tải thành công!")
except Exception as e:
    print(f"❌ LỖI KHI TẢI MODEL/DATA: {e}.")











# Chạy với index

def run_and_get_results(image_index):

    start_time = time.time()
    
    # Lấy dữ liệu cho ảnh được chọn
    single_graph_data = TEST_DATA.to_data_list()[image_index]
    img_id = single_graph_data.img_id
    original_image_path = os.path.join(settings.BASE_DIR, 'data/raw/img', f'{img_id}.jpg')
    if not os.path.exists(original_image_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh gốc: {original_image_path}")

    # Chạy dự đoán
    with torch.no_grad():
        out = MODEL(single_graph_data.to(DEVICE))
        pred_indices = out.max(dim=1)[1].cpu().numpy()

    # Xử lý kết quả và vẽ lên ảnh
    label_map = {i: label for i, label in enumerate(CONFIG['labels'])}
    predicted_labels = [label_map.get(i, 'error') for i in pred_indices]
    
    grapher = Grapher(filename=img_id, data_fd=os.path.join(settings.BASE_DIR, 'data/raw'))
    df_vis = grapher.get_df_for_visualization()
    df_vis['predicted_label'] = predicted_labels
    
    image_to_draw = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)

    extracted_info = defaultdict(list)
    for _, row in df_vis.iterrows():
        if row['predicted_label'] != 'other':
            extracted_info[row['predicted_label'].upper()].append(row['Object'])
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(image_to_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image_to_draw, row['predicted_label'].upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Định dạng lại các trường Address, Company
    formatted_info = {key: ' '.join(value) if key in ['ADDRESS', 'COMPANY'] else value for key, value in extracted_info.items()}

    # Lưu ảnh kết quả và tạo URL
    result_filename = f"{img_id}_annotated.jpg"
    result_folder = os.path.join(settings.MEDIA_ROOT, 'results')
    os.makedirs(result_folder, exist_ok=True)
    result_image_path = os.path.join(result_folder, result_filename)
    cv2.imwrite(result_image_path, cv2.cvtColor(image_to_draw, cv2.COLOR_RGB2BGR))
    result_image_url = os.path.join(settings.MEDIA_URL, 'results', result_filename).replace("\\", "/")

    #  ghi log 
    y_true = single_graph_data.y.cpu().numpy() - 1
    val_loss = F.nll_loss(out, torch.tensor(y_true, device=DEVICE)).item()
    processing_time = time.time() - start_time
    write_evaluation_log(img_id, CONFIG, formatted_info, original_image_path, y_true, pred_indices, val_loss=val_loss, processing_time=processing_time)

    return formatted_info, result_image_url



def run_by_index_view(request):
    context = {}
    if request.method == 'POST':
        try:
            image_index_str = request.POST.get('image_index', '').strip()
            if not image_index_str:
                raise ValueError("Vui lòng nhập chỉ số ảnh.")

            image_index = int(image_index_str)

            if not (MODEL and TEST_DATA):
                raise Exception("Model hoặc dữ liệu test chưa được tải thành công.")
            
            if not (0 <= image_index < len(TEST_DATA.img_id)):
                raise IndexError(f"Chỉ số ảnh không hợp lệ. Phải nằm trong khoảng 0 đến {len(TEST_DATA.img_id) - 1}.")

            # Gọi hàm helper và nhận kết quả
            extracted_data, annotated_url = run_and_get_results(image_index)
            
            # Chuẩn bị dữ liệu cho template (để tránh lỗi is_list)
            results_for_template = []
            for key, value in extracted_data.items():
                results_for_template.append({
                    'key': key,
                    'value': value,
                    'is_list': isinstance(value, list)
                })

            context['result_data'] = {
                'extracted_text': results_for_template,
                'annotated_image_url': annotated_url,
            }
            img_id = TEST_DATA.img_id[image_index]
            messages.success(request, f"Đã trích xuất thành công cho ảnh '{img_id}' (index: {image_index})!")

        except Exception as e:
            messages.error(request, f"Lỗi: {e}")

    return render(request, 'test.html', context)





# Ảnh thật
def find_total_with_rules(df):
    """
    Tìm tổng tiền cuối cùng bằng cách chấm điểm các ứng viên
    dựa trên từ khóa mạnh và vị trí.
    """
    # Các từ khóa được chấm điểm (điểm càng cao càng đáng tin)
    strong_keywords = ['grand total', 'tổng cộng thanh toán', 'thanh toán', 'tổng tiền', 'total amount']
    medium_keywords = ['total', 'tổng cộng', 'cộng']
    
    possible_totals = []

    # 1. Lọc ra tất cả các dòng có vẻ là số tiền
    # Regex để tìm các số có thể chứa dấu chấm, phẩy
    numeric_rows = df[df['Object'].astype(str).str.match(r'^\$?[\d,.]+$', na=False)]

    for idx, num_row in numeric_rows.iterrows():
        score = 0
        text_value = str(num_row['Object'])
        
        # Bỏ qua các số quá nhỏ hoặc có vẻ là số lượng
        if '.' not in text_value and ',' not in text_value and len(text_value) < 3:
            continue
            
        # 2. Tìm từ khóa ở bên trái trên cùng dòng
        line_ymin, line_ymax = num_row['ymin'] - 5, num_row['ymax'] + 5
        line_df = df[df['ymin'].between(line_ymin, line_ymax)]
        
        # Tìm text bên trái của số
        text_on_left = line_df[line_df['xmax'] < num_row['xmin']]['Object'].str.cat(sep=' ').lower()

        # 3. Chấm điểm dựa trên từ khóa và vị trí
        if any(keyword in text_on_left for keyword in strong_keywords):
            score += 10
        elif any(keyword in text_on_left for keyword in medium_keywords):
            score += 5
            
        # Thưởng điểm cho các số nằm ở nửa dưới của hóa đơn
        if num_row['ymin'] > (df['ymax'].max() / 2):
            score += 2
        
        # Chỉ xem xét các ứng viên có điểm > 0
        if score > 0:
            try:
                # Chuyển text thành số để có thể so sánh nếu cần
                value = float(text_value.replace(',', '').replace('$', ''))
                possible_totals.append({'score': score, 'value': value, 'index': idx})
            except:
                continue
    
    if not possible_totals:
        return None

    # 4. Sắp xếp các ứng viên theo điểm số giảm dần
    sorted_totals = sorted(possible_totals, key=lambda k: k['score'], reverse=True)
    
    # Trả về index của ứng viên có điểm cao nhất
    return sorted_totals[0]['index']

def find_dates_with_rules(df):
    """Tìm ngày tháng bằng biểu thức chính quy (regex)."""
    date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
    date_indices = []
    
    date_rows = df[df['Object'].str.contains(date_pattern, na=False)]
    for idx, row in date_rows.iterrows():
        date_indices.append(idx)
        
    return date_indices
def find_items_with_rules(df):
    """
    Tìm các mục hàng hóa (item) dựa vào vị trí của các từ khóa tiêu đề.
    """
    # Các từ khóa thường xuất hiện ở đầu bảng item
    header_keywords = ['item', 'desc', 'qty', 'sl', 'price', 'đơn giá', 'amount', 't.tiền', 'thành tiền']
    
    # Tìm vị trí (chiều cao) của các từ khóa này
    header_rows = df[df['Object'].str.contains('|'.join(header_keywords), case=False, na=False)]
    
    if header_rows.empty:
        return [] # Không tìm thấy tiêu đề, không thể xác định vùng item

    # Xác định vùng bắt đầu của danh sách item (ngay bên dưới dòng tiêu đề thấp nhất)
    start_y = header_rows['ymax'].max()

    # Tìm vị trí của các từ khóa kết thúc danh sách item (ví dụ: "Subtotal", "Total")
    footer_keywords = ['subtotal', 'sub-total', 'tổng', 'total', 'grand total']
    footer_rows = df[df['Object'].str.contains('|'.join(footer_keywords), case=False, na=False)]
    
    end_y = float('inf')
    if not footer_rows.empty:
        # Vùng item kết thúc ngay phía trên dòng "Total" đầu tiên
        end_y = footer_rows['ymin'].min()

    # Lấy tất cả các dòng nằm giữa vùng bắt đầu và kết thúc
    item_df = df[(df['ymin'] > start_y) & (df['ymax'] < end_y)]
    
    # Trả về index của các dòng được xác định là item
    return item_df.index.tolist()

def apply_rules_to_predictions(df_with_predictions):
    """
    Hàm chính để áp dụng các luật và ghi đè dự đoán của model.
    """
    df = df_with_predictions.copy()
    
    # 1. Áp dụng luật tìm ngày tháng
    date_indices = find_dates_with_rules(df)
    if date_indices:
        print(f"[RULES] Đã tìm thấy {len(date_indices)} ngày tháng. Ghi đè nhãn...")
        df.loc[date_indices, 'predicted_label'] = 'date'
        
    # 2. Áp dụng luật tìm tổng tiền
    total_index = find_total_with_rules(df)
    if total_index is not None:
        print(f"[RULES] Đã tìm thấy tổng tiền. Ghi đè nhãn...")
        df.loc[total_index, 'predicted_label'] = 'total'
        
        # 3. ⭐ ÁP DỤNG LUẬT TÌM ITEM (THÊM MỚI)
    # ========================================
    item_indices = find_items_with_rules(df)
    if item_indices:
        print(f"[RULES] Đã tìm thấy {len(item_indices)} mục item. Ghi đè nhãn...")
        # Ghi đè nhãn cho tất cả các dòng đã tìm thấy
        df.loc[item_indices, 'predicted_label'] = 'item'
    # ========================================
    
    return df






def save_results_to_csv(dataframe_with_predictions, formatted_filename_base):
    """
    Lưu DataFrame ra file CSV với tên file đã được định dạng sẵn.
    """
    debug_dir = os.path.join(settings.MEDIA_ROOT, 'box')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Sử dụng trực tiếp tên file được truyền vào
    csv_path = os.path.join(debug_dir, f"{formatted_filename_base}_details.csv")
    
    try:
        columns_to_save = ['xmin', 'ymin', 'xmax', 'ymax', 'Object', 'predicted_label']
        df_to_save = dataframe_with_predictions[columns_to_save]
        df_to_save.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ Đã lưu file kiểm tra tại: {csv_path}")
        return csv_path
    except Exception as e:
        print(f"Lỗi khi lưu file CSV: {e}")
        return None
# Ảnh thật


# --- Hàm OCR ---
def run_easyocr_with_coords(image_path):
    image = cv2.imread(image_path)
    if image is None: raise FileNotFoundError(f"OpenCV không thể đọc ảnh: {image_path}")


    
    print("--- [EasyOCR] Đang xử lý ảnh đã được tiền xử lý... ---")
    # ⭐ SỬA LỖI: Chỉ gọi readtext MỘT LẦN trên ảnh đã xử lý
    results = OCR_READER.readtext(image)
    
    ocr_list = []
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        ocr_list.append([int(tl[0]), int(tl[1]), int(br[0]), int(br[1]), text])
    return pd.DataFrame(ocr_list, columns=['xmin', 'ymin', 'xmax', 'ymax', 'Object'])

def create_data_object(ocr_dataframe, image_object, file_id):
    """
    Sử dụng Grapher và SentenceTransformer để tạo Data Object với 778 features.
    """
    # B1: Dùng Grapher để lấy cấu trúc đồ thị và các feature cơ bản
    grapher = Grapher(ocr_dataframe=ocr_dataframe, image_object=image_object, filename=file_id)
    G, _, df_processed = grapher.graph_formation(use_clean_df=True)
    df_features = grapher.relative_distance()
    df_features = grapher.get_text_features(df_features)
    
    # B2: Dùng SentenceTransformer để tạo 768 features ngữ nghĩa
    text_to_embed = df_features['Object'].fillna('').tolist()
    text_embeddings = SENT_MODEL.encode(text_to_embed, show_progress_bar=False, device=DEVICE)

    # B3: Nối 2 loại feature lại với nhau
    
    # ⭐ THÊM 'line_number' VÀO DANH SÁCH NÀY
    base_feature_cols = [
        'rd_r', 'rd_b', 'rd_l', 'rd_t', 
        'line_number', # <-- Feature bị thiếu
        'n_upper', 'n_alpha', 'n_spaces', 'n_numeric', 'n_special'
    ]
    
    base_features = df_features[base_feature_cols].values.astype(np.float32)
    
    # Nối theo chiều ngang (cột) để tạo ra 10 + 768 = 778 features
    final_features = np.concatenate((base_features, text_embeddings), axis=1)
    
    # B4: Tạo Data Object
    adj = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor(final_features, dtype=torch.float)
    
    return Data(x=x, edge_index=adj, img_id=file_id), df_features

# --- VIEW CHÍNH (ĐÃ ĐƯỢC SỬA LẠI HOÀN CHỈNH) ---
def ocr_and_predict_view(request):
    context = {}
    if request.method == 'POST':
        image_file = request.FILES.get("invoice_image")
        
        if not (image_file and MODEL):
            messages.error(request, "Lỗi: Chưa chọn file hoặc model chưa được tải.")
            return render(request, 'real.html', context)
        
        # --- PHẦN LƯU FILE ĐÃ ĐƯỢC CHUẨN HÓA ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        original_filename_base = os.path.splitext(image_file.name)[0]
        safe_filename_base = "".join([c for c in original_filename_base if c.isalnum() or c in (' ', '_')]).rstrip()
        unique_name = f"{timestamp}_{safe_filename_base}"
        
        fs = FileSystemStorage()
        saved_filename = fs.save(f"uploads/{unique_name}.jpg", image_file)
        uploaded_file_path = fs.path(saved_filename)
        context['original_image_url'] = fs.url(saved_filename)
        # --- KẾT THÚC PHẦN LƯU FILE ---

        try:
            # B1: Chạy OCR
            ocr_dataframe = run_easyocr_with_coords(uploaded_file_path)
            if ocr_dataframe.empty: raise ValueError("OCR không nhận diện được văn bản.")

            # B2: Tạo Data Object
            image_obj = cv2.imread(uploaded_file_path)
            single_data, df_with_features = create_data_object(ocr_dataframe, image_obj, unique_name)

            # B3: Chạy dự đoán
            with torch.no_grad():
                out = MODEL(single_data.to(DEVICE))
                pred_indices = out.max(dim=1)[1].cpu().numpy()
            
            # B4: Gán nhãn và áp dụng luật
            label_map = {i: label for i, label in enumerate(CONFIG['labels'])}
            df_with_features['predicted_label'] = [label_map.get(i, 'other') for i in pred_indices]
            df_with_features = apply_rules_to_predictions(df_with_features)

            # ⭐ GỌI HÀM LƯU CSV VỚI TÊN FILE MỚI
            save_results_to_csv(df_with_features, unique_name)
            
            # B5: ⭐ XỬ LÝ KẾT QUẢ CUỐI CÙNG (PHẦN BỊ THIẾU ĐÃ ĐƯỢC THÊM LẠI)
            # ====================================================================

            image_to_draw = cv2.cvtColor(image_obj, cv2.COLOR_BGR2RGB)
            extracted_info = defaultdict(list)
            
            # Vòng lặp để trích xuất text và vẽ lên ảnh
            for _, row in df_with_features.iterrows():
                if row['predicted_label'] != 'other':
                    key = row['predicted_label'].upper()
                    extracted_info[key].append(str(row['Object']))
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    cv2.rectangle(image_to_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image_to_draw, key, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Định dạng lại text
            formatted_info = {key: ' '.join(value) if key in ['ADDRESS', 'COMPANY'] else value for key, value in extracted_info.items()}
            
                        
            # ⭐ LƯU ẢNH KẾT QUẢ VỚI TÊN FILE MỚI
            result_filename = f"{unique_name}_annotated.jpg"
            result_folder = os.path.join(settings.MEDIA_ROOT, 'results')
            os.makedirs(result_folder, exist_ok=True)
            cv2.imwrite(os.path.join(result_folder, result_filename), cv2.cvtColor(image_to_draw, cv2.COLOR_RGB2BGR))
            result_image_url = os.path.join(settings.MEDIA_URL, 'results', result_filename).replace("\\", "/")

            # B6: Chuẩn bị dữ liệu để hiển thị
            results_for_template = [{'key': k, 'value': v, 'is_list': isinstance(v, list)} for k, v in formatted_info.items()]
            context['result_data'] = {
                'extracted_text': results_for_template,
                'annotated_image_url': result_image_url,
            }
            messages.success(request, "Đã trích xuất và dự đoán thành công!")

        except Exception as e:
            traceback.print_exc()
            messages.error(request, f"Lỗi trong quá trình xử lý: {e}")
            
    return render(request, 'real.html', context)