import os
import time
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
import traceback 

def trangchu(request):
    return render(request,"index.html")

def run_real(request):
    return render(request,"real.html")

def run_test(request):
    return render(request,"test.html")

# Tải Model EasyOCR (1 lần duy nhất)
# ['vi', 'en'] để nhận dạng cả tiếng Việt và tiếng Anh
print("--- Đang tải mô hình EasyOCR. Quá trình này có thể mất vài phút... ---")
OCR_READER = easyocr.Reader(['vi', 'en'], gpu=torch.cuda.is_available())
print("✅ EasyOCR đã sẵn sàng!")


# --- 2. TẢI MODEL VÀ DỮ LIỆU CHUNG (Tải 1 lần khi server khởi động) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
TEST_DATA = None

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
# # --- 2. CÁC HÀM HỖ TRỢ ---
# def run_ocr_with_coords(image_path):
#     """Chạy OCR và trả về DataFrame sạch với 5 cột."""
#     image = cv2.imread(image_path)
#     if image is None: raise FileNotFoundError(f"OpenCV không thể đọc ảnh: {image_path}")
#     ocr_data = pytesseract.image_to_data(image, lang='vie+eng', output_type=pytesseract.Output.DATAFRAME)
#     ocr_data = ocr_data[ocr_data.conf > 30]
#     ocr_data.dropna(subset=['text'], inplace=True)
#     ocr_data['text'] = ocr_data['text'].str.strip()
#     ocr_data = ocr_data[ocr_data.text != '']
#     ocr_data['xmax'] = ocr_data['left'] + ocr_data['width']
#     ocr_data['ymax'] = ocr_data['top'] + ocr_data['height']
#     ocr_data.rename(columns={'left': 'xmin', 'top': 'ymin', 'text': 'Object'}, inplace=True)
#     return ocr_data[['xmin', 'ymin', 'xmax', 'ymax', 'Object']]

# def create_temp_csv_for_grapher(ocr_df, file_id):
#     """
#     Hàm "chuyển đổi": Nhận DataFrame sạch, chuyển thành định dạng 8 tọa độ
#     và lưu thành file CSV tạm thời mà Grapher cũ có thể đọc.
#     """
#     formatted_rows = []
#     for _, row in ocr_df.iterrows():
#         xmin, ymin, xmax, ymax, text = row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['Object']
#         if ',' in str(text): text = f'"{text}"'
#         # Tạo chuỗi 8 tọa độ + text
#         formatted_string = f"{xmin},{ymin},{xmax},{ymin},{xmax},{ymax},{xmin},{ymax},{text}"
#         formatted_rows.append(formatted_string)
    
#     # Tạo DataFrame với một cột duy nhất
#     df_for_csv = pd.DataFrame(formatted_rows)
    
#     # Lưu file CSV tạm thời vào đúng thư mục mà Grapher sẽ tìm
#     temp_box_dir = os.path.join(settings.BASE_DIR, 'data/raw/box')
#     os.makedirs(temp_box_dir, exist_ok=True)
#     csv_path = os.path.join(temp_box_dir, f"{file_id}.csv")
#     df_for_csv.to_csv(csv_path, index=False, header=False, encoding='utf-8')
#     print(f"--- [DEBUG] Đã tạo file CSV tạm tại: {csv_path} ---")


# # --- 3. VIEW CHÍNH ---
# def ocr_and_predict_view(request):
#     context = {}
#     if request.method == 'POST':
#         image_file = request.FILES.get("invoice_image")
#         if not (image_file and MODEL):
#             messages.error(request, "Lỗi: Chưa chọn file hoặc model chưa được tải.")
#             return render(request, 'real.html', context)
        
#         fs = FileSystemStorage()
#         file_id = str(uuid.uuid4())
#         filename = fs.save(f"uploads/{file_id}.jpg", image_file)
#         uploaded_file_path = fs.path(filename)
#         context['original_image_url'] = fs.url(filename)

#         try:
#             # B1: Chạy OCR
#             ocr_dataframe = run_ocr_with_coords(uploaded_file_path)
#             if ocr_dataframe.empty: raise ValueError("OCR không nhận diện được văn bản.")

#             # B2: Tạo file CSV tạm thời với định dạng cũ
#             create_temp_csv_for_grapher(ocr_dataframe, file_id)

#             # B3: Khởi tạo Grapher gốc. Nó sẽ tự tìm và đọc file CSV vừa tạo
#             grapher = Grapher(
#                 filename=file_id, 
#                 data_fd=os.path.join(settings.BASE_DIR, 'data/raw'),
#                 image_path=uploaded_file_path
#             )
            
#             # B4: Chạy các bước xử lý của Grapher
#             G, _, df_processed = grapher.graph_formation()
#             df_features = grapher.relative_distance()
            
#             # B5: Tạo Data Object và dự đoán
#             adj = torch.tensor(list(G.edges)).t().contiguous()
#             feature_cols = ['rd_r', 'rd_b', 'rd_l', 'rd_t', 'n_upper', 'n_alpha', 'n_spaces', 'n_numeric', 'n_special']
#             x = torch.tensor(df_features[feature_cols].values, dtype=torch.float)
#             single_data = Data(x=x, edge_index=adj, img_id=file_id)

#             with torch.no_grad():
#                 out = MODEL(single_data.to(DEVICE))
#                 pred_indices = out.max(dim=1)[1].cpu().numpy()
            
#             # B6: Xử lý kết quả và tạo ảnh chú thích
#             # ==================================================
#             label_map = {i: label for i, label in enumerate(CONFIG['labels'])}
#             df_features['predicted_label'] = [label_map.get(i) for i in pred_indices]
            
#             image_to_draw = cv2.cvtColor(cv2.imread(uploaded_file_path), cv2.COLOR_BGR2RGB)
#             extracted_info = defaultdict(list)
#             for _, row in df_features.iterrows():
#                 if row['predicted_label'] != 'other':
#                     key = row['predicted_label'].upper()
#                     extracted_info[key].append(str(row['Object']))
#                     x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#                     cv2.rectangle(image_to_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     cv2.putText(image_to_draw, key, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
#             # Định dạng và lưu ảnh kết quả
#             formatted_info = {key: ' '.join(value) if key in ['ADDRESS', 'COMPANY'] else value for key, value in extracted_info.items()}
#             result_filename = f"{file_id}_annotated.jpg"
#             result_folder = os.path.join(settings.MEDIA_ROOT, 'results')
#             os.makedirs(result_folder, exist_ok=True)
#             cv2.imwrite(os.path.join(result_folder, result_filename), cv2.cvtColor(image_to_draw, cv2.COLOR_RGB2BGR))
#             result_image_url = os.path.join(settings.MEDIA_URL, 'results', result_filename).replace("\\", "/")

#             # B7: Chuẩn bị context để hiển thị
#             # ==================================================
#             results_for_template = []
#             for key, value in formatted_info.items():
#                 results_for_template.append({
#                     'key': key,
#                     'value': value,
#                     'is_list': isinstance(value, list)
#                 })

#             context['result_data'] = {
#                 'extracted_text': results_for_template,
#                 'annotated_image_url': result_image_url,
#             }
#             messages.success(request, "Đã trích xuất và dự đoán thành công!")


#         except Exception as e:
#             traceback.print_exc()
#             messages.error(request, f"Lỗi trong quá trình xử lý: {e}")
            
#     return render(request, 'real.html', context)


# --- 2. HÀM HỖ TRỢ OCR (Phiên bản EasyOCR) ---
# --- 3. CÁC HÀM HỖ TRỢ ---
def run_easyocr_with_coords(image_path):
    """Chạy EasyOCR để lấy chữ và tọa độ, trả về DataFrame."""
    results = OCR_READER.readtext(image_path)
    ocr_list = []
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        xmin, ymin = int(tl[0]), int(tl[1])
        xmax, ymax = int(br[0]), int(br[1])
        ocr_list.append([xmin, ymin, xmax, ymax, text])
    return pd.DataFrame(ocr_list, columns=['xmin', 'ymin', 'xmax', 'ymax', 'Object'])

def create_data_object(ocr_dataframe, image_object, file_id):
    """Sử dụng Grapher để chuyển dữ liệu OCR thành Data Object cho model GCN."""
    # Khởi tạo Grapher "thông minh" bằng cách truyền trực tiếp dữ liệu
    grapher = Grapher(ocr_dataframe=ocr_dataframe, image_object=image_object, filename=file_id)
    
    # Chạy các bước xử lý của Grapher
    G, _, df_processed = grapher.graph_formation(use_clean_df=True)
    df_features = grapher.relative_distance()
    df_features = grapher.get_text_features(df_features)

    
    # Tạo Data Object
    adj = torch.tensor(list(G.edges)).t().contiguous()
    feature_cols = ['rd_r', 'rd_b', 'rd_l', 'rd_t', 'n_upper', 'n_alpha', 'n_spaces', 'n_numeric', 'n_special']
    x = torch.tensor(df_features[feature_cols].values, dtype=torch.float)
    
    return Data(x=x, edge_index=adj, img_id=file_id), df_features

# --- 4. VIEW CHÍNH ---
def ocr_and_predict_view(request):
    context = {}
    if request.method == 'POST':
        image_file = request.FILES.get("invoice_image")
        if not (image_file and MODEL):
            messages.error(request, "Lỗi: Chưa chọn file hoặc model AI chưa được tải.")
            return render(request, 'real.html', context)
        
        # Lưu ảnh tải lên
        fs = FileSystemStorage()
        file_id = str(uuid.uuid4())
        filename = fs.save(f"uploads/{file_id}.jpg", image_file)
        uploaded_file_path = fs.path(filename)
        context['original_image_url'] = fs.url(filename)

        try:
            # B1: Chạy EasyOCR
            ocr_df = run_easyocr_with_coords(uploaded_file_path)
            if ocr_df.empty:
                raise ValueError("OCR không nhận diện được văn bản.")
            
            # B2: Tạo Data Object từ dữ liệu OCR
            image_obj = cv2.imread(uploaded_file_path)
            single_data, df_with_features = create_data_object(ocr_df, image_obj, file_id)

            # B3: Chạy dự đoán bằng Model GCN
            with torch.no_grad():
                out = MODEL(single_data.to(DEVICE))
                pred_indices = out.max(dim=1)[1].cpu().numpy()
            
            # B4: Xử lý kết quả và tạo ảnh chú thích
            label_map = {i: label for i, label in enumerate(CONFIG['labels'])}
            df_with_features['predicted_label'] = [label_map.get(i, 'other') for i in pred_indices]
            
            image_to_draw = cv2.cvtColor(image_obj, cv2.COLOR_BGR2RGB)
            extracted_info = defaultdict(list)
            for _, row in df_with_features.iterrows():
                if row['predicted_label'] != 'other':
                    key = row['predicted_label'].upper()
                    extracted_info[key].append(str(row['Object']))
                    x1,y1,x2,y2 = int(row['xmin']),int(row['ymin']),int(row['xmax']),int(row['ymax'])
                    cv2.rectangle(image_to_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image_to_draw, key, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            formatted_info = {key: ' '.join(v) if key in ['ADDRESS', 'COMPANY'] else v for key, v in extracted_info.items()}
            result_filename = f"{file_id}_annotated.jpg"
            result_folder = os.path.join(settings.MEDIA_ROOT, 'results')
            os.makedirs(result_folder, exist_ok=True)
            cv2.imwrite(os.path.join(result_folder, result_filename), cv2.cvtColor(image_to_draw, cv2.COLOR_RGB2BGR))
            
            # B5: Chuẩn bị dữ liệu để hiển thị
            results_for_template = [{'key': k, 'value': v, 'is_list': isinstance(v, list)} for k, v in formatted_info.items()]
            context['result_data'] = {
                'extracted_text': results_for_template,
                'annotated_image_url': os.path.join(settings.MEDIA_URL, 'results', result_filename).replace("\\", "/"),
            }
            messages.success(request, "Đã trích xuất và dự đoán thành công!")

        except Exception as e:
            traceback.print_exc()
            messages.error(request, f"Lỗi trong quá trình xử lý: {e}")
            
    return render(request, 'real.html', context)