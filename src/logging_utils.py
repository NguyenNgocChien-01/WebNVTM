
import os
import csv
from datetime import datetime

def ensure_log_dir(log_dir=None):
    """Đảm bảo thư mục log tồn tại."""
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def log_to_csv(log_name, rows, fieldnames, log_dir=None):
    """Ghi dữ liệu vào file CSV trong thư mục logs."""
    log_dir = ensure_log_dir(log_dir)
    log_path = os.path.join(log_dir, log_name)
    with open(log_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Đã lưu log tại: {log_path}")

def log_extraction_results(img_id, extracted_info, config, stats, file_path, status="Success", error_message=None):
    """
    Ghi log kết quả trích xuất thông tin từ hóa đơn vào file CSV
    
    Parameters:
    -----------
    img_id: str
        ID của ảnh hóa đơn
    extracted_info: dict
        Thông tin đã trích xuất được từ hóa đơn
    config: dict 
        Cấu hình của model
    stats: dict
        Các thông số về quá trình xử lý
    file_path: str
        Đường dẫn file ảnh gốc
    status: str
        Trạng thái xử lý ("Success" hoặc "Error")
    error_message: str
        Thông báo lỗi nếu có
    """
    
    # Tạo tên file log với timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"extraction_log_{timestamp}.csv"
    log_dir = ensure_log_dir()
    log_path = os.path.join(log_dir, log_filename)
    
    # Chuẩn bị dữ liệu cho log
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "image_id": img_id,
        "image_path": file_path,
        "status": status,
        "error_message": error_message if error_message else "",
        
        # Thông số kỹ thuật của model
        "model_type": config.get("model_type", "GCN"),
        "hidden_dims": str(config.get("hidden_dims", [])),
        "dropout_rate": config.get("dropout_rate", 0.5),
        "learning_rate": config.get("learning_rate", 0.001),
        "activation": config.get("activation", "relu"),
        "optimizer": config.get("optimizer", "adam"),
        
        # Thông số xử lý
        "processing_time": stats.get("processing_time", 0),
        "total_text_detected": stats.get("total_text_detected", 0),
        "total_labels_predicted": stats.get("total_labels_predicted", 0),
        "text_detection_ratio": stats.get("total_text_detected", 0) / stats.get("total_labels_predicted", 1) if stats.get("total_labels_predicted", 0) > 0 else 0,
        
        # Metrics về độ tin cậy của model
        "model_confidence": stats.get("model_confidence", 0),
        "validation_loss": stats.get("val_loss", 0),
        "mean_confidence": stats.get("mean_confidence", 0),
        "min_confidence": stats.get("min_confidence", 0),
        "max_confidence": stats.get("max_confidence", 0),
        "std_confidence": stats.get("std_confidence", 0),
        "confidence_per_label": str(stats.get("confidence_per_label", {})),
        
        # Thống kê về labels
        "label_distribution": str(stats.get("label_distribution", {})),
        "other_ratio": stats.get("other_ratio", 0),
        "successful_extraction_ratio": stats.get("successful_extraction_ratio", 0),
        
        # Kết quả trích xuất
        "company_name": " ".join(extracted_info.get("COMPANY", [])),
        "company_confidence": stats.get("confidence_per_label", {}).get("COMPANY", 0),
        "tax_id": " ".join(extracted_info.get("TAX_ID", [])),
        "tax_id_confidence": stats.get("confidence_per_label", {}).get("TAX_ID", 0),
        "address": " ".join(extracted_info.get("ADDRESS", [])),
        "address_confidence": stats.get("confidence_per_label", {}).get("ADDRESS", 0),
        "date": " ".join(extracted_info.get("DATE", [])),
        "date_confidence": stats.get("confidence_per_label", {}).get("DATE", 0),
        "total_amount": " ".join(extracted_info.get("TOTAL", [])),
        "total_confidence": stats.get("confidence_per_label", {}).get("TOTAL", 0),
        "items": "|".join(extracted_info.get("ITEMS", [])),
        "items_confidence": stats.get("confidence_per_label", {}).get("ITEMS", 0),
    }
    
    # Kiểm tra xem file đã tồn tại chưa
    file_exists = os.path.exists(log_path)
    
    # Ghi vào file CSV
    with open(log_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    
    return log_path

def make_log_row(
    phase,
    label=None,
    precision=None,
    recall=None,
    f1=None,
    acc=None,
    auc=None,
    loss=None,
    mmc=None,
    tn=None,
    fp=None,
    fn=None,
    tp=None,
    status="Success",
    error_message=None,
    time_run=None,
    extra=None
):
    """Tạo 1 dòng log chuẩn cho train hoặc predict."""
    row = {
        "Time": time_run or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Phase": phase,
        "Label": label or "",
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": acc,
        "AUC": auc,
        "Loss": loss,
        "MMC": mmc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Status": status,
        "Error": error_message or ""
    }
    if extra:
        row.update(extra)
    return row

def log_train_results(
    config,
    metrics,
    dataset_stats=None,
    log_dir=None
):
    """Ghi log kết quả train vào logs/ dưới dạng CSV."""
    time_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_name = f"train_{time_run.replace(' ', '_').replace(':', '')}.csv"
    fieldnames = [
        "Time", "Phase", "Label", "Precision", "Recall", "F1-Score", "Accuracy", "AUC", "Loss", "MMC", "TN", "FP", "FN", "TP", "Status", "Error"
    ]
    rows = []
    # Macro avg
    macro = metrics.get('classification_report', {}).get('macro avg', {})
    rows.append(make_log_row(
        phase="train",
        label="Macro Avg",
        precision=macro.get('precision'),
        recall=macro.get('recall'),
        f1=macro.get('f1-score'),
        acc=metrics.get('accuracy'),
        auc=metrics.get('auc'),
        loss=metrics.get('loss'),
        mmc=metrics.get('mmc'),
        time_run=time_run
    ))
    # Từng nhãn
    labels = metrics.get('classification_report', {})
    for label in config.get('labels', []):
        if label in labels:
            l = labels[label]
            rows.append(make_log_row(
                phase="train",
                label=label,
                precision=l.get('precision'),
                recall=l.get('recall'),
                f1=l.get('f1-score'),
                acc=None,
                auc=None,
                loss=None,
                mmc=None,
                time_run=time_run
            ))
    log_to_csv(log_name, rows, fieldnames, log_dir)

def log_predict_results(
    config,
    report,
    cm=None,
    status="Success",
    error_message=None,
    log_dir=None
):
    """Ghi log kết quả predict vào logs/ dưới dạng CSV."""
    time_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_name = f"predict_{time_run.replace(' ', '_').replace(':', '')}.csv"
    fieldnames = [
        "Time", "Phase", "Label", "Precision", "Recall", "F1-Score", "Accuracy", "AUC", "Loss", "MMC", "TN", "FP", "FN", "TP", "Status", "Error"
    ]
    rows = []
    # Macro avg
    macro = report.get('macro avg', {})
    rows.append(make_log_row(
        phase="predict",
        label="Macro Avg",
        precision=macro.get('precision'),
        recall=macro.get('recall'),
        f1=macro.get('f1-score'),
        acc=report.get('accuracy'),
        auc=None,
        loss=None,
        mmc=None,
        time_run=time_run,
        status=status,
        error_message=error_message
    ))
    # Từng nhãn
    for label in config.get('labels', []):
        if label in report:
            l = report[label]
            rows.append(make_log_row(
                phase="predict",
                label=label,
                precision=l.get('precision'),
                recall=l.get('recall'),
                f1=l.get('f1-score'),
                acc=None,
                auc=None,
                loss=None,
                mmc=None,
                time_run=time_run,
                status=status,
                error_message=error_message
            ))
    log_to_csv(log_name, rows, fieldnames, log_dir)
