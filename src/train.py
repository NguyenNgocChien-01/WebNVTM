import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
import os
import time
import datetime
import csv
from src.config import CONFIG
import json
import psutil

from src.model import InvoiceGCN

# src/train.py
import json
import datetime
import os

def log_training_results(config, metrics, execution_time, dataset_stats):
    """
    Ghi toàn bộ thông tin vào một file JSON phẳng, không phân mục.
    """
    log_dir = "outputs/logs/training_sessions"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"session_{timestamp}_acc_{metrics.get('accuracy', 0):.4f}.json"
    log_path = os.path.join(log_dir, filename)

    # --- PHẦN THAY ĐỔI ---
    # Bắt đầu với một dictionary rỗng
    log_data = {}

    # Gộp dictionary siêu tham số vào
    log_data.update(config['model_params'])
    
    # Gộp dictionary thống kê dữ liệu
    log_data.update(dataset_stats)

    # Gộp dictionary kết quả cuối cùng
    final_results = {
        "epoch_stopped_at": metrics.get('epoch_stopped_at'),
        "best_validation_loss": metrics.get('best_val_loss'),
        "final_test_accuracy": metrics.get('accuracy'),
        "execution_time_seconds": round(execution_time, 2),
        "classification_report": metrics.get('classification_report'),
        "normalized_confusion_matrix_text": str(metrics.get('normalized_cm').tolist()) # Lưu ma trận dưới dạng text
    }
    log_data.update(final_results)
    

    # --- KẾT THÚC PHẦN THAY ĐỔI ---

    # Ghi vào file JSON với định dạng đẹp
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Đã lưu log huấn luyện chi tiết (dạng phẳng) tại: {log_path}")


def run_training_session(config, train_data, test_data, device):
    """
    Phiên bản huấn luyện nâng cao: theo dõi, dừng sớm và ghi log chi tiết.
    """
    start_time = time.time()
    
    # --- 1. KHỞI TẠO ---
    model_params = config['model_params']
    model = InvoiceGCN(
        input_dim=train_data.x.shape[1],
        hidden_dims=model_params['hidden_dims'],
        n_classes=model_params['n_classes'],
        dropout_rate=model_params['dropout_rate'],
        chebnet=model_params.get('chebnet', True),
        K=model_params.get('K', 3)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_params['lr'], weight_decay=model_params['weight_decay'])
    train_data, test_data = train_data.to(device), test_data.to(device)

    # --- 2. XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU ---
    y_labels = train_data.y.cpu().numpy()
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_labels), y=y_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # --- 3. VÒNG LẶP HUẤN LUYỆN VÀ THEO DÕI ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 200  # Dừng nếu val_loss không cải thiện sau 200*2 = 400 epochs
    best_model_state = None

    num_epochs = model_params['num_epochs']
    print(f"\nBắt đầu quá trình huấn luyện với {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(train_data)
        y_true_train = train_data.y - 1
        loss = F.nll_loss(out, y_true_train, weight=class_weights_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                val_out = model(test_data)
                y_true_test = test_data.y - 1
                val_loss = F.nll_loss(val_out, y_true_test)
                
                y_pred = val_out.max(dim=1)[1]
                acc = y_pred.eq(y_true_test).sum().item() / test_data.num_nodes
                
                print(f"Epoch: {epoch:04d}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {acc:.4f}")

                # Logic Dừng sớm (Early Stopping)
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    epochs_no_improve = 0
                    best_model_state = model.state_dict().copy()
                else:
                    epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"\nValidation loss không cải thiện sau {patience*200} epochs. Dừng sớm tại epoch {epoch}.")
                break
    
    # --- 4. ĐÁNH GIÁ CUỐI CÙNG VÀ LƯU TRỮ ---
    print("\nHoàn tất huấn luyện! Đang đánh giá model tốt nhất...")
    # Tải lại model có val_loss tốt nhất
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        final_out = model(test_data)
        y_true_final = (test_data.y - 1).cpu().numpy()
        y_pred_final = final_out.max(dim=1)[1].cpu().numpy()

        target_names = CONFIG['labels']
        label_indices = list(range(len(target_names)))
        final_report = classification_report(y_true_final, y_pred_final, labels=label_indices, target_names=target_names, zero_division=0, output_dict=True)
        final_accuracy = final_report['accuracy']
        
        print("\nBáo cáo chi tiết cuối cùng trên tập Test:")
        print(classification_report(y_true_final, y_pred_final, labels=label_indices, target_names=target_names, zero_division=0))

    # --- 5. GHI LOG VÀ LƯU MODEL ---
    execution_time = time.time() - start_time
    
    metrics_to_log = {
        'epoch_stopped_at': epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': loss.item(),
        'final_val_loss': F.nll_loss(final_out, (test_data.y - 1)).item(),
        'final_accuracy': final_accuracy,
        'classification_report': final_report
    }
    
    # Gọi hàm ghi log
    log_training_results(model_params, metrics_to_log, execution_time)
    
    # Lưu model tốt nhất
    save_path = config['model_save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if best_model_state:
        torch.save(best_model_state, save_path)
        print(f"Đã lưu model tốt nhất tại: {save_path}")