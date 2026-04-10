
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
import os
import time
import datetime
from src.config import CONFIG
from src.model import InvoiceGCN
from src import logging_utils




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
        'accuracy': final_accuracy,
        'classification_report': final_report
    }
    # Ghi log bằng logging_utils
    logging_utils.log_train_results(CONFIG, metrics_to_log)
    
    # Lưu model tốt nhất
    save_path = config['model_save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if best_model_state:
        torch.save(best_model_state, save_path)
        print(f"Đã lưu model tốt nhất tại: {save_path}")
