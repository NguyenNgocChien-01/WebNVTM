import logging_utils

if __name__ == "__main__":
    # Giả lập kết quả train
    config = {"labels": ["label1", "label2"]}
    metrics = {
        "accuracy": 0.95,
        "classification_report": {
            "macro avg": {"precision": 0.96, "recall": 0.95, "f1-score": 0.95},
            "label1": {"precision": 0.97, "recall": 0.96, "f1-score": 0.96},
            "label2": {"precision": 0.95, "recall": 0.94, "f1-score": 0.94},
            "accuracy": 0.95
        }
    }
    logging_utils.log_train_results(config, metrics)

    # Giả lập kết quả predict
    report = {
        "macro avg": {"precision": 0.93, "recall": 0.92, "f1-score": 0.92},
        "label1": {"precision": 0.94, "recall": 0.91, "f1-score": 0.92},
        "label2": {"precision": 0.92, "recall": 0.93, "f1-score": 0.92},
        "accuracy": 0.92
    }
    logging_utils.log_predict_results(config, report)
