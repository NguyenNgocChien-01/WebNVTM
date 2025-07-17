from src.data_processing import Grapher # Import lớp đã sửa
import os

if __name__ == "__main__":
    # 1. Định nghĩa đường dẫn đến thư mục gốc chứa dữ liệu của bạn
    DATA_FOLDER = "data/raw" # Giả sử đang chạy từ thư mục gốc dự án

    # 2. Chọn một file ID để kiểm tra
    file_id = '000' # Ví dụ với file 000.jpg, 000.csv

    print(f"Bắt đầu kiểm tra lớp Grapher với file: {file_id}")

    try:
        # 3. Khởi tạo Grapher theo cách dùng cho training
        connect = Grapher(filename=file_id, data_fd=DATA_FOLDER)

        # 4. Chạy các phương thức
        print("Đang tạo cấu trúc đồ thị...")
        # use_clean_df=False là mặc định, dùng cho file CSV thô
        G, result, df_graph = connect.graph_formation(export_graph=False)

        print("Đang tính toán đặc trưng text...")
        df_text_features = connect.get_text_features(df_graph)

        print("Đang tính toán đặc trưng khoảng cách...")
        df_final = connect.relative_distance()

        print("\n✅ Hoàn thành kiểm tra!")
        print("5 dòng đầu tiên của DataFrame cuối cùng:")
        print(df_final.head())

    except Exception as e:
        print(f"\n❌ ĐÃ XẢY RA LỖI: {e}")