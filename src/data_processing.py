import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import math
import itertools
import networkx as nx
import torch



import os
import cv2
import pandas as pd
import numpy as np
import networkx as nx
import itertools

class Grapher:
    def __init__(self, filename=None, data_fd=None, ocr_dataframe=None, image_object=None):
        """
        Hàm khởi tạo "thông minh":
        - Nếu nhận 'filename' và 'data_fd', nó sẽ hoạt động như cũ (đọc file cho training).
        - Nếu nhận 'ocr_dataframe' và 'image_object', nó sẽ dùng trực tiếp (cho dự đoán real-time).
        """
        self.filename = filename if filename else 'realtime_file'
        
        # KỊCH BẢN 1: DÀNH CHO TRAINING (TỰ ĐỌC FILE)
        if filename and data_fd:
            self.data_fd = data_fd
            box_path = os.path.join(data_fd, "box", filename + '.csv')
            labels_path = os.path.join(data_fd, "labels", filename + '.csv')
            image_path = os.path.join(data_fd, "img", filename + '.jpg')

            try:
                with open(box_path, 'r', encoding='utf-8') as f:
                    self.df = pd.DataFrame(f.readlines())
            except FileNotFoundError:
                raise FileNotFoundError(f"LỖI (Training): Không tìm thấy file box OCR tại: {box_path}")

            self.image = cv2.imread(image_path)
            if self.image is None:
                raise FileNotFoundError(f"LỖI (Training): Không đọc được ảnh tại: {image_path}")
            
            try:
                self.df_withlabels = pd.read_csv(labels_path, header=None)
            except FileNotFoundError:
                self.df_withlabels = pd.DataFrame()

        # KỊCH BẢN 2: DÀNH CHO DỰ ĐOÁN ẢNH MỚI (NHẬN TRỰC TIẾP DỮ LIỆU)
        elif ocr_dataframe is not None and image_object is not None:
            self.df = ocr_dataframe
            self.image = image_object
            self.df_withlabels = pd.DataFrame()
        
        else:
            raise ValueError("Grapher cần được cung cấp (filename, data_fd) hoặc (ocr_dataframe, image_object)")

    def graph_formation(self, use_clean_df=False, export_graph=False):
        """
        Hàm này đã được nâng cấp để xử lý cả DataFrame sạch và CSV thô.
        """
        image = self.image

        # ⭐ LOGIC PHÂN LUỒNG XỬ LÝ DỮ LIỆU
        if use_clean_df:
            df = self.df.copy()
            for col in ['xmin', 'ymin', 'xmax', 'ymax']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax'], inplace=True)
            df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
        else:
            df = self.df[0].str.split(',', expand=True)
            temp = df.copy()
            temp.fillna('', inplace=True)
            if 8 in temp.columns and 9 in temp.columns:
                 temp[8] = temp[8].astype(str).str.cat(temp.iloc[:, 9:].astype(str), sep=", ")
            temp = temp.loc[:, :8]
            temp.columns = list(range(8)) + ['Object']
            temp_coords = temp.iloc[:,:8].apply(pd.to_numeric, errors='coerce').dropna()
            df = pd.concat([temp_coords, temp.loc[temp_coords.index, 'Object']], axis=1)

            df.columns = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'Object']
            df['xmin'] = df[['x1', 'x2', 'x3', 'x4']].min(axis=1)
            df['ymin'] = df[['y1', 'y2', 'y3', 'y4']].min(axis=1)
            df['xmax'] = df[['x1', 'x2', 'x3', 'x4']].max(axis=1)
            df['ymax'] = df[['y1', 'y2', 'y3', 'y4']].max(axis=1)
            df = df[['xmin', 'ymin', 'xmax', 'ymax', 'Object']]

        # --- PHẦN LOGIC CHUNG (Tạo đồ thị) ---
        df.sort_values(by=['ymin'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["ymax"] = df["ymax"].apply(lambda x: x - 1)

        master = []
        for idx, row in df.iterrows():
            flat_master = list(itertools.chain(*master))
            if idx not in flat_master:
                top_a, bottom_a = row['ymin'], row['ymax']
                line = [idx]
                for idx_2, row_2 in df.iterrows():
                    if idx_2 not in flat_master and idx != idx_2:
                        top_b, bottom_b = row_2['ymin'], row_2['ymax']
                        if (top_a <= bottom_b) and (bottom_a >= top_b):
                            line.append(idx_2)
                master.append(line)

        df2 = pd.DataFrame({'words_indices': master, 'line_number': range(1, len(master) + 1)})
        df2 = df2.set_index('line_number').words_indices.apply(pd.Series).stack().reset_index(level=0).rename(columns={0: 'words_indices'})
        df2['words_indices'] = df2['words_indices'].astype('int')
        final = df.merge(df2, left_on=df.index, right_on='words_indices').drop('words_indices', axis=1)
        df = final.sort_values(by=['line_number', 'xmin']).reset_index(drop=True)
        
        df.reset_index(inplace=True)
        grouped = df.groupby('line_number')
        horizontal_connections, bottom_connections = {}, {}
        
        for _, group in grouped:
            a = group['index'].tolist()
            for i in range(len(a) - 1):
                df.loc[df['index'] == a[i], 'right'] = int(a[i + 1])
                df.loc[df['index'] == a[i + 1], 'left'] = int(a[i])
                horizontal_connections[a[i]] = a[i+1]
        
        for idx, row in df.iterrows():
            if idx not in bottom_connections:
                right_a, left_a = row['xmax'], row['xmin']
                for idx_2, row_2 in df.iterrows():
                    if idx < idx_2 and idx_2 not in bottom_connections.values():
                        right_b, left_b = row_2['xmax'], row_2['xmin']
                        if (left_b <= right_a) and (right_b >= left_a):
                            bottom_connections[idx] = idx_2
                            df.loc[df['index'] == idx, 'bottom'] = idx_2
                            df.loc[df['index'] == idx_2, 'top'] = idx
                            break
        
        result = {}
        for d in (horizontal_connections, bottom_connections):
            for k, v in d.items(): result.setdefault(k, []).append(v)
        
        G = nx.from_dict_of_lists(result)
        
        if not self.df_withlabels.empty:
            last_col_index = self.df_withlabels.shape[1] - 1
            if last_col_index in self.df_withlabels.columns:
                df['labels'] = self.df_withlabels[last_col_index]
        else:
            df['labels'] = 'other'
        
        self.df = df
        return G, result, df

    def get_text_features(self, df):
        data = df['Object'].tolist()
        special_chars = list('&@#()-+*/=%._,\'\\|":')
        
        features = []
        for words in data:
            words = str(words) if pd.notna(words) else ''
            features.append([
                sum(1 for char in words if char.isupper()),
                sum(1 for char in words if char.isalpha()),
                sum(1 for char in words if char.isspace()),
                sum(1 for char in words if char.isnumeric()),
                sum(1 for char in words if char in special_chars)
            ])
            
        df[['n_upper', 'n_alpha', 'n_spaces', 'n_numeric', 'n_special']] = features
        return df
    
    def relative_distance(self):
        df = self.df.copy()
        if self.image is None: return df
        image_height, image_width = self.image.shape[:2]

        if 'index' not in df.columns: df.reset_index(inplace=True)

        for col in ['rd_r', 'rd_b', 'rd_l', 'rd_t']:
            if col not in df.columns: df[col] = 0.0

        for index, row in df.iterrows():
            if 'right' in df.columns and pd.notna(row['right']):
                right_neighbor = df.loc[df['index'] == int(row['right'])]
                if not right_neighbor.empty:
                    df.loc[index, 'rd_r'] = (right_neighbor['xmin'].values[0] - row['xmax']) / image_width
            if 'left' in df.columns and pd.notna(row['left']):
                left_neighbor = df.loc[df['index'] == int(row['left'])]
                if not left_neighbor.empty:
                    df.loc[index, 'rd_l'] = (row['xmin'] - left_neighbor['xmax'].values[0]) / image_width
            if 'bottom' in df.columns and pd.notna(row['bottom']):
                bottom_neighbor = df.loc[df['index'] == int(row['bottom'])]
                if not bottom_neighbor.empty:
                    df.loc[index, 'rd_b'] = (bottom_neighbor['ymin'].values[0] - row['ymax']) / image_height
            if 'top' in df.columns and pd.notna(row['top']):
                top_neighbor = df.loc[df['index'] == int(row['top'])]
                if not top_neighbor.empty:
                    df.loc[index, 'rd_t'] = (row['ymin'] - top_neighbor['ymax'].values[0]) / image_height

        self.df = df
        return df

    
    def get_df_for_visualization(self):
        """
        Phương thức này sẽ dùng self.file_path đã được khởi tạo ở trên.
        """
        # Bây giờ self.file_path đã tồn tại và có thể được sử dụng ở đây
        with open(os.path.join(self.data_fd, "box", self.filename + '.csv'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # ... (phần còn lại của hàm giữ nguyên) ...
        df = pd.DataFrame([line.strip().split(',') for line in lines])
        text_data = df.iloc[:, 8:].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        df = df.iloc[:, :4]
        df.columns = ['xmin', 'ymin', 'xmax', 'ymax']
        df['Object'] = text_data
        df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].apply(pd.to_numeric)
        df.sort_values(by='ymin', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


def load_data(config):
    """Tải dữ liệu train/test đã được xử lý."""
    # Lấy đường dẫn thư mục từ config
    processed_folder = config.get('processed_folder')
    if not processed_folder:
        print("Lỗi: Đường dẫn 'processed_folder' không được định nghĩa trong CONFIG.")
        return None, None
        
    print(f"Đang tải dữ liệu đã xử lý từ: {processed_folder}")
    try:
        train_path = os.path.join(processed_folder, 'train_data.dataset')
        test_path = os.path.join(processed_folder, 'test_data.dataset')
        
        # Tải dữ liệu
        train_data = torch.load(train_path, weights_only=False)
        test_data = torch.load(test_path, weights_only=False)
        
        print("Tải dữ liệu thành công!")
        return train_data, test_data
        
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dataset tại '{processed_folder}'.")
        print("Vui lòng đảm bảo bạn đã chạy pipeline xử lý dữ liệu và đường dẫn trong CONFIG là chính xác.")
        return None, None
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn khi tải dữ liệu: {e}")
        return None, None