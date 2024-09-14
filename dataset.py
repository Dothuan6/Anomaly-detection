#%%
import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import resample
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from PIL import Image
from torch.utils.data import random_split
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 2408
IMAGE_HEIGHT , IMAGE_WIDTH =  112, 112
SEQUENCE_LENGTH = 5  # Number of frames to extract from each video/segment
SECONDS = 60
MAX_SEGMENT = 15
# MAX_SEGMENT = 500
INTERVAL = 1
# Get the names of all classes in Train UCF - Crime Dataset.
DATASET_DIR = "dataset/Train"
TEST_DIR = "dataset/Test"
CLASSES_LIST = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Others",  "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

def frames_extraction(video_path, sequence_length=SEQUENCE_LENGTH, interval=INTERVAL, max_segments=MAX_SEGMENT):
    frames_list = []
    
    # Đọc video từ đường dẫn video_path
    video_reader = cv2.VideoCapture(video_path)

    # Số lượng frame của video (FPS)
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    # Tính số lượng frame của video
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = video_frames_count / fps
    if video_duration_seconds <= SECONDS - 30:
        interval = 1
    elif video_duration_seconds <= SECONDS + 60:
        interval = 2
    elif video_duration_seconds <= SECONDS + 180:
        interval = 3
    elif video_duration_seconds <= SECONDS + 300:
        interval = 4
    elif video_duration_seconds <= SECONDS + 420:
        interval = 5
    else:
        interval = 6
    # Tính số lượng frame cần bỏ qua để đạt được khoảng thời gian 5 giây
    # Ví dụ: Nếu video có 30 FPS thì cần bỏ qua 150 frame để đạt được 5 giây
    skip_frames_window = max(int(interval * fps), 1) # 5*30 = 150 frames
    segmented_frames = []
    # Tính toán số lượng frame cần trích xuất
    # window_size = min(SEQUENCE_LENGTH, video_frames_count // skip_frames_window) # đảm bảo không vượt quá số frame cần trích xuất S_Lenght
    # Duyệt qua các frame của video
    for frame_counter in range(0, video_frames_count, skip_frames_window):
        if len(segmented_frames) >= max_segments:
            break
        # Đặt vị trí frame hiện tại của video
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

        # Đọc frame từ video
        success, frame = video_reader.read() 

        # Kiểm tra nếu không đọc được frame thì dừng vòng lặp
        if not success:
            break

        # Thay đổi kích thước frame về độ cao và chiều rộng cố định
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Chuẩn hóa frame bằng cách chia mỗi giá trị pixel cho 255 để đưa về khoảng 0 và 1
        normalized_frame = resized_frame / 255
        # Thêm frame đã chuẩn hóa vào danh sách frames_list
        frames_list.append(normalized_frame.astype(np.float32))
        
        if len(frames_list) == sequence_length:
            segmented_frames.append(frames_list)
            frames_list = []
    # Giải phóng đối tượng VideoCapture
    video_reader.release()
    # Adjusting the number of frames to match SEQUENCE_LENGTH
    if len(frames_list) > 0 and len(segmented_frames) < max_segments:
        if len(frames_list) < sequence_length:
            padding = [np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)] * (sequence_length - len(frames_list))
            frames_list.extend(padding)
        segmented_frames.append(frames_list)
    print(f"Extracted {len(segmented_frames)} segments from {video_path}, each with {sequence_length} frames.")
    # Trả về danh sách các frame
    return segmented_frames

def create_dataset():
    '''
    Hàm này sẽ trích xuất các frame từ video và tạo ra tập dữ liệu cho việc huấn luyện mô hình
    Returns:
        features:          Danh sách frame của video
        labels:            Danh sách nhãn của video
        video_files_paths: Danh sách đường dẫn của video
    '''
    features = []
    labels = []
    video_files_paths = []
    
    # Duyệt qua các lớp trong tập dữ liệu
    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f'Nhãn: {class_name}')
        # Lấy danh sách các video trong thư mục của lớp
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name)) #Vandalism001_x264.mp4
        for file_name in files_list:
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            segmented_frames = frames_extraction(video_file_path)
            # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
            # So ignore the vides having frames less than the SEQUENCE_LENGTH.
            for frames in segmented_frames:
                # Append the data to their repective lists.
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    # Converting the list to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)  
    
    # Return the frames, class index, and video file path.
    return features, labels, video_files_paths
def create_test_dataset():
    '''
    Hàm này sẽ trích xuất các frame từ video và tạo ra tập dữ liệu cho việc huấn luyện mô hình
    Returns:
        features:          Danh sách frame của video
        labels:            Danh sách nhãn của video
        video_files_paths: Danh sách đường dẫn của video
    '''
    features = []
    labels = []
    video_files_paths = []
    
    # Duyệt qua các lớp trong tập dữ liệu
    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f'Nhãn: {class_name}')
        # Lấy danh sách các video trong thư mục của lớp
        files_list = os.listdir(os.path.join(TEST_DIR, class_name)) #Vandalism001_x264.mp4
        for file_name in files_list:
            video_file_path = os.path.join(TEST_DIR, class_name, file_name)
            segmented_frames = frames_extraction(video_file_path)

            # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
            # So ignore the vides having frames less than the SEQUENCE_LENGTH.
            for frames in segmented_frames:
                # Append the data to their repective lists.
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    # Converting the list to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)  
    
    # Return the frames, class index, and video file path.
    return features, labels, video_files_paths

class CrimeDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        print(f"Features length: {len(self.features)}")
        print(f"Labels length: {len(self.labels)}")
    # def _uniform_sample(self, frames, n_frames)
    #     stride = max(1, len(frames) // n_frames)
    #     sampled = [frames[i] for i in range(0,len(frames), stride)]
    #     return sampled[:n_frames]
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        frames = self.features[idx]  # Shape: [sequence_length, height, width, channels]
        label = self.labels[idx]

        # Apply transformation to each frame
        transformed_frames = []
        for frame in frames:
            image = Image.fromarray(frame, mode='RGB')
            if self.transform:
                image = self.transform(image)
            transformed_frames.append(image)    

        # Stack frames to maintain 4D input [sequence_length, channels, height, width]
        data = torch.stack(transformed_frames,dim=0)
        data = data.permute(1, 0, 2, 3)
        return data, label

def stratified_split_dataset(features, labels, test_size=0.2, random_state=SEED):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, val_idx in splitter.split(features, labels):
        train_features, val_features = features[train_idx], features[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
    return train_features, train_labels, val_features, val_labels

# def hybrid_sampling(features, labels, target_size_ratio=0.5):
#     # Kết hợp features và labels thành một danh sách
#     dataset = list(zip(features, labels))
    
#     # Phân tích phân phối các lớp
#     label_counts = Counter(labels)
#     print(f"Phân phối các lớp trước khi hybrid sampling: {label_counts}")
    
#     # Xác định số lượng mẫu tối thiểu và tối đa
#     max_class_size = max(label_counts.values())
#     min_class_size = min(label_counts.values())
    
#     # Đặt target size sau khi undersampling lớp chiếm đa số
#     target_majority_size = int(target_size_ratio * max_class_size)
    
#     # Chia dataset theo từng lớp
#     datasets_by_class = {label: [] for label in set(labels)}
#     for feature, label in dataset:
#         datasets_by_class[label].append((feature, label))
    
#     # Thực hiện undersampling cho lớp chiếm đa số
#     hybrid_data = []
#     for label in datasets_by_class:
#         if label_counts[label] == max_class_size:
#             # Giảm lớp chiếm đa số xuống target_majority_size
#             hybrid_data += resample(datasets_by_class[label], 
#                                     replace=False,  # Không sao chép
#                                     n_samples=target_majority_size,  # Số lượng mẫu sau khi giảm
#                                     random_state=SEED)
#         else:
#             hybrid_data += datasets_by_class[label]
    
#     # Thực hiện oversampling cho lớp chiếm thiểu số
#     for label in datasets_by_class:
#         if label_counts[label] == min_class_size:
#             # Tăng lớp chiếm thiểu số lên bằng target_majority_size
#             hybrid_data += resample(datasets_by_class[label], 
#                                     replace=True,  # Sao chép các mẫu ít dữ liệu
#                                     n_samples=target_majority_size,  # Số lượng mẫu sau khi tăng
#                                     random_state=SEED)
    
#     # Trộn ngẫu nhiên lại dữ liệu sau khi sampling
#     np.random.shuffle(hybrid_data)
    
#     # Tách lại features và labels
#     hybrid_features, hybrid_labels = zip(*hybrid_data)
    
#     print(f"Phân phối các lớp sau khi hybrid sampling: {Counter(hybrid_labels)}")

# def hybrid_sampling(features, labels, target_size_ratio=0.7):
#     # Kết hợp features và labels thành một danh sách
#     dataset = list(zip(features, labels))
    
#     # Phân tích phân phối các lớp
#     label_counts = Counter(labels)
#     print(f"Phân phối các lớp trước khi hybrid sampling: {label_counts}")
    
#     # Xác định số lượng mẫu tối đa
#     max_class_size = max(label_counts.values())
    
#     # Đặt target size sau khi undersampling lớp chiếm đa số
#     target_majority_size = int(target_size_ratio * max_class_size)
    
#     # Chia dataset theo từng lớp
#     datasets_by_class = {label: [] for label in set(labels)}
#     for feature, label in dataset:
#         datasets_by_class[label].append((feature, label))
    
#     # Thực hiện undersampling cho lớp chiếm đa số
#     hybrid_data = []
#     for label in datasets_by_class:
#         if label_counts[label] == max_class_size:
#             # Giảm lớp chiếm đa số xuống target_majority_size
#             hybrid_data += resample(datasets_by_class[label], 
#                                     replace=False,  # Không sao chép
#                                     n_samples=target_majority_size,  # Số lượng mẫu sau khi giảm
#                                     random_state=SEED)
#         else:
#             hybrid_data += datasets_by_class[label]
    
#     # Tăng ngẫu nhiên các lớp thiểu số và trung bình
#     for label in datasets_by_class:
#         if label_counts[label] < target_majority_size:
#             # Xác định số lượng mẫu ngẫu nhiên cần tăng cho mỗi lớp thiểu số
#             random_increase_size = random.randint(label_counts[label], target_majority_size)
            
#             # Tăng các lớp đó bằng cách sao chép ngẫu nhiên các mẫu hiện có
#             hybrid_data += resample(datasets_by_class[label], 
#                                     replace=True,  # Sao chép các mẫu ít dữ liệu
#                                     n_samples=random_increase_size,  # Tăng ngẫu nhiên số lượng mẫu
#                                     random_state=SEED)
    
#     # Trộn ngẫu nhiên lại dữ liệu sau khi sampling
#     np.random.shuffle(hybrid_data)
    
#     # Tách lại features và labels
#     hybrid_features, hybrid_labels = zip(*hybrid_data)
    
#     print(f"Phân phối các lớp sau khi hybrid sampling: {Counter(hybrid_labels)}")
    
#     return hybrid_features, hybrid_labels


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Focal Loss for classification tasks.

        Parameters:
        gamma (float): Focusing parameter, typically set between 0 and 5.
        alpha (float or list): Weighting factor for different classes.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
        if isinstance(alpha, (float, int)): 
            self.alpha = torch.tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)

    def forward(self, inputs, targets):
        """
        Forward pass for the Focal Loss.

        Parameters:
        inputs (Tensor): Predicted logits from the model, shape (N, C).
        targets (Tensor): Ground truth labels, shape (N,).

        Returns:
        Tensor: Loss value.
        """
        # Convert to probabilities
        probs = F.softmax(inputs, dim=1)
        log_probs = F.log_softmax(inputs, dim=1)

        targets = targets.view(-1, 1)  # reshape target to match input shape

        probs = probs.gather(1, targets)  # Get the probability for the correct class
        log_probs = log_probs.gather(1, targets)  # Get the log probability for the correct class

        probs = probs.squeeze()
        log_probs = log_probs.squeeze()

        # Focal loss term
        loss = -((1 - probs) ** self.gamma) * log_probs

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.squeeze())
            loss = loss * alpha_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
# %%
