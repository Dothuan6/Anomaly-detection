#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import create_dataset
import cv2

CLASSES_LIST = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Others",  "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
DATASET_DIR = "dataset/Train"
TEST_DIR = "dataset/Test"
analytics_dir = 'analytics'
os.makedirs(analytics_dir, exist_ok=True)

# Tạo và phân tích dữ liệu
def analyze_class_distribution(labels):
    print('Analyzing class distribution...')
    unique, counts = np.unique(labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    plt.figure(figsize=(10, 6))
    plt.bar(CLASSES_LIST, counts)
    plt.xlabel('Labels')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.show()
    
    # Lưu kết quả vào file
    file_path = os.path.join(analytics_dir, 'class_distribution.txt')
    print(f'Saving class distribution to {file_path}...')
    with open(file_path, 'w') as f:
        f.write('Class,Count\n')
        for class_name, count in class_distribution.items():
            f.write(f'{class_name},{count}\n')
    
# _, labels, _ = create_dataset()
# analyze_class_distribution(labels)
# Phân phối độ dài video
def analyze_video_lengths(video_paths):
    print('Analyzing video lengths...')
    video_lengths = []

    for video_path in video_paths:
        video_reader = cv2.VideoCapture(video_path)
        fps = int(video_reader.get(cv2.CAP_PROP_FPS))
        frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_count / fps
        video_lengths.append(duration_seconds)
        video_reader.release()

    plt.hist(video_lengths, bins=20, edgecolor='black')
    plt.xlabel('Video Length (seconds)')
    plt.ylabel('Number of Videos')
    plt.title('Video Length Distribution')
    plt.show()

    # Lưu kết quả vào file
    file_path = os.path.join(analytics_dir, 'video_lengths.txt')
    print(f'Saving video lengths to {file_path}...')
    with open(file_path, 'w') as f:
        f.write('Video Length (seconds)\n')
        for length in video_lengths:
            f.write(f'{length}\n')

# # Gọi hàm phân tích
# _, _, video_files_paths = create_dataset()
# analyze_video_lengths(video_files_paths)

# đếm tổng số giờ video trong tập dữ liệu
def save_total_duration(directory, file_name):
    file_path = os.path.join(analytics_dir, file_name)
    total_seconds = 0
    
    for class_name in CLASSES_LIST:
        class_dir = os.path.join(directory, class_name)
        files_list = os.listdir(class_dir)
        
        for file_name in files_list:
            video_file_path = os.path.join(class_dir, file_name)
            video_reader = cv2.VideoCapture(video_file_path)
            fps = video_reader.get(cv2.CAP_PROP_FPS)
            total_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps > 0:
                duration_seconds = total_frames / fps
                total_seconds += duration_seconds
            
            video_reader.release()

    total_hours = total_seconds / 3600

    with open(file_path, 'w') as f:
        f.write(f'Total duration in hours: {total_hours:.2f}\n')

# tổng số video
def save_video_count(directory, file_name):
    file_path = os.path.join(analytics_dir, file_name)
    total_videos = 0
    
    for class_name in CLASSES_LIST:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            files_list = os.listdir(class_dir)
            total_videos += len(files_list)
    
    with open(file_path, 'w') as f:
        f.write(f'Total number of videos: {total_videos}\n')

# Tỉ lệ trích xuất frame
def analyze_segments_distribution(features, file_name, image_name):
    print('Analyzing segments distribution...')
    segment_lengths = [len(segments) for segments in features]
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.hist(segment_lengths, bins=20, edgecolor='black')
    plt.xlabel('Number of Segments per Video')
    plt.ylabel('Number of Videos')
    plt.title('Segment Distribution')

    # Đường dẫn lưu hình ảnh vào folder images
    image_dir = 'images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    image_path = os.path.join(image_dir, image_name)
    plt.savefig(image_path)  # Lưu biểu đồ vào thư mục images
    print(f'Image saved to {image_path}')

    plt.show()

    # Lưu kết quả vào file txt
    analytics_dir = 'analytics'
    if not os.path.exists(analytics_dir):
        os.makedirs(analytics_dir)
    
    file_path = os.path.join(analytics_dir, file_name)
    print(f'Saving segments distribution to {file_path}...')
    with open(file_path, 'w') as f:
        f.write('Number of Segments per Video\n')
        for length in segment_lengths:
            f.write(f'{length}\n')

# Gọi hàm phân tích
features, _, _ = create_dataset()

# Thực hiện phân tích và lưu kết quả
# analyze_class_distribution(labels)
# analyze_video_lengths(video_files_paths)
# save_total_duration(DATASET_DIR, 'train_total_duration.txt')
# save_total_duration(TEST_DIR, 'test_total_duration.txt')
# save_video_count(DATASET_DIR, 'train_video_count.txt')
# save_video_count(TEST_DIR, 'test_video_count.txt')
analyze_segments_distribution(features, 'segments_distribution.txt', 'segments_distribution.png')
# %%
