#%%
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
from models import ModelResNet18SingleFrame
from models import ModelResNet18LateFrame
from models import ModelResNet18EarlyFrame
from models import ModelResNet18CNN_LSTM
from models import S3D
from models import VideoClassifier
from dataset import create_dataset
from dataset import CrimeDataset
from dataset import FocalLoss
from dataset import stratified_split_dataset
from train import train_model
from collections import Counter
from torchsummary import summary
SEED = 2408
EPOCHS = 15
BATCH_SIZE = 8
CLASSES_LIST = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Others",  "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
IMAGE_HEIGHT , IMAGE_WIDTH = 112, 112
NUM_CLASSES = 14
LEARNING_RATE = 1e-3
features, labels, video_files_paths = create_dataset()
train_features, train_labels, val_features, val_labels = stratified_split_dataset(features, labels, test_size=0.3,random_state=SEED)

# Tạo transform cho dữ liệu
transform = transforms.Compose([
    # transforms.ToPILImage(), #Chuyển đổi đầu vào (có thể là một mảng NumPy hoặc một tensor) thành hình ảnh PIL
    # transforms.RandomResizedCrop(128), # tăng cường dữ liệu Cắt ngẫu nhiên hình ảnh PIL thành kích thước 128*128 pixel
    # transforms.RandomHorizontalFlip(p=0.5), 
    # Chuyển đổi hình ảnh PIL thành tensor PyTorch. Hình ảnh được chuyển đổi từ định dạng PIL 
    # (với giá trị pixel từ 0 đến 255) thành tensor với giá trị từ 0.0 đến 1.0.\
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    #Chuẩn hóa tensor hình ảnh bằng cách sử dụng giá trị trung bình và độ lệch chuẩn cho mỗi kênh (kênh màu). 
    # Bước này giúp chuẩn hóa đầu vào, giúp mô hình hội tụ nhanh hơn trong quá trình huấn luyện.
    # transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    #Chuyển đổi tensor thành mảng NumPy và thay đổi thứ tự của các trục. 
    # x.numpy() chuyển tensor thành mảng NumPy,
    # và np.rollaxis(x, 0, 3) thay đổi thứ tự các trục
    # sao cho chiều kênh (ban đầu ở vị trí 0) được chuyển sang vị trí 3. 
    # Điều này thường được thực hiện để biến đổi hình ảnh từ định dạng 
    # (C, H, W) (kênh, chiều cao, chiều rộng) sang (H, W, C) (chiều cao, chiều rộng, kênh),
    # lambda x: np.rollaxis(x.numpy(), 0, 3)
])
# Áp dụng hybrid sampling cho tập huấn luyện để cân bằng dữ liệu
# train_features, train_labels = hybrid_sampling(train_features, train_labels, target_size_ratio=0.5)

# Create the dataset
train_dataset = CrimeDataset(features=train_features, labels=train_labels, transform=transform)
val_dataset = CrimeDataset(features=val_features, labels=val_labels, transform=transform)

print(f"Number of samples in training set: {len(train_dataset)}")
print(f"Number of samples in validation set: {len(val_dataset)}")

# tạo data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ModelResNet18SingleFrame(num_classes=NUM_CLASSES)
# model = ModelResNet18LateFrame(num_classes=NUM_CLASSES)
model = ModelResNet18EarlyFrame(num_classes=NUM_CLASSES, num_input_channel=5*3)
summary(model, (5, 3, 112, 112))
# model = ModelResNet18CNN_LSTM(num_classes=NUM_CLASSES)
# model = S3D(num_classes=NUM_CLASSES)
# model = VideoClassifier(num_classes=NUM_CLASSES)
criterion = FocalLoss(gamma=1.5 ,alpha=0.25, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, device=device)
# # Save the trained model
model_dir = "models"
model_save_path = os.path.join(model_dir, "model_resnet18_2D_CNN_earlyframe_5frame20segments_2.pth")
torch.save(trained_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# lưu kết quả vào file txt
results = "results"
if not os.path.exists(results):
    os.makedirs(results)
results_file_name = "model_resnet18_2D_CNN_earlyframe_5frame20segments_2.txt"
results_file_path = os.path.join(results, results_file_name)
with open(results_file_path, 'w') as f:
    f.write(f"Training results for model: model_resnet18_2D_CNN_earlyframe_5frame20segments_2\n\n")
    f.write(f"Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\n")
    for epoch in range(EPOCHS):
        f.write(f"{epoch+1}\t{history['train_loss'][epoch]:.4f}\t{history['train_acc'][epoch]:.2f}\t"
                f"{history['val_loss'][epoch]:.4f}\t{history['val_acc'][epoch]:.2f}\n")
# %%
