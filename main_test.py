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
from dataset import create_dataset
from dataset import create_test_dataset
from dataset import CrimeDataset
from keras.utils import to_categorical
from test_model import test_model
from test_model import evaluate_precision_recall_f1
from test_model import plot_confusion_matrix
from test_model import plot_roc_curve

SEED = 24
EPOCHS = 15
BATCH_SIZE = 64
CLASSES_LIST = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Other",  "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
IMAGE_HEIGHT , IMAGE_WIDTH = 112, 112
NUM_CLASSES = 14
LEARNING_RATE = 1e-3
# labels = to_categorical(labels, num_classes=NUM_CLASSES)
test_features, test_labels, test_video_files_paths = create_test_dataset()

# Transformations (if any)
transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
])

dataset = CrimeDataset(test_features, test_labels, transform=transform)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Number of test samples: {len(dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelResNet18EarlyFrame(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('models/model_resnet18_2D_CNN_earlyframe_5frame20segments_2.pth'))
criterion = nn.CrossEntropyLoss()
# Test the model
test_loss, test_acc, all_preds, all_probs, all_labels, history = test_model(model, test_loader, criterion,device=device)

# Evaluate metrics
evaluate_precision_recall_f1(all_preds, all_labels, CLASSES_LIST)
plot_confusion_matrix(all_labels, all_preds, CLASSES_LIST)
plot_roc_curve(all_labels, all_probs, CLASSES_LIST)
# %%
