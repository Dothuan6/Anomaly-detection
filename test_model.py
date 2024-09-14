# %%
import time
import torch
import os 
import tqdm as tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from dataset import CrimeDataset
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("Torch-Cls")
seed = 24
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test_model(model, test_loader, criterion, device='cpu'):
    model.eval()  # Set model to evaluation mode
    history = {
        'test_loss': [],
        'test_acc': [] ,
    }

    running_loss = 0.0
    running_items = 0
    running_corrects = 0

    all_preds = []
    all_probs = []  # To store probabilities
    all_labels = []
    best_test_acc = 0.0
    
    since = time.time()  # Start timing the testing process
    
    with torch.no_grad():  # Disable gradient calculation
        data_loader = test_loader 
        
        _phase = tqdm.tqdm(data_loader, 
                           total=len(data_loader), 
                           desc="Testing",
                           unit="batch",
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        for inputs, labels in _phase:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)  # Get class probabilities
            _, preds = torch.max(outputs, 1)  # Get predictions
            loss = criterion(outputs, labels)  # Compute loss
            
            # Statistics
            running_items += outputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Append results for further analysis
            all_preds.extend(preds.cpu().numpy())  # Collect predictions
            all_probs.extend(probs.cpu().numpy())  # Collect probabilities
            all_labels.extend(labels.cpu().numpy())  # Collect true labels
            
            # Loss and accuracy per batch
            test_loss = running_loss / running_items
            test_acc = running_corrects.double() / running_items
            
            desc = f"Testing loss {test_loss:.4f} acc {test_acc:.4f}"
            _phase.set_description(desc)
        
        # Save final results
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc.item())
        
        best_test_acc = max(best_test_acc, test_acc)
        print(f"Độ chính xác cao nhất: {best_test_acc:.4f}")
        
    # Time tracking
    time_elapsed = time.time() - since
    history['INFO'] = (
        "Quá trình kiểm thử trong vòng: {:.0f}h {:.0f}m {:.0f}s - Độ chính xác cao nhất: {:.4f}".format(
            time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60, best_test_acc
        )
    )
    
    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')
    print(f'Number of test samples: {running_items}')
    print(f'Number of correct predictions: {running_corrects}')
    print(f'Number of wrong predictions: {running_items - running_corrects}')
    
    return test_loss, test_acc, all_preds, all_probs, all_labels, history

def evaluate_precision_recall_f1(predictions, labels, class_names):
    """
    Đánh giá Precision, Recall và F1-Score cho các lớp.
    
    Args:
        predictions (list of int): Các dự đoán của mô hình.
        labels (list of int): Các nhãn thực tế.
        class_names (list of str): Tên của các lớp.
        
    precision: tỉ lệ số lượng dự đoán đúng trên tổng số dự đoán.
    recall: tỉ lệ số lượng dự đoán đúng trên tổng số nhãn thực tế.
    f1-score: trung bình điều hòa giữa precision và recall.
    
    Returns:
        None
    """
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    print("Classification Report:")
    print(report)

def plot_confusion_matrix(labels, predictions, class_names):
    """
    Vẽ Confusion Matrix.
    
    Args:
        labels (list of int): Các nhãn thực tế.
        predictions (list of int): Các dự đoán của mô hình.
        class_names (list of str): Tên của các lớp.
    
    Returns:
        None
    """
    cm = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    
def plot_roc_curve(labels, predictions, class_names):
    """
    Vẽ AUC-ROC Curve.
    
    Args:
        labels (list of int): Các nhãn thực tế.
        predictions (numpy array): Xác suất dự đoán của mô hình.
        num_classes (int): Số lớp.
    
    Returns:
        None
    """
    num_classes = len(class_names)
    # Chuyển nhãn thành định dạng nhị phân
    y_true = label_binarize(labels, classes=range(num_classes))
    y_score = np.array(predictions)
    plt.figure(figsize=(10, 7))
    for i in range(num_classes):    
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='Class {0} (area = {1:0.2f})'.format(class_names[i], roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

# def predict(model, video_path, transform, max_frames=16):
#     frames = CrimeDataset.load_and_process_video(video_path, max_frames)
#     if frames is None:
#         print("Error loading video.")
#         return None
#     frames = transform(frames).unsqueeze(0).to(device)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(frames)
#         _, preds = torch.max(outputs, 1)
#     return preds.item()

# video_path = '/kaggle/input/kinetics/train/writing/o1S2VoH3stY.mp4'
# pred_class = predict(model, video_path, transform)
# print(f'Predicted class: {pred_class}')