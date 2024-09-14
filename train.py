# %%
import time
import torch
import os 
import tqdm as tqdm
import copy
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    history = {
        'train_loss': [],
        'train_acc': [] ,
        "val_loss": [],
        "val_acc": [],
        'lr':[],
    }
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        LOGGER.info(f"Epoch {epoch + 1}/{num_epochs}:")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_items = 0
            running_corrects = 0
            
            data_loader = train_loader if phase == 'train' else val_loader
        
            _phase = tqdm.tqdm(data_loader, 
                            total=len(data_loader), 
                            desc=f"Epoch {epoch + 1}/{num_epochs}",
                            unit="batch",
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            for inputs, labels in _phase:
                inputs = inputs.to(device)
                labels = labels.to(device)
        
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) #đưa input qua models batch_size * số classes
                    _, preds = torch.max(outputs, 1) # lấy vị trí có giá trị lớn nhất
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_items += outputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
                
                epoch_loss = running_loss / running_items
                epoch_acc = running_corrects.double() / running_items
                
                desc = f"Epoch {epoch + 1} loss {epoch_loss:.4f} acc {epoch_acc:.4f}"
                _phase.set_description(desc)
            
            if phase == "train":
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    history['best_epoch'] = epoch + 1
                print(f"Độ chính xác cao nhất: {best_val_acc:.4f}")
    
    return model, history
# %%