#%%
from torchvision import models
from keras.layers import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18

CLASSES_LIST = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Normal",  "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
weights = R3D_18_Weights.DEFAULT
# single-frame : pooling không nhận thức được thời gian
# image -> CNN -> MLP(classes dự đoán) -> output(giá trị trung bình)
class ModelResNet18SingleFrame(nn.Module):
    def __init__(self, num_classes=14):
        super(ModelResNet18SingleFrame, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) ## load pre-trained weights from torchvision do phiên bản không còn sử dụng pre-trained nữa nên thay bằng weights='imagenet'
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
        )
        
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x_3d):
        #(bs, C, T, H, W) -> (bs, T, C, H, W)
        x_3d = x_3d.permute(0, 2, 1, 3, 4) # đảo chiều của channel và time
        
        #output của resnet50
        logits = []
        for t in range(x_3d.size(1)): # theo chiều T (time) lước qua từng bức ảnh
            # (bs, C, H, W) 
            out = self.resnet(x_3d[:, t, :, :, :]) #output của resnet50 có 512 vector
            
            x = self.fc1(out)
            x = F.relu(x)
            x = self.fc2(x)
            
            logits.append(x)
        
        # mean pooling
        # stack mỗi sample thành 1 tensor
        # sau đó cộng trung bình
        logits = torch.stack(logits, dim=0) 
        logits = torch.mean(logits, dim=0)
        return logits
    
class ModelResNet18LateFrame(nn.Module):
    def __init__(self, num_classes=14):
        super(ModelResNet18LateFrame, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) ## load pre-trained weights from torchvision do phiên bản không còn sử dụng pre-trained nữa nên thay bằng weights='imagenet'
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
        )
        
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x_3d):
        #(bs, C, T, H, W) -> (bs, T, C, H, W)
        x_3d = x_3d.permute(0, 2, 1, 3, 4) # đảo chiều của channel và time
        
        #output của resnet50
        features = []
        for t in range(x_3d.size(1)): # theo chiều T (time) lước qua từng bức ảnh
            # (bs, C, H, W) 
            out = self.resnet(x_3d[:, t, :, :, :]) #output của resnet50 có 512 vector
            features.append(out)
        # avg pooling
        out = torch.mean(torch.stack(features),0)
        x = self.fc1(out)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class ModelResNet18EarlyFrame(nn.Module):
    def __init__(self, num_classes=14, num_input_channel=5*3):
        super(ModelResNet18EarlyFrame, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) ## load pre-trained weights from torchvision do phiên bản không còn sử dụng pre-trained nữa nên thay bằng weights='imagenet'
        self.resnet.conv1 = nn.Conv2d(num_input_channel ,64, kernel_size= 3, stride=1, padding=1, bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
        )
        
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x_3d):
        #(bs, C, T, H, W) -> (bs, T, C, H, W)
        # concatenate all C and T demensions to make it (bs, C*T, H, W)
        x_3d = x_3d.permute(0, 2, 1, 3, 4) # đảo chiều của channel và time
        x_3d= x_3d.reshape(x_3d.size(0), x_3d.size(1)*x_3d.size(2), x_3d.size(3), x_3d.size(4))
        
        out = self.resnet(x_3d)
        x = self.fc1(out)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class ModelResNet18CNN_LSTM(nn.Module):
    def __init__(self, num_classes=14):
        super(ModelResNet18CNN_LSTM, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) ## load pre-trained weights from torchvision do phiên bản không còn sử dụng pre-trained nữa nên thay bằng weights='imagenet'
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
        )
        self.lstm = nn.LSTM(input_size=512, hidden_size=389, num_layers=3)
        self.fc1 = nn.Linear(389, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x_3d):
        #(bs, C, T, H, W) -> (bs, T, C, H, W)
        # concatenate all C and T demensions to make it (bs, C*T, H, W)
        x_3d = x_3d.permute(0, 2, 1, 3, 4) # đảo chiều của channel và time
        
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)
            
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x

class InceptionBlock(nn.Module):
    def __init__(
        self, input_dim,
        num_outputs_0_0a,
        num_outputs_1_0a, 
        num_outputs_1_0b, 
        num_outputs_2_0a, 
        num_outputs_2_0b,
        num_outputs_3_0b,
        gating = True  
    ):
        super(InceptionBlock, self).__init__()
        self.conv_b0 = STConv3D(input_dim, num_outputs_0_0a,[1, 1, 1])
        self.conv_b1_a = STConv3D(input_dim, num_outputs_1_0a,[1, 1, 1])
        self.conv_b1_b = STConv3D(num_outputs_1_0a, num_outputs_1_0b,[3, 3, 3], padding = 1 , separable = True)
        self.conv_b2_a = STConv3D(input_dim, num_outputs_2_0a,[1, 1, 1])
        self.conv_b2_b = STConv3D(num_outputs_2_0a, num_outputs_2_0b,[3, 3, 3], padding = 1, separable = True)
        
        self.maxpool_b3 = nn.MaxPool3d((3,3,3), stride=1, padding=1)
        self.conv_b3_b = STConv3D(input_dim, num_outputs_3_0b,[1, 1, 1])
        self.gating = gating
        self.output_dim = (
            num_outputs_0_0a + num_outputs_1_0b + num_outputs_2_0b + num_outputs_3_0b
        )
        
        if gating:
            self.gating_b0 = SelfGating(num_outputs_0_0a)
            self.gating_b1 = SelfGating(num_outputs_1_0b)
            self.gating_b2 = SelfGating(num_outputs_2_0b)
            self.gating_b3 = SelfGating(num_outputs_3_0b)
            
    def forward(self, input):
        b0 = self.conv_b0(input)
        b1 = self.conv_b1_a(input)
        b1 = self.conv_b1_b(b1)
        b2 = self.conv_b2_a(input)
        b2 = self.conv_b2_b(b2)
        b3 = self.maxpool_b3(input)
        b3 = self.conv_b3_b(b3)
        if self.gating:
            b0 = self.gating_b0(b0)
            b1 = self.gating_b1(b1)
            b2 = self.gating_b2(b2)
            b3 = self.gating_b3(b3)
        return torch.cat([b0, b1, b2, b3], dim=1)

class SelfGating(nn.Module):
    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        
    def forward(self, input_tensor):
        spatiotemporal_average = torch.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = torch.sigmoid(weights)
        return weights[:,:,None,None,None] * input_tensor

class STConv3D(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size,stride = 1,  padding=0, separable=False):
        super(STConv3D, self).__init__() 
        self.separable = separable
        self.relu = nn.ReLU(inplace=True)
        assert len(kernel_size) == 3
        if separable and kernel_size[0] != 1:
            spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
            temporal_kernel_size = [kernel_size[0], 1, 1]
            if isinstance(stride, list) and len(stride) == 3:
                spatial_stride = [1, stride[1], stride[2]]
                temporal_stride = [stride[0], 1, 1]
            else:
                spatial_stride = [1, stride, stride]
                temporal_stride = [stride, 1, 1]
            if isinstance(padding, list) and len(padding) == 3:
                spatial_padding = [0, padding[1], padding[2]]
                temporal_padding = [padding[0], 0, 0]
            else:
                spatial_padding = [0, padding, padding]
                temporal_padding = [padding, 0, 0]
        if separable:
            self.conv1 = nn.Conv3d(
                input_dim,
                output_dim,
                kernel_size=spatial_kernel_size,
                stride=spatial_stride,
                padding=spatial_padding,
                bias=False,
            )
            self.bn1 = nn.BatchNorm3d(output_dim)
            self.conv2 = nn.Conv3d(
                output_dim,
                output_dim,
                kernel_size=temporal_kernel_size,
                stride=temporal_stride,
                padding=temporal_padding,
                bias=False,
            )
            self.bn2 = nn.BatchNorm3d(output_dim)
        else:
            self.conv1 = nn.Conv3d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.bn1 = nn.BatchNorm3d(output_dim)
    def forward(self, input):
        out = self.relu(self.bn1(self.conv1(input)))
        if self.separable:
            out = self.relu(self.bn2(self.conv2(out)))
        return out

class MaxPool3dPadding(nn.Module):
    def __init__(self, kernel_size, stride = None, padding = 'SAME'):
        super(MaxPool3dPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = self._get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = nn.ConstantPad3d(padding_shape, 0)
        self.pool = nn.MaxPool3d(kernel_size, stride, ceil_mode=True)
    def _get_padding_shape(self, filter_shape, stride):
        def _pad_top_bottom(filter_dim, stride_val):
            pad_along = max(filter_dim - stride_val, 0)
            pad_top = pad_along // 2
            pad_bottom = pad_along - pad_top
            return pad_top, pad_bottom
        
        padding_shape = []
        for filter_dim, stride_val in zip(filter_shape, stride):
            pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
            padding_shape.append(pad_top)
            padding_shape.append(pad_bottom)
        depth_top = padding_shape.pop(0)
        depth_bottom = padding_shape.pop(0)
        padding_shape.append(depth_top)
        padding_shape.append(depth_bottom)
        
        return tuple(padding_shape)
    
    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class S3D(nn.Module):
    def __init__(
        self, dict_path = None, num_classes = 14, gating = True, space_to_depth = True 
    ):
        super(S3D, self).__init__()
        self.num_classes = num_classes
        self.gating = gating
        self.space_to_depth = space_to_depth
        if space_to_depth:
            self.conv1 = STConv3D(5,64, [2,4,4], stride=1, padding=(1,2,2), separable=False)
        else:
            self.conv1 = STConv3D(3,64, [3,7,7], stride=2, padding=(1,3,3), separable=False)
        
        self.conv_2b = STConv3D(64,64, [1,1,1], separable=False)
        self.conv_2c = STConv3D(64,192, [3,3,3], padding=1, separable=True)
        self.gating = SelfGating(192)
        self.maxpool_2a = MaxPool3dPadding(
            kernel_size=(1 ,3, 3), stride=(1, 2, 2), padding='SAME'
        )
        self.maxpool_3a = MaxPool3dPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME'
        )
        self.mixed_3b = InceptionBlock(
            192, 64, 96, 128, 16, 32, 32
        )
        self.mixed_3c = InceptionBlock(
            self.mixed_3b.output_dim, 128, 128, 192, 32, 96, 64
        )
        self.maxpool_4a = MaxPool3dPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME'
        )
        self.mixed_4b = InceptionBlock(
            self.mixed_3c.output_dim, 192, 96, 208, 16, 48, 64
        )
        self.mixed_4c = InceptionBlock(
            self.mixed_4b.output_dim, 160, 112, 224, 24, 64, 64
        )
        self.mixed_4d = InceptionBlock(
            self.mixed_4c.output_dim, 128, 128, 256, 24, 64, 64
        )
        self.mixed_4e = InceptionBlock(
            self.mixed_4d.output_dim, 112, 144, 288, 32, 64, 64
        )
        self.mixed_4f = InceptionBlock(
            self.mixed_4e.output_dim, 256, 160, 320, 32, 128, 128
        )
        self.maxpool_5a = self.maxPooling3d_5a_2x2 = MaxPool3dPadding(
            kernel_size=(2,2,2), stride=(2, 2, 2), padding='SAME'
        )
        self.mixed_5b = InceptionBlock(
            self.mixed_4f.output_dim, 256, 160, 320, 32, 128, 128
        )
        self.mixed_5c = InceptionBlock(
            self.mixed_5b.output_dim, 384, 192, 384, 48, 128, 128
        )
        self.fc = nn.Linear(self.mixed_5c.output_dim, num_classes)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        
        net = self.conv1(inputs)
        net = self.maxpool_2a(net)
        net = self.conv_2b(net)
        net = self.conv_2c(net)
        if self.gating:
            net = self.gating(net)
        net = self.maxpool_3a(net)
        net = self.mixed_3b(net)
        net = self.mixed_3c(net)
        net = self.maxpool_4a(net)
        net = self.mixed_4b(net)
        net = self.mixed_4c(net)
        net = self.mixed_4d(net)
        net = self.mixed_4e(net)
        net = self.mixed_4f(net)
        net = self.maxpool_5a(net)
        net = self.mixed_5b(net)
        net = self.mixed_5c(net)
        net = torch.mean(net, dim=[2, 3, 4])
        
        return self.fc(net)

class VideoClassifier(nn.Module):
    
    def __init__(self, num_classes):
        super(VideoClassifier, self).__init__()
        self.model = r3d_18(weights = weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)


# %%
