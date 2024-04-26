#!/usr/bin/env python
# coding: utf-8


import math
import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, roc_curve, auc

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
import nibabel as nib
from torchvision.transforms import functional as F
from natsort import natsorted
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

import seaborn as sns

class VGG(nn.Module):
    def __init__(self, feature_extractor):
        super(VGG, self).__init__()
        self.feature_extractor = feature_extractor

        self.T1_feature_dimension = 1 * 128 * 6 * 6 * 6 # raw T1 data 200*200*200
        
        feature_dimension = self.T1_feature_dimension

        self.classifier = nn.Sequential(
            nn.Dropout(), # commented on 20201230
            nn.Linear(feature_dimension, 1024), #nn.Linear(feature_dimension, 2048),
            nn.ReLU(True),

            nn.Dropout(),
            nn.Linear(1024, 128), #nn.Linear(2048, 512),
            nn.Sigmoid(),
            nn.Linear(128, 2), #nn.Linear(512, 2),
        )

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x1):
        x1 = self.feature_extractor(x1)
        x = x1.view(x1.size(0), -1)

        x = self.classifier(x)
        return x

# 5-27-2021
class SE_block(nn.Module):
    se_ratio = 16
    def __init__(self,ch,ratio=se_ratio):
        super(SE_block, self).__init__()

        # Both should use the same complex number for initiaization which is then split into real and imaginary parts
        # Weight initialiation using the real part 
        self.globalpooling = nn.AdaptiveAvgPool3d(1)
        # Weight initialization using the imag part
        self.linear1 = nn.Linear(ch,ch//ratio)
        self.relu1 = nn.ReLU()
        self.lienar2 = nn.Linear(ch//ratio,ch)
        self.sigmoid1 = nn.Sigmoid()
        self.ch = ch

    def forward(self, x):
        x_init=x
        x1=self.globalpooling(x).squeeze()
        x1=self.linear1(x1)
        x1=self.relu1(x1)
        x1=self.lienar2(x1)
        x1=self.sigmoid1(x1).reshape(x.shape[0],self.ch,1,1,1)

        return torch.mul(x_init,x1)





def make_layers(cfg, batch_norm=True, se_block=True):
    layers = []
    DropoutRate = 0.10

    in_channels = 1

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)] # MaxPool without Dropout # commented on 20201230
            #layers += [nn.Dropout(p=DropoutRate), nn.MaxPool3d(kernel_size=2, stride=2)] # MaxPool with Dropout rate of 0.25 # commented on 20201230
        else:
            if batch_norm:
                if se_block:
                    conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                    se3d = SE_block(ch=v)
                    layers += [conv3d, nn.BatchNorm3d(v), se3d, nn.ReLU(inplace=True)]
                else:
                    conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                    layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    feature_extractor_T1 = nn.Sequential(*layers)

    return feature_extractor_T1 

cfg = {
    'A': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128], #'A': [32, 'M', 64, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'], 
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 
          128, 128, 128, 128, 'M'],
    'F': [32, 'M', 128, 'M', 512, 'M'],
}




def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E']))

def vgg16_bn():
    """VGG 16-layer model (configuration 'D') with batch normalization"""
    return VGG(make_layers(cfg['D']))

def vgg11_bn():
    """VGG 11-layer model (configuration 'A') with batch normalization"""
    return VGG(make_layers(cfg['A']))

def vgg6_bn():
    """VGG 8-layer model (configuration 'F') with batch normalization"""
    return VGG(make_layers(cfg['F']))








def downsize_transform(data): 
    target_size = (96, 96, 96)
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    downsampled = torch.nn.functional.interpolate(data, size=target_size, mode='trilinear')

    return downsampled.squeeze(0)




class CustomDataset(Dataset):
    def __init__(self, root_dir, transforms = None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.cn_dir = os.path.join(self.root_dir, "MNI152_affine_WB_iso1mm/CN")
        self.scz_dir = os.path.join(self.root_dir, "MNI152_affine_WB_iso1mm/schiz")
        self.samples, self.labels = self._load_samples()

    def _load_samples(self):
        samples = []
        
        samples = [file for file in os.listdir(self.cn_dir) if file.endswith(".nii.gz")]
        labels = [0] * len(samples)
        samples += [file for file in os.listdir(self.scz_dir) if file.endswith(".nii.gz")]
        labels += [1] * (len(samples) - len(labels))

        return samples, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if label == 0:
            file_path = os.path.join(self.cn_dir, self.samples[idx])
        else:
            file_path = os.path.join(self.scz_dir, self.samples[idx])
        one_hot_label = torch.zeros(2)
        one_hot_label[label] = 1
        label = one_hot_label

        # Load the NIfTI image
        img = nib.load(file_path)

        # Get the image data array
        img_data = np.float32(img.get_fdata())[0:192, :, 0:192]
        if self.transforms:
            img_tensor = self.transforms(img_data)
        else:
            img_tensor = torch.from_numpy(img_data).unsqueeze(0)
        return img_tensor, label




def train_net(net, epochs, train_dataloader, valid_loader, optimizer, loss_function, device ):
    net.to(device)
    ret_train_loss = []
    ret_valid_loss = []

    for epoch in range(epochs):
        net.train()

        train_loss = []
        for i, (img, label) in enumerate(train_dataloader):
            img= img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            y_pred = net(img)
            loss = loss_function(y_pred, label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # print(f'{i + 1}/{len(train_dataloader)}| current training loss: {train_loss[-1]}', end='\r')

        train_epoch_loss = np.mean(train_loss)
        ret_train_loss.append(train_epoch_loss)
        # print(f'epoch {epoch}| training loss: {train_epoch_loss}', end='\r')

        # Validation phase
        net.eval()
        valid_loss = []
        with torch.no_grad():
            for i, (img, label) in enumerate(valid_loader):
                img = img.to(device)
                label = label.to(device)
                y_pred = net(img)
                loss = loss_function(y_pred, label)
                valid_loss.append(loss.item())
                # print(f'{i + 1}/{len(valid_loader)}| current validation loss: {valid_loss[-1]}', end='\r')

        epoch_vloss = np.mean(valid_loss)

        print(f"epoch {epoch} | training loss: {train_epoch_loss:.4f} | validation loss: {epoch_vloss:.4f}")
        ret_valid_loss.append(epoch_vloss)

    return ret_train_loss, ret_valid_loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg11_bn()
    root_dir = "/media/youzhi/SSD/bme_project/data"
    folds_dir = [dir for dir in os.listdir(root_dir) if dir.startswith("fold")]
    folds_dir = [os.path.join(root_dir, dir) for dir in folds_dir]
    folds_dir = natsorted(folds_dir)

    dataloaders = []
    for i in range(len(folds_dir)):
        fold_dir = folds_dir[i]
        dataset = CustomDataset(fold_dir, downsize_transform) #, downsize_transform)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        dataloaders.append(dataloader)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        loss_function = nn.CrossEntropyLoss()
        check_dir = "/media/youzhi/SSD/bme_project/checkpoints"
        temp_dir = "/media/youzhi/SSD/bme_project/checkpoints/temp.pth"
        min_valid_loss = math.inf
        overall_train_loss = []
        overall_valid_loss = []
        torch.save(model.state_dict(), temp_dir)
        valid_set = dataloaders[1]
        training_sets = [index for index in range(9)if index != i]

    for i in tqdm(range(len(training_sets))):
        training_set = dataloaders[i]
        for epoch in tqdm(range(20)):
            train_loss, valid_loss = train_net(model, 1, training_set, valid_set, optimizer, loss_function, device)
            if valid_loss[0] < min_valid_loss:
                min_valid_loss = valid_loss[0]
                model_filename = f'Epoch_{epoch}_VLoss_{valid_loss[0]:.4f}.pth'
                model_path = os.path.join(check_dir, model_filename)
                torch.save(model.state_dict(), model_path)
            overall_train_loss.append(train_loss)
            overall_valid_loss.append(valid_loss)
        

    np.savetxt('training_loss.txt', overall_train_loss)

    np.savetxt('valid_loss.txt', overall_valid_loss)