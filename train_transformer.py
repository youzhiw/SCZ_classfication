#!/usr/bin/env python
# coding: utf-8

import torch
import math
import numpy as np
import os
from natsort import natsorted
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_value_
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from scipy.ndimage import zoom
from glob import glob
import nibabel as nib
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights

def downsize_transform(data): 
    target_size = (192, 192, 192)
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    downsampled = torch.nn.functional.interpolate(data, size=target_size, mode='trilinear')

    return downsampled.squeeze(0).squeeze(0)

class TransformerDataset(Dataset):
    def __init__(self, img_dir, grad_dir, transforms = None):
        self.img_dir = img_dir
        self.grad_dir = grad_dir
        self.transforms = transforms
        self.cn_dir = os.path.join(self.img_dir, "MNI152_affine_WB_iso1mm/CN")
        self.scz_dir = os.path.join(self.img_dir, "MNI152_affine_WB_iso1mm/schiz")
        self.grad_cn_dir = os.path.join(self.grad_dir, "MNI152_affine_WB_iso1mm/CN")
        self.grad_scz_dir = os.path.join(self.grad_dir, "MNI152_affine_WB_iso1mm/CN") # change this later when rerun the extract grad script
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
        k = 3
        label = self.labels[idx]
        grad_file = self.samples[idx].split(".")[0]
        if label == 0:
            file_path = os.path.join(self.cn_dir, self.samples[idx])
            grad_path = os.path.join(self.grad_cn_dir, grad_file+ ".nii_activation.nii.gz")
        else:
            file_path = os.path.join(self.scz_dir, self.samples[idx])
            grad_path = os.path.join(self.grad_scz_dir, grad_file+ ".nii_activation.nii.gz")
        one_hot_label = torch.zeros(2)
        one_hot_label[label] = 1
        label = one_hot_label

        # Load the NIfTI image
        img = nib.load(file_path)

        
        grad = nib.load(grad_path)

        # Get the image data array
        img_data = np.float32(img.get_fdata())
        img_data = self.transforms(img_data)

        grad_data = np.float32(grad.get_fdata())
        sums_x = np.sum(grad_data, axis=(1, 2))
        sums_y = np.sum(grad_data, axis=(0, 2))
        sums_z = np.sum(grad_data, axis=(0, 1))

        x_slices = np.argsort(sums_x)[::-1][:k]
        x_copy = x_slices.copy()
        y_slices = np.argsort(sums_y)[::-1][:k]
        y_copy = y_slices.copy()
        z_slices = np.argsort(sums_z)[::-1][:k]
        z_copy = z_slices.copy()

        x_slice = img_data[x_copy, :, :]
        y_slice = img_data[:, y_copy, :].reshape((3, 192, 192))
        z_slice = img_data[:, :, z_copy].reshape((3, 192, 192))

        return np.concatenate((x_slice, y_slice, z_slice), axis = 0), label

class SwinT(nn.Module):
    def __init__(self):
        super(SwinT, self).__init__()
        self.model1 = swin_t(weights = Swin_T_Weights.IMAGENET1K_V1)
        self.model1.features[0][0] = nn.Conv2d(9, 96, kernel_size=(4, 4), stride=(4, 4))
        self.fc1 = nn.Linear(1000, 2)
    def forward(self, x):
        x = self.model1(x)
        x = self.fc1(x)
        return x

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curr_dir = os.getcwd()
    root_dir = os.path.join(curr_dir, 'data')
    grad_root_dir = os.path.join(curr_dir, 'activations')
    folds_dir = [dir for dir in os.listdir(root_dir) if dir.startswith("fold")]
    grads_dir = [os.path.join(grad_root_dir, dir) for dir in folds_dir]
    folds_dir = [os.path.join(root_dir, dir) for dir in folds_dir]
    folds_dir = natsorted(folds_dir)
    grads_dir = natsorted(grads_dir)
    dataloaders = []
    for i in range(len(folds_dir)):
        fold_dir = folds_dir[i]
        grad_dir = grads_dir[i]
        dataset = TransformerDataset(fold_dir, grad_dir, downsize_transform) #, downsize_transform)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        dataloaders.append(dataloader)

    model = SwinT()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss_function = nn.CrossEntropyLoss()
    # check_dir = "/media/youzhi/SSD/bme_project/checkpoints"
    check_dir = os.path.join(curr_dir, 'transformer_checkpoints')
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    min_valid_loss = math.inf
    overall_train_loss = []
    overall_valid_loss = []
    valid_set = dataloaders[1]
    training_sets = [index for index in range(9)if index != 1]

    for i in tqdm(training_sets):
        training_set = dataloaders[i]
        for epoch in (range(20)):
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

if __name__ == '__main__':
    main()