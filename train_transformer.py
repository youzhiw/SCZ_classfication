#!/usr/bin/env python
# coding: utf-8

import torch
import math
import numpy as np
import os
from natsort import natsorted
from torch import optim
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.nn.utils import clip_grad_value_
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from scipy.ndimage import zoom
from glob import glob
import nibabel as nib
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from torchvision.models import swin_b, Swin_B_Weights

def downsize_transform(data): 
    target_size = (224, 224, 224)
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    downsampled = torch.nn.functional.interpolate(data, size=target_size, mode='trilinear')

    return downsampled.squeeze(0).squeeze(0)

def naive_sample(images):
    means = [np.mean(img) for img in images]
    non_zero_indices = np.nonzero(means)
    #mode = np.argmax(means)
    mini = np.min(non_zero_indices)
    maxi = np.max(non_zero_indices) 
    adj = (maxi-mini) / 4
    mini_adj = np.min(non_zero_indices) + adj
    maxi_adj = np.max(non_zero_indices) - adj
    return np.linspace(mini_adj, maxi_adj, 15, dtype=int)

def get_k_significant(grads, axis):

    if axis == "y":
        grads = grads.permute(1, 0, 2)
    elif axis == "z":
        grads = grads.permute(3, 1, 2)
    means = [np.mean(img) for img in grads]
    non_zero_indices = np.nonzero(means)

    mini = np.min(non_zero_indices)
    maxi = np.max(non_zero_indices)
    adj = (maxi-mini) / 4
    mini_adj = np.min(non_zero_indices) + adj
    maxi_adj = np.max(non_zero_indices) - adj
    return np.linspace(mini_adj, maxi_adj, 15, dtype=int)

class TransformerDataset(Dataset):

    def __init__(self, img_dir, grad_dir, transforms = None):
        self.img_dir = img_dir
        self.grad_dir = grad_dir
        self.transforms = transforms
        self.cn_dir = os.path.join(self.img_dir, "MNI152_affine_WB_iso1mm/CN")
        self.scz_dir = os.path.join(self.img_dir, "MNI152_affine_WB_iso1mm/schiz")
        self.grad_cn_dir = os.path.join(self.grad_dir, "MNI152_affine_WB_iso1mm/CN")

        # change this later when rerun the extract grad script
        self.grad_scz_dir = os.path.join(self.grad_dir, "MNI152_affine_WB_iso1mm/schiz")
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
        #grads = nib.load(grad_path)

        # Get the image data array
        img_data = np.float32(img.get_fdata())
        slices = naive_sample(img_data)
        img_data = self.transforms(img_data)

        z_slices = img_data[:, :, slices].permute(2, 0, 1)
        return z_slices, label


class SwinT(nn.Module):
    def __init__(self):
        super(SwinT, self).__init__()
        self.model1 = swin_b(weights = Swin_B_Weights.IMAGENET1K_V1)
        # self.model1.features[0][0] = nn.Conv2d(9, 96, kernel_size=(4, 4), stride=(4, 4))
        #for param in self.model1.parameters():
        #    param.requires_grad = False
        #
        #for param in self.model1.head.parameters():
        #    param.requires_grad = True
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
        pbar = tqdm(enumerate(train_dataloader), leave=False)
        for i, (img, label) in pbar:
            batch_shape = img.shape[1]
            img = img.view(-1, 224, 224)
            img = img.unsqueeze(1).repeat(1, 3, 1, 1)
            label = torch.repeat_interleave(label, batch_shape, dim=0)
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            y_pred = net(img)
            loss = loss_function(y_pred, label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # print(f'{i + 1}/{len(train_dataloader)}| current training loss: {train_loss[-1]}', end='\r')
            pbar.set_description(f'current training loss: {train_loss[-1]:.4f}')

        train_epoch_loss = np.mean(train_loss)
        ret_train_loss.append(train_epoch_loss)
        # print(f'epoch {epoch}| training loss: {train_epoch_loss}', end='\r')

        # Validation phase
        net.eval()
        valid_loss = []
        with torch.no_grad():
            pbar = tqdm(enumerate(valid_loader), leave=False)
            for i, (img, label) in pbar:
                batch_shape = img.shape[1]
                img = img.view(-1, 224, 224)
                img = img.unsqueeze(1).repeat(1, 3, 1, 1)
                label = torch.repeat_interleave(label, batch_shape, dim=0)
                img = img.to(device)
                label = label.to(device)
                y_pred = net(img)
                loss = loss_function(y_pred, label)
                valid_loss.append(loss.item())
                # print(f'{i + 1}/{len(valid_loader)}| current validation loss: {valid_loss[-1]}', end='\r')
                pbar.set_description(f'current valid loss: {valid_loss[-1]:.4f}')

        epoch_vloss = np.mean(valid_loss)

        # print(f"training loss: {train_epoch_loss:.4f} | validation loss: {epoch_vloss:.4f}")
        ret_valid_loss.append(epoch_vloss)

    return train_epoch_loss, epoch_vloss

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
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
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

    pbar = tqdm(training_sets)
    for i in pbar:
        tqdm.write(f"Fold {i+1}")
        pbar.set_description(f"Fold {i+1}")
        training_set = dataloaders[i]
        inner_pbar = tqdm(range(20), leave = False)
        for epoch in inner_pbar:
            inner_pbar.set_description(f"Epoch {epoch+1}")
            train_loss, valid_loss = train_net(model, 1, training_set, valid_set, optimizer, loss_function, device)
            tqdm.write(f"training loss: {train_loss:.4f} | validation loss: {valid_loss:.4f}")
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                model_filename = f'Epoch_{epoch}_VLoss_{valid_loss:.4f}.pth'
                model_path = os.path.join(check_dir, model_filename)
                torch.save(model.state_dict(), model_path)
            overall_train_loss.append(train_loss)
            overall_valid_loss.append(valid_loss)
            with open("transformer_training_lossVb.txt", "a") as file:
                file.write(str(train_loss) + '\n')

            with open("transformer_validation_lossVb.txt", "a") as file:
                file.write(str(valid_loss) + '\n')
        

    #np.savetxt('training_loss.txt', overall_train_loss)

    #np.savetxt('valid_loss.txt', overall_valid_loss)

if __name__ == '__main__':
    main()
