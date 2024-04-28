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

    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activation_gradient(self):
        return self.gradients
    
    def get_activations(self, x1):
        x1 = self.feature_extractor(x1)
        return x1

    def forward(self, x1):
        x1 = self.feature_extractor(x1)
        h = x1.register_hook(self.activations_hook)
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
    def __init__(self, root_dir, grad_dir, transforms = None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.cn_dir = os.path.join(self.root_dir, "MNI152_affine_WB_iso1mm/CN")
        self.scz_dir = os.path.join(self.root_dir, "MNI152_affine_WB_iso1mm/schiz")
        self.samples, self.labels = self._load_samples()
        self.grad_dir = grad_dir
        self.grad_cn_dir = os.path.join(self.grad_dir, "MNI152_affine_WB_iso1mm/CN")
        self.grad_scz_dir = os.path.join(self.grad_dir, "MNI152_affine_WB_iso1mm/CN")

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
        img_data = np.float32(img.get_fdata())
        if self.transforms:
            img_tensor = self.transforms(img_data)
        else:
            img_tensor = torch.from_numpy(img_data).unsqueeze(0)
        return img_tensor, label
    
    def getpath(self, idx):
        # should return: image_tensor, grad path
        label = self.labels[idx]
        file_name = os.path.splitext(self.samples[idx])[0] + '_activation.nii.gz'
        if label == 0:
            image_path = os.path.join(self.cn_dir, self.samples[idx])
            grad_path = os.path.join(self.grad_cn_dir, file_name)
        else:
            image_path = os.path.join(self.scz_dir, self.samples[idx])
            grad_path = os.path.join(self.grad_scz_dir, file_name)

        img = nib.load(image_path)
        img_data = np.float32(img.get_fdata())
        img_tensor = self.transforms(img_data)
        return img_tensor, image_path, grad_path


def get_feature_map(model, input):
    pred = model(input)

    pred[:, 1].backward()

    gradients = model.get_activation_gradient()

    pooled_gradients = torch.abs(torch.mean(gradients, dim = [0, 2, 3, 4]))

    activations = model.get_activations(input).detach()

    for idx in range(128):
        activations[:, idx, :, :, :] *= pooled_gradients[idx]
    heatmap = torch.mean(activations, 1).squeeze().cpu()

    heatmap = np.maximum(heatmap, 0)

    heatmap /= torch.max(heatmap)

    resampled_heatmap = zoom(heatmap, tuple(np.array([192, 192, 192]) / np.array(heatmap.shape))) #cubic spline
    return resampled_heatmap

def center_crop_or_pad(input_scan, desired_dimension):
    input_dimension = input_scan.shape
    #print('Input dimension: ', input_dimension, '\ndesired dimension: ', desired_dimension)

    x_lowerbound_target = int(np.floor((desired_dimension[0] - input_dimension[0]) / 2)) if desired_dimension[0] >= input_dimension[0] else 0
    y_lowerbound_target = int(np.floor((desired_dimension[1] - input_dimension[1]) / 2)) if desired_dimension[1] >= input_dimension[1] else 0
    z_lowerbound_target = int(np.floor((desired_dimension[2] - input_dimension[2]) / 2)) if desired_dimension[2] >= input_dimension[2] else 0
    x_upperbound_target = x_lowerbound_target + input_dimension[0] if desired_dimension[0] >= input_dimension[0] else None
    y_upperbound_target = y_lowerbound_target + input_dimension[1] if desired_dimension[1] >= input_dimension[1] else None
    z_upperbound_target = z_lowerbound_target + input_dimension[2] if desired_dimension[2] >= input_dimension[2] else None

    x_lowerbound_input = 0 if desired_dimension[0] >= input_dimension[0] else int(np.floor((input_dimension[0] - desired_dimension[0]) / 2))
    y_lowerbound_input = 0 if desired_dimension[1] >= input_dimension[1] else int(np.floor((input_dimension[1] - desired_dimension[1]) / 2))
    z_lowerbound_input = 0 if desired_dimension[2] >= input_dimension[2] else int(np.floor((input_dimension[2] - desired_dimension[2]) / 2))
    x_upperbound_input = None if desired_dimension[0] >= input_dimension[0] else x_lowerbound_input + desired_dimension[0]
    y_upperbound_input = None if desired_dimension[1] >= input_dimension[1] else y_lowerbound_input + desired_dimension[1]
    z_upperbound_input = None if desired_dimension[2] >= input_dimension[2] else z_lowerbound_input + desired_dimension[2]


    output_scan = np.zeros(desired_dimension).astype(np.float32)

    output_scan[x_lowerbound_target : x_upperbound_target, \
                y_lowerbound_target : y_upperbound_target, \
                z_lowerbound_target : z_upperbound_target] = \
    input_scan[x_lowerbound_input: x_upperbound_input, \
               y_lowerbound_input: y_upperbound_input, \
               z_lowerbound_input: z_upperbound_input]

    return output_scan

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = "/media/youzhi/SSD/bme_project/data"
    root_grad_dir = "/media/youzhi/SSD/bme_project/activations"

    folds_dir = [dir for dir in os.listdir(root_dir) if dir.startswith("fold")]
    grads_dir = [os.path.join(root_grad_dir, dir) for dir in folds_dir]
    folds_dir = [os.path.join(root_dir, dir) for dir in folds_dir]

    folds_dir = natsorted(folds_dir)

    grads_dir = natsorted(grads_dir)
    
    model = vgg11_bn()
    state_dict = torch.load('/media/youzhi/SSD/bme_project/checkpoints/Epoch_28_VLoss_0.0211.pth')
    model.load_state_dict(state_dict)

    datasets = []
    for i in range(len(folds_dir)):
        fold_dir = folds_dir[i]
        grad_dir = grads_dir[i]
        dataset = CustomDataset(fold_dir, grad_dir, downsize_transform) 
        datasets.append(dataset)

    for j in range(len(datasets)):
        #   
        dataset = datasets[j]
        print(f'fold {j}')
        for i in tqdm(range(len(dataset))):
            # get the image_path, grad_path from dataset, 
            # create the grad_path if it doesn't exist
            # run the image through the VGG model and get the gradCAMs
            # save the gradCAMS in the specified grad_path
            img, image_path, grad_path = dataset.getpath(i)
            heatmap = get_feature_map(model, img.unsqueeze(0))
            current_T1 = nib.load(image_path)
            resized_heatmap = center_crop_or_pad(heatmap, current_T1.shape)
            new_heatmap_nii = nib.Nifti1Image(resized_heatmap, current_T1.affine, current_T1.header)
            os.makedirs(os.path.dirname(grad_path), exist_ok=True)
            nib.save(new_heatmap_nii, grad_path)

if __name__ == '__main__':
    main()
            
