
import torch
import torch.nn as nn


class convBlock(nn.Module):
    def __init__(self,inplace,outplace,kernel_size=3,padding=1):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(inplace,outplace,kernel_size=kernel_size,padding=padding,bias=False)
        self.bn1 = nn.BatchNorm3d(outplace)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class VGG83D(nn.Module):
    def __init__(self,inplace):
        super().__init__()
        
        ly = [64,128,256,512]
        
        self.ly = ly
        
        self.maxp = nn.MaxPool3d(2)
        
        self.conv11 = convBlock(inplace,ly[0])
        self.conv12 = convBlock(ly[0],ly[0])
        
        self.conv21 = convBlock(ly[0],ly[1])
        self.conv22 = convBlock(ly[1],ly[1])
        
        self.conv31 = convBlock(ly[1],ly[2])
        self.conv32 = convBlock(ly[2],ly[2])
        
        self.conv41 = convBlock(ly[2],ly[3])
        self.conv42 = convBlock(ly[3],ly[3])
        
    def forward(self,x):

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxp(x)
 
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxp(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.maxp(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.maxp(x)

        return x

class VGG113D(nn.Module):
    def __init__(self,inplace):
        super().__init__()
        
        #ly = [16,32,64,64,128,128,128,128]
        ly = [16,32,64,64,128,128,256,256]
        
        self.ly = ly
        
        self.maxp1 = nn.MaxPool3d(2)
        self.maxp2 = nn.MaxPool3d(2)
        self.maxp3 = nn.MaxPool3d(2)
        self.maxp4 = nn.MaxPool3d(2)
        self.maxp5 = nn.MaxPool3d(2)
        
        self.conv11 = convBlock(inplace,ly[0])

        self.conv21 = convBlock(ly[0],ly[1])
        
        self.conv31 = convBlock(ly[1],ly[2])
        self.conv32 = convBlock(ly[2],ly[3])
        
        self.conv41 = convBlock(ly[3],ly[4])
        self.conv42 = convBlock(ly[4],ly[5])
        
        self.conv51 = convBlock(ly[5],ly[6])
        self.conv52 = convBlock(ly[6],ly[7])
        
    def forward(self,x):

        x = self.conv11(x)
        x = self.maxp1(x)

        x = self.conv21(x)
        x = self.maxp2(x)
 
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.maxp3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.maxp4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.maxp5(x)

        return x

class VGG16(nn.Module):
    def __init__(self,inplace):
        super().__init__()
        
        ly = [64,128,256,512,512]
        
        self.ly = ly
        
        self.maxp = nn.MaxPool2d(2)
        
        self.conv11 = convBlock(inplace,ly[0])
        self.conv12 = convBlock(ly[0],ly[0])
        
        self.conv21 = convBlock(ly[0],ly[1])
        self.conv22 = convBlock(ly[1],ly[1])
        
        self.conv31 = convBlock(ly[1],ly[2])
        self.conv32 = convBlock(ly[2],ly[2])
        self.conv33 = convBlock(ly[2],ly[2])
        
        self.conv41 = convBlock(ly[2],ly[3])
        self.conv42 = convBlock(ly[3],ly[3])
        self.conv43 = convBlock(ly[3],ly[3])
        
        self.conv51 = convBlock(ly[3],ly[3])
        self.conv52 = convBlock(ly[3],ly[3])
        self.conv53 = convBlock(ly[3],ly[3])
        
    def forward(self,x):

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxp(x)
 
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxp(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.maxp(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.maxp(x)
        
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)

        return x    
