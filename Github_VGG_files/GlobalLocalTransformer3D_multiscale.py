"""
This is the code for global-local transformer for brain age estimation

@email: heshengxgd@gmail.com

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import vgg3D_for_transformer as vnet

import numpy as np

class GlobalAttention(nn.Module):
    def __init__(self, 
                 transformer_num_heads=8,
                 hidden_size=512,
                 transformer_dropout_rate=0.0):
        super().__init__()
        
        self.num_attention_heads = transformer_num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self,locx,glox):
        locx_query_mix = self.query(locx)
        glox_key_mix = self.key(glox)
        glox_value_mix = self.value(glox)
        
        query_layer = self.transpose_for_scores(locx_query_mix)
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output

class convBlock(nn.Module):
    def __init__(self,inplace,outplace,kernel_size=3,padding=1):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv3d(inplace,outplace,kernel_size=kernel_size,padding=padding,bias=False)
        self.bn1 = nn.BatchNorm3d(outplace)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
    
class Feedforward(nn.Module):
    def __init__(self,inplace,outplace):
        super().__init__()
        
        self.conv1 = convBlock(inplace,outplace,kernel_size=1,padding=0)
        self.conv2 = convBlock(outplace,outplace,kernel_size=1,padding=0)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class GlobalLocalBrainAge(nn.Module):
    def __init__(self,inplace,
                 patch_size=64,
                 step=-1,
                 nblock=6,
                 drop_rate=0.5,
                 backbone='vgg163D'):
        """
        Parameter:
            @patch_size: the patch size of the local pathway
            @step: the step size of the sliding window of the local patches
            @nblock: the number of blocks for the Global-Local Transformer
            @Drop_rate: dropout rate
            @backbone: the backbone of extract the features
        """
        
        super().__init__()
        
        self.patch_size = patch_size
        self.step = step
        self.nblock = nblock
        
        if self.step <= 0:
            self.step = int(patch_size//2)
            
        if backbone == 'vgg8':
            self.global_feat = vnet.VGG8(inplace)
            self.local_feat = vnet.VGG8(inplace)
            hidden_size = 512
        elif backbone == 'vgg16':
            self.global_feat = vnet.VGG16(inplace)
            self.local_feat = vnet.VGG16(inplace)
            hidden_size = 512
        elif backbone == 'vgg113D':
            self.global_feat = vnet.VGG113D(inplace)
            self.local_feat = vnet.VGG113D(inplace)
            hidden_size = 256
        elif backbone == 'vgg163D':
            self.global_feat = vnet.VGG163D(inplace)
            self.local_feat = vnet.VGG163D(inplace)
            hidden_size = 256
        elif backbone == 'vgg8conv3d':
            self.global_feat = vnet.vgg8conv3d(inplace)
            self.local_feat = vnet.vgg8conv3d(inplace)
            hidden_size = 256
        elif backbone == 'VGG6conv3D':
            self.global_feat = vnet.VGG6conv3D(inplace)
            self.local_feat = vnet.VGG6conv3D(inplace)
            hidden_size = 256
        else:
            raise ValueError('% model does not supported!'%backbone)
    
        self.attnlist = nn.ModuleList()
        self.fftlist = nn.ModuleList()
        
        for n in range(nblock):
            atten = GlobalAttention(
                    transformer_num_heads=8,
                    hidden_size=hidden_size,
                    transformer_dropout_rate=drop_rate)
            self.attnlist.append(atten)
            
            #fft = Feedforward(inplace=hidden_size*2,outplace=hidden_size)
            fft = Feedforward(inplace=hidden_size,
                              outplace=hidden_size)
            self.fftlist.append(fft)
            
        self.avg2d = nn.AdaptiveAvgPool2d(1)
        self.avg3d = nn.AdaptiveAvgPool3d(1)
        out_hidden_size = hidden_size

       # print('hidden', out_hidden_size) #256
            
        self.gloout1 = nn.Linear(out_hidden_size, 2) # canbe changed to a MLP
       # self.gloout2 = nn.Linear(128, 1)
        self.locout1 = nn.Linear(out_hidden_size, 2) # canbe changed to a MLP
        
        self.voting = nn.Conv1d(3,1,1,stride=1,padding=0)
       # self.finalout = nn.Linear(6, 1)
       # self.locout2 = nn.Linear(128, 1)
        
    def forward(self,xinput):
        #@+print(xinput.size())
        _,_,H,W,D =xinput.size()
        outlist = []
        
        xglo = self.global_feat(xinput) # vgg encoder
        #print(xglo.size())
        xgfeat = torch.flatten(self.avg3d(xglo),1) # global features
        #print('xgfeat',xgfeat.size())
            
        glo = self.gloout1(xgfeat) # single layer MLP
       # glo = self.gloout2(glo)

        outlist=[glo]
        #print(glo.shape)
        
        

        
        B2,C2,H2,W2,D1 = xglo.size()
        xglot = xglo.view(B2,C2,H2*W2*D1)
        #print(xglot,'xglot.shape')
        xglot = xglot.permute(0,2,1)
        #print(xglot,'xglot.shape')
        
        # local path



                # MULTI SCALE 

        largest_power_of_two = 2

 #       patch_total_num = len(range(0,D-self.patch_size,self.step))*len(range(0,H-self.patch_size,self.step))*len(range(0,W-self.patch_size,self.step))

        
        count = 0

        patch_total_num = largest_power_of_two

        if patch_total_num==0:
            position = []
        else:
            position = np.zeros((patch_total_num,3),dtype=np.int64)
        #print('position.shape',position.shape)

 #       for z in range(0,D-self.patch_size,self.step): 
 #           for y in range(0,H-self.patch_size,self.step):
 #               for x in range(0,W-self.patch_size,self.step):
 #                   locx = xinput[:,:,y:y+self.patch_size,x:x+self.patch_size,z:z+self.patch_size]
 #                   xloc = self.local_feat(locx) # vgg encoder of patches
 #                   #print('xloc.shape',xloc.shape)
 #                   position[count,0] = x+np.rint(self.patch_size/2)
 #                   position[count,1] = y+np.rint(self.patch_size/2)
 #                   position[count,2] = z+np.rint(self.patch_size/2)
 #                   count = count + 1
        from scipy.ndimage import zoom
        device = torch.device("cuda:0")

        # a = torch.empty([6])

        BS = xinput.shape[0]


        for i in range(1, largest_power_of_two + 1):
            #print(len(outlist))
            locx = xinput#.detach().cpu().numpy()
            locx = locx.detach().cpu().numpy()
            #print(locx.shape)
            DS = 1 / 2**i
            US = 2**i
            for batch in range(BS):
                temp1 = locx[batch, 0, :,:,:]
              #  temp2 = locx[batch, 1, :,:,:]
               # print('111,',locx[batch, 0, :,:,:].shape)
                #print('pre',temp.shape)
                temp1 = zoom(temp1, (DS, DS, DS), mode ='nearest')  #1/2, 1/4, 1/8, 1/16, 1/32
               # temp2 = zoom(temp2, (DS, DS, DS), mode ='nearest')
               # print('2222,',locx[batch, 0, :,:,:].shape)
                #print('mid',temp.shape)
                temp1 = zoom(temp1, (US, US, US), mode ='nearest')
               # temp2 = zoom(temp2, (US, US, US), mode ='nearest')
               # print('33333,', locx[batch, 0, :,:,:].shape )
                #print('post',temp.shape)
                locx[batch, 0, :,:,:] = temp1
              #  locx[batch, 1, :,:,:] = temp2

            locx = torch.from_numpy(locx)
            locx = locx.to(device)
            xloc = self.local_feat(locx) # vgg encoder of patches
            

            for n in range(self.nblock):
                B1,C1,H1,W1,D1 = xloc.size()
                xloct = xloc.view(B1,C1,H1*W1*D1)
                xloct = xloct.permute(0,2,1)
            
                tmp = self.attnlist[n](xloct,xglot) # global-local attention
                tmp = tmp.permute(0,2,1)
                tmp = tmp.view(B1,C1,H1,W1,D1)
                #tmp_C = torch.cat([tmp,xloc],1) # after C logo
                tmp1 = tmp + xloc # after C logo
                #print('tmp_C.shape',tmp_C.shape)
                tmp2 = self.fftlist[n](tmp1)
                #print('tmp_FF.shape',tmp_FF.shape)
                xloc = tmp1 + tmp2
            xloc = torch.flatten(self.avg3d(xloc),1)
            
            out = self.locout1(xloc)
            #out = out.detach().cpu()
            #out = self.locout2(out)
            outlist.append(out) # no positioning information ...
           # print(outlist[0].shape)
           #print(len(outlist))


        # BS = xinput.shape[0]

        
        # for i in range(1, largest_power_of_two + 1):
        #     #print(len(outlist))
        #     locx = xinput#.detach().cpu().numpy()
        #     #locx = locx.detach().cpu().numpy()
        #     #print(locx.shape)
        #     DS = 1 / 2**i
        #     US = 2**i
        #     for batch in range(BS):
        #         temp = locx[batch, 0, :,:,:].clone()
        #        # print('111,',locx[batch, 0, :,:,:].shape)
        #         #print('pre',temp.shape)
        #        # temp = F.interpolate(temp, scale_factor = DS, mode ='nearest').clone()  #1/2, 1/4, 1/8, 1/16, 1/32
        #        # print('2222,',locx[batch, 0, :,:,:].shape)
        #         #print('mid',temp.shape)
        #        # temp = F.interpolate(temp, scale_factor = US, mode ='nearest').clone()
        #        # print('33333,', locx[batch, 0, :,:,:].shape )
        #         #print('post',temp.shape)
        #         locx[batch, 0, :,:,:] = temp.clone()

        #     # locx = torch.from_numpy(locx)
        #     locx = locx.to(device)
        #     xloc = self.local_feat(locx) # vgg encoder of patches


            # for n in range(self.nblock):
            #     B1,C1,H1,W1,D1 = xloc.size()
            #     xloct = xloc.view(B1,C1,H1*W1*D1)
            #     xloct = xloct.permute(0,2,1)
            
            #     tmp = self.attnlist[n](xloct,xglot) # global-local attention
            #     tmp = tmp.permute(0,2,1)
            #     tmp = tmp.view(B1,C1,H1,W1,D1)
            #     #tmp_C = torch.cat([tmp,xloc],1) # after C logo
            #     tmp1 = tmp + xloc # after C logo
            #     #print('tmp_C.shape',tmp_C.shape)
            #     tmp2 = self.fftlist[n](tmp1)
            #     #print('tmp_FF.shape',tmp_FF.shape)
            #     xloc = tmp1 + tmp2
            # xloc = torch.flatten(self.avg3d(xloc),1)
            
            # out = self.locout1(xloc)
            # #out = out.detach().cpu()
            # #out = self.locout2(out)
            # outlist.append(out) # no positioning information ...
        # output = []
        # print('len', len(outlist))

        # for i in range(len(outlist)):
        #     print(outlist[i].shape)

        #     a0 = outlist[i][0]
        #     a1 = outlist[i][1]
        #     a2 = outlist[i][2]
        #     a3 = outlist[i][3]
        #     a4 = outlist[i][4]
        #     a5 = outlist[i][5]
        #     a = torch.cat((a0,a1,a2,a3,a4,a5))
        #     #a = a.to(device)
        #     output.append(self.finalout(a))
           # outtensor = torch.Tensor(outlist)
#outlist = torch.cat((outlist, out))
            #outlist = torch.permute(outlist, (1,0))
        #outlist = torch.squeeze(outlist)
       # for i in range(6):
       #     a[i] = outlist[i]
      
        #output = self.finalout(a)
        #print(len(outlist))

        #for i in range(len(outlist)):
          #  outlist[i] = outlist[i]#.unsqueeze(axis=1)
          #  print(outlist[i].shape)

        # Prediction_list_eachlength = [len(a) for a in outlist[0][:]]
        #     #print('Prediction_list_eachlength',len(Prediction_list_eachlength))

        # output = torch.zeros(BS,len(Prediction_list_eachlength))
        # for patch_indx in range(0,output.shape[1]):
        #     Prediction_patch_allpatch = torch.squeeze(outlist[0][patch_indx])
        #       #print('Prediction_patch_allpatch.shape',Prediction_patch_allpatch.shape)
        #     output[:,patch_indx] = Prediction_patch_allpatch
        output = torch.stack((outlist[0].clone(),outlist[1].clone(), outlist[2].clone()), dim = 1)

      #  print(output.shape)

        output = output.to(device)
                
        Prediction = self.voting(output)
        #print(Prediction.shape)
        Prediction = Prediction.squeeze()
                #print(Prediction.shape)
      #  print(Prediction.shape)

        return Prediction, position

class Voting(torch.nn.Module):
    def __init__(self):
        super(Voting, self).__init__()
        self.fc = nn.Linear(6,1)

    def forward(self, x):
        output = self.fc(x)
        return output
        
if __name__ == '__main__':
    x1 = torch.rand(1,5,130,170)
    
    mod = GlobalLocalBrainAge(5,
                        patch_size=64,
                        step=32,
                        nblock=6,
                        backbone='vgg8')
    zlist = mod(x1)
    for z in zlist:
        print(z.shape)
    print('number is:',len(zlist))
   
        
