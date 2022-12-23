#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os

class MPNet(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(MPNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )
        
        self.upsample = F.upsample_nearest
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0) 
        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refine3= nn.Conv2d(32+4, 3, kernel_size=3,stride=1,padding=1)
        self.tanh=nn.Tanh()
  

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()    
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
         
            for ii in range(2):
                if ii==1: 
                    x1 = torch.cat((x, h), 1)
                    i = self.conv_i(x1)
          
                    
                    f = self.conv_f(x1)
                    g = self.conv_g(x1)
                    o = self.conv_o(x1)
                    c = f * c + i * g
                    h1 = o * torch.tanh(c)
                else:
                    x2 = torch.cat((x, h), 1) 
                    i = self.conv_i(x2)
                    f = self.conv_f(x2)
                    g = self.conv_g(x2)
                    o = self.conv_o(x2)
                    c = f * c + i * g
                    h2 = o * torch.tanh(c)
            h=(h1+h2)/2
            x = h
            resx = x 
            
            
            shape_out = x.data.size()
            shape_out = shape_out[2:4]
         
            x_new = x[:,:20,:,:]
            x101 = F.avg_pool2d(x_new, 32) 
            x102 = F.avg_pool2d(x_new, 16) 
            x103 = F.avg_pool2d(x_new, 8)
            x104 = F.avg_pool2d(x_new, 4) 
            x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
            x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
            x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
            x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)
            dehaze = torch.cat((x1010, x1020, x1030, x1040, x), 1)
            residual = self.tanh(self.refine3(dehaze))
            y = residual
          

            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)
            
            
            x = x-y 
            

            x = x + input
            x_list.append(x)

        return x, x_list

class MPNet_SLSTM(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(MPNet_SLSTM, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )
        self.upsample = F.upsample_neares
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  
        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refine3= nn.Conv2d(32+4, 3, kernel_size=3,stride=1,padding=1)
        self.tanh=nn.Tanh()

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x1 = x
            x = torch.cat((input, x), 1)
            x = self.conv0(x)
            
            
            
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x

            shape_out = x.data.size()
            # print(shape_out)
            shape_out = shape_out[2:4]
        
            x_new = x[:,:20,:,:]
            x101 = F.avg_pool2d(x_new, 32) 
            x102 = F.avg_pool2d(x_new, 16) 
            x103 = F.avg_pool2d(x_new, 8) 
            x104 = F.avg_pool2d(x_new, 4)  
            x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
            x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
            x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
            x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)
            dehaze = torch.cat((x1010, x1020, x1030, x1040, x), 1)
            residual = self.tanh(self.refine3(dehaze))
            y = residual
         
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

           
            x = x-y 
         
            
            x_list.append(x)

        return x, x_list



class MPNet_x(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(MPNet_x, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )
        self.upsample = F.upsample_nearest
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  
        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refine3= nn.Conv2d(32+4, 3, kernel_size=3,stride=1,padding=1)
        self.tanh=nn.Tanh()
    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            #x = torch.cat((input, x), 1)
            x = self.conv0(x)

            for ii in range(2):
                if ii==1: 
                    x1 = torch.cat((x, h), 1)
                    i = self.conv_i(x1)
                    
                    f = self.conv_f(x1)
                    g = self.conv_g(x1)
                    o = self.conv_o(x1)
                    c = f * c + i * g
                    h1 = o * torch.tanh(c)
                else:
                    x2 = torch.cat((x, h), 1) 
                    i = self.conv_i(x2)
                    f = self.conv_f(x2)
                    g = self.conv_g(x2)
                    o = self.conv_o(x2)
                    c = f * c + i * g
                    h2 = o * torch.tanh(c)
            h=(h1+h2)/2
            x = h
            resx = x 
            
            shape_out = x.data.size()
            # print(shape_out)
            shape_out = shape_out[2:4]
            x_new = x[:,:20,:,:]
            x101 = F.avg_pool2d(x_new, 32) 
            x102 = F.avg_pool2d(x_new, 16) 
            x103 = F.avg_pool2d(x_new, 8) 
            x104 = F.avg_pool2d(x_new, 4) 
            x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
            x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
            x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
            x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)
            dehaze = torch.cat((x1010, x1020, x1030, x1040, x), 1)
            residual = self.tanh(self.refine3(dehaze))
            y = residual

            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)

            x = x-y 

            x_list.append(x)

        return x, x_list


class MPNet_r(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(MPNet_r, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        #mask = Variable(torch.ones(batch_size, 3, row, col)).cuda()
        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            for j in range(5):
                resx = x
                x = F.relu(self.res_conv1(x) + resx)

            x = self.conv(x)
            x = input + x
            x_list.append(x)

        return x, x_list


