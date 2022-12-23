import argparse
import os
import time
import torch.nn as nn
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from DerainDataset import *
from networks import *
from SSIM import SSIM
from utils import *
from torch.optim.lr_scheduler import MultiStepLR

#便于脚本运行时改变参数
parser = argparse.ArgumentParser(description="MPNet_Test")
parser.add_argument("--logdir", type=str, default="/root/autodl-tmp/MPNet/logs/Rain100H/MPNet_second", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/MPNet1/datasets/test/Rain100H/Rain100H/rainy", help='path to training data')
parser.add_argument("--save_path", type=str, default="/root/autodl-tmp/MPNet1/results/PReNet", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=7, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    os.makedirs(opt.save_path, exist_ok=True)

   
    # Build model加载模型
    print('Loading model ...\n')
    model = MPNet(opt.recurrent_iter, opt.use_GPU)
    print_network(model)
    #是否GPU
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))
    model.eval()
# loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()#损失用SSIM
    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)

            # input image读入图片
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y))

            if opt.use_GPU:
                y = y.cuda()

            with torch.no_grad(): 
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                
                
                out, _ = model(y)
                out = torch.clamp(out, 0., 1.)
               
                
                if opt.use_GPU:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name, ': ', dur_time)

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            
            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)

            count += 1

    print('Avg. time:', time_test/count)


if __name__ == "__main__":
    main()





