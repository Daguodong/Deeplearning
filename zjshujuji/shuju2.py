import  os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import torchvision
from torchvision import transforms as t
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = t.Compose([
    t.Resize(28),
    t.CenterCrop(28),
    t.ToTensor(),
    t.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5,])

])

dataset=ImageFolder('D:/pycharm项目/制作自己的数据集/二类小文件夹/',transform=transform)   #ImageFolder统一返回RGB图，即使输入的是灰度图
print(dataset.class_to_idx)   #  打印{'一': 0, '零': 1}
print('*'*20)


print(dataset[0][0].size())   #  torch.Size([3, 28, 28])
print('*'*20)


to_img=t.ToPILImage()
print(to_img(dataset[0][0]*0.5+0.5))
print('*'*20)


for imgs,label in dataset:
    print(imgs.size(),label)
#print(dataset.imgs)
print('*'*20)



dataset=DataLoader(dataset,batch_size=1)  #与20--22对比
dataiter=iter(dataset)
imgs ,labels=next(dataiter)
print(imgs.size(),)    # torch.Size([1, 3, 28, 28])说明原始的灰度图经过ImageFolder操作变为RGB输出

