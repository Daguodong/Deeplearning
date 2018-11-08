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
from torch.utils.data.sampler import WeightedRandomSampler

transform = t.Compose([
    t.Resize(28),
    t.CenterCrop(28),
    t.ToTensor(),
    t.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5,])

])

class test(data.Dataset):
    def __init__(self,root,transforms=None):
        imgs=os.listdir(root)
        self.imgs=[os.path.join(root,img)for img in imgs]
        self.transforms=transforms

    def __getitem__(self,index):
        img_path=self.imgs[index]
        label=0 if '0_'in img_path.split('/')[-1] else 1
        data=Image.open(img_path)
        if self.transforms:
            data=self.transforms(data)
        return data,label
    def __len__(self):
        return len(self.imgs)


dataset=test('D:/pycharm项目/制作自己的数据集/二类/',transforms=transform)
#img,label=dataset[0]

for img,label in dataset:
    print(img.size(),label)
print('*'*20)
print(dataset[0][0].size())
print('*'*20)
to_img=t.ToPILImage()
print(to_img(dataset[0][0]*0.5+0.5))  #打印一张图片
print('*'*20)


weights=[4if label==1 else 1 for data,label in dataset]
print(weights)  # 打印权重
print('*'*20)
sampler = WeightedRandomSampler(weights,num_samples=21,replacement=True) # num_samples=21会覆盖dataset实际大小
dataset=DataLoader(dataset,batch_size=3,sampler=sampler)

dataiter=iter(dataset)
imgs ,labels=next(dataiter)
print(imgs.size())          #打印一个batch_size即3张28x28的灰度图
print('*'*20)
for batch_datas,batch_labels in dataset:
    print(batch_datas.size(),batch_labels.size())   #打印每个batch_size大小和batch_size的label大小
print('*'*20)
for datas ,labels in dataset:
    print(labels.tolist())   #打印batch_size的label

