import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import random
import numpy as np

"""
get label from address of 
the label of cat is 0
the label of dog is 1
"""
def getlabel():
    img_list,label_list=[],[]
    base_address="E:\\z\\kaggle\\train\\train"
    addlist=os.listdir(base_address)
    for ad in addlist:
        if ad.split('.')[0]=='dog':
            label_list.append(1)
        else:
            label_list.append(0)
        imag_add=os.path.join(base_address,ad)
        img_list.append(imag_add)
    return img_list,label_list

"""
get train set and test set index
"""
def gettrainset(n_samples,train_radio=0.8,test_radio=0.1,val_radio=0.1):
    np.random.seed(100)
    idx=np.arange(n_samples)
    np.random.shuffle(idx)
    split_point1=int(n_samples*train_radio)
    split_point2=int(n_samples*(train_radio+test_radio))
    train_index=idx[:split_point1]
    test_index=idx[split_point1:split_point2]
    val_index=idx[split_point2:]
    return train_index,test_index,val_index

"""
construct the torch set
"""
class myimageset(Dataset):
    def __init__(self,data,label):
        self.data=[]
        self.label=[]
        transform=transforms.Compose([transforms.Resize((150,150)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                     ])
        for i in range(len(data)):
            img = Image.open(data[i])
            img = img.convert('RGB')
            self.data.append(transform(img))
            img.close()
            self.label.append(label[i])
    def __getitem__(self,index):
        return self.data[index],torch.torch.as_tensor(float(self.label[index]))
    def __len__ (self):
        return len(self.data)

"""
test code:
train_image,train_label,test_image,test_label,val_image,val_label=load_image()
t=myimageset(data=val_image,label=val_label)
train_loader=DataLoader(t,batch_size=3,shuffle=False,drop_last=True)
for b,(x,y) in enumerate(train_loader):
    print(y)
    if b==5:
        break
"""
