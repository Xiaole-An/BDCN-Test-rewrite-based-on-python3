import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
from PIL import Image
import torchvision.transforms as transforms
'''
把图片大小调整为正方形
'''

class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.imgdir = img_dir
        self.img_name = os.listdir(self.imgdir)
        self.img_transform = transforms.Compose([
            transforms.ToTensor()])

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        imgs = Image.open(os.path.join(self.imgdir, self.img_name[idx])).convert('RGB')
        imgs_new = resize(imgs)
        imgs_tensor = self.img_transform(imgs_new)
        name = self.img_name[idx]


        return imgs_tensor, name



def resize(img):
    new_size = [1024, 1024]
    im = img.resize(new_size)
    return im

#
# test_datasetdir = r'D:/Edge-detection/exposure_datasets/Test'
#
# exp = TestDataset(test_datasetdir)
# print(exp[0].shape)