import bdcn
import torch
import numpy as np
from torch.utils.data import DataLoader
from data import TestDataset
import os
from torch.nn import functional as F
from torch.autograd import Variable
import cv2
import torchvision
from scipy import misc
from imageio import imwrite
from PIL import Image

def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))


model = bdcn.BDCN()
model.load_state_dict(torch.load(r'bdcn_pretrained_on_bsds500.pth'))
# model.cuda()
model.eval()
save_dir = r'result/'
test_root = r''
test_img = TestDataset(test_root)
testloader = DataLoader(
        test_img, batch_size=1, shuffle=False, num_workers=0)


#
# def saveImg(img, save_dir, type, name, Gray=False):
#     # fname, fext = name.split('.')
#     fname = name
#     imgPath = os.path.join(save_dir, "%s_%s" % (fname, type))
#     img_array = tensor2array(img.data[0])
#     image_pil = Image.fromarray(img_array)
#     if Gray:
#         image_pil.convert('L').save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
#     else:
#         image_pil.save(imgPath)
#
#
# def tensor2array(image_tensor, imtype=np.uint8, normalize=True):
#     if isinstance(image_tensor, list):
#         image_numpy = []
#         for i in range(len(image_tensor)):
#             image_numpy.append(tensor2array(image_tensor[i], imtype, normalize))
#         return image_numpy
#     image_numpy = image_tensor.cpu().float().numpy()
#     if normalize:
#         image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
#     else:
#         image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
#     image_numpy = np.clip(image_numpy, 0, 255)
#     if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
#         image_numpy = image_numpy[:, :, 0]
#     return image_numpy.astype(imtype)


for i, (data, name) in enumerate(testloader):
    # data = data.cuda()
    with torch.no_grad():
        data = Variable(data)
        out = model(data)
        fuse = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]

        str = ''.join(name)  # tuple -> list
        # print(name)
        # print(out.size())
        # torchvision.utils.save_image(255-fuse*255, os.path.join(save_dir, str))
        cv2.imwrite(os.path.join(save_dir, str), 255 - fuse * 255)
        # imgs = np.array(out)
        # print(imgs.size())
        # imgs
        # saveImg(out, './result', 'png', name)
        # print(out[0])
    # print(out)
    # fuse = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
    # if not os.path.exists(os.path.join(save_dir, 'fuse')):
    #     os.mkdir(os.path.join(save_dir, 'fuse'))
    # # cv2.imwrite(os.path.join(save_dir, 'fuse', '%s.png'%nm[i]), 255-fuse*255)
    # cv2.imwrite(os.path.join(save_dir), 255 - fuse * 255)


#
# for i in range(testloader.size):
#     image, name = testloader.load_data()
#     # gt = np.asarray(gt, np.float32)
#     # gt /= (gt.max() + 1e-8)
#     image = image.cuda()
#     _, res = model(image)
#     # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
#     res = res.sigmoid().data.cpu().numpy().squeeze()
#     res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#     misc.imsave(save_path+name, res)
