from train import val,plotter
import torch
import torch.nn as nn
from midas.midas_net_custom import MidasNet_Fusion\

model = MidasNet_Fusion()
model = model.cuda()
savePath = '/content/drive/MyDrive/DDP/HDR/MiDasSceneFlow_2encoder_withNopretrained_model(20+35)_10.pt'
model = model.cuda()
model.load_state_dict(torch.load(savePath))
model.eval()
criterion = nn.MSELoss()

from utils import dataloader
from dataloader import myImageLoader
datapath = '/content/drive/MyDrive/DDP/HDR/SceneFlow'
train_left_img, train_right_img, train_left_disp = dataloader(datapath)
total = int(len(train_left_disp))
trainCount = int(0.9*total)


test_left_img, test_right_img, test_left_disp = train_left_img[trainCount:], train_right_img[trainCount:], train_left_disp[trainCount:]

TestImgLoader = torch.utils.data.DataLoader(
    myImageLoader(test_left_img, test_right_img, test_left_disp, False),
    batch_size=1, shuffle=False, num_workers=2, drop_last=False)

def evaluate():
    val(0,TestImgLoader,criterion)
    plotter(0,TestImgLoader)

if __name__ == "__main__":
    evaluate()