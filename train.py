import torch,os
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataloader import myImageLoader
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def train(epoch, trainloader, optimizer, criterion,testloader):
    running_loss = 0.0
    count = 0 
    for i, data in enumerate(tqdm(trainloader), 0):
        count += 1
        # get the inputs
        l = data[0].float()
        r = data[1].float()
        depth = data[2].float()
        if torch.cuda.is_available():
            l,r, depth  = l.cuda(), r.cuda() ,depth.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        predicted_depth =torch.squeeze(model(l,r))
        
        loss = criterion(predicted_depth, depth)
        loss.backward()
        optimizer.step()
        # print statistics
        # if count%25 == 1:
        #   plotter(epoch,TestImgLoader)
        running_loss += loss.item()

    print('epoch %d training loss: %.3f' %
            (epoch + 1, running_loss / (len(trainloader))))

def val(epoch, trainloader, criterion):
    running_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(trainloader), 0):
        count += 1
        # get the inputs
        l = data[0].float()
        r = data[1].float()
        depth = data[2].float()
        if torch.cuda.is_available():
            l,r, depth  = l.cuda(), r.cuda() ,depth.cuda()

        # forward + backward + optimize
        predicted_depth =torch.squeeze(model(l,r))
        
        loss = criterion(predicted_depth, torch.squeeze(depth))
        # print statistics
        running_loss += loss.item()
    print('epoch %d validation loss: %.3f' %
            (epoch + 1, running_loss / (len(trainloader))))
    
def plotter(epoch, trainloader):
    running_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(trainloader), 0):
        # get the inputs
        count += 1
        l = data[0].float()
        r = data[1].float()
        depth = data[2].float()
        if torch.cuda.is_available():
            l,r  = l.cuda(), r.cuda() 
        # forward + backward + optimize
        sh = l.shape
        dummy = torch.from_numpy(np.ones(sh)).float().cuda()
        predicted_depth_stereo =torch.squeeze(model(l,r)).detach().cpu().numpy()
        predicted_depth_midas =torch.squeeze(model(l,l)).detach().cpu().numpy()
        image = torch.squeeze(l).cpu().numpy()
        depth = torch.squeeze(depth).numpy()
        image = np.transpose(image,(1,2,0))
        image = image.astype(np.uint8)
        fig,ax = plt.subplots(1,4,figsize= (10,10))
        predicted_depth_stereo = (predicted_depth_stereo-predicted_depth_stereo.min())/(predicted_depth_stereo.max()-predicted_depth_stereo.min())
        predicted_depth_midas = (predicted_depth_midas-predicted_depth_midas.min())/(predicted_depth_midas.max()-predicted_depth_midas.min())
        depth = (depth-depth.min())/(depth.max()-depth.min())
        our_ssim = np.round(ssim(predicted_depth_stereo,depth,data_range = 1),2)
        midas_ssim = np.round(ssim(predicted_depth_midas,depth,data_range = 1),2)
        ax[0].imshow(image)
        ax[0].set_title('image')
        # ax[0].imshow(image[:,:,3:],alpha = 0.5)
        ax[1].imshow(predicted_depth_stereo)
        ax[1].set_title('our stereo depth')
        ax[1].set_xlabel('ssim :' + str(our_ssim))
        ax[2].imshow(predicted_depth_midas)
        ax[2].set_title('stereoWithBothL')
        ax[2].set_xlabel('ssim :' + str(midas_ssim))
        ax[3].imshow(depth)
        ax[3].set_title('GT')
        plt.show()
        if count == 2:
          break


from midas.midas_net_custom import MidasNet_Fusion
from utils import dataloader
datapath = '/content/drive/MyDrive/DDP/HDR/SceneFlow'
train_left_img, train_right_img, train_left_disp = dataloader(datapath)
total = int(len(train_left_disp))
trainCount = int(0.9*total)



test_left_img, test_right_img, test_left_disp = train_left_img[trainCount:], train_right_img[trainCount:], train_left_disp[trainCount:]

train_left_img, train_right_img, train_left_disp = train_left_img[:trainCount], train_right_img[:trainCount], train_left_disp[:trainCount]
print('train: ',len(train_left_img), 'val: ',len(test_left_img))

TrainImgLoader = torch.utils.data.DataLoader(
    myImageLoader(train_left_img, train_right_img, train_left_disp, True),
    batch_size=8, shuffle=True, num_workers=2, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    myImageLoader(test_left_img, test_right_img, test_left_disp, False),
    batch_size=1, shuffle=False, num_workers=2, drop_last=False)



criterion = nn.MSELoss()
model = MidasNet_Fusion()
optimiser = optim.Adam(model.parameters())
model = model.cuda()
num_epochs = 20
savePath = '/content/drive/MyDrive/DDP/HDR/MiDasSceneFlow_2encoder_withNopretrained'
model = model.cuda()
for epoch in range(num_epochs):
  runningLoss = 0.0
  print(' training epoch: ',epoch +1)
  train(epoch,TrainImgLoader,optimiser,criterion,TestImgLoader)
  val(epoch,TestImgLoader,criterion)
  plotter(epoch,TestImgLoader)
  torch.save(model.state_dict(), f'{savePath}_model(20+35)'+str(epoch)+'.pt')
  torch.save(optimiser.state_dict(), f'{savePath}_optimiser(20+35)'+str(epoch)+'.pt')