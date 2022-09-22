"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .base_model import BaseModel
from .blocks import FeatureFusionBlock, FeatureFusionBlock_custom, Interpolate, _make_encoder
class FusionNet(nn.Module):
  def __init__(self):
      super(FusionNet, self).__init__()
      features=64
      self.groups = 1
      self.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(False),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # nn.ReLU(),
        )
      self.model1 = MidasNet_small('/content/drive/MyDrive/Dual Degree Project/MS Project/Consistent DE from Multi-Exposure Data/ALG2-Fusion1_extendedMIDAS_MultiExposureStereo/MiDaS/model-small-70d6b9c8.pt')
      self.model2 = MidasNet_small('/content/drive/MyDrive/Dual Degree Project/MS Project/Consistent DE from Multi-Exposure Data/ALG2-Fusion1_extendedMIDAS_MultiExposureStereo/MiDaS/model-small-70d6b9c8.pt')
  def forward(self,x,y):
      x = self.model1(x)
      y = self.model2(y)
      x *=y
      out = self.output_conv(x)
      return torch.squeeze(out, dim=1)



class MidasNet_small(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=64, backbone="efficientnet_lite3", non_negative=True, exportable=True, channels_last=False, align_corners=True,
        blocks={'expand': True}):
        """Init.
        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet_small, self).__init__()

        use_pretrained = False if path else True
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
  
        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)

        
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )
        print("yes")
        if path:
            self.load(path)


    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)


        layer_1 = self.pretrained.layer1(x)
        # print(layer_1.shape)
        layer_2 = self.pretrained.layer2(layer_1)
        # print(layer_2.shape)
        layer_3 = self.pretrained.layer3(layer_2)
        # print(layer_3.shape)
        layer_4 = self.pretrained.layer4(layer_3)
        # print(layer_4.shape)
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # print(layer_1_rn.shape,layer_2_rn.shape,layer_3_rn.shape,layer_4_rn.shape)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        # print(path_4.shape)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        # print(path_3.shape)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        # print(path_2.shape)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # print(path_1.shape)
        
        out = self.scratch.output_conv(path_1)
        return out
        # return torch.squeeze(out, dim=1)



class MidasNet_Fusion(nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=64, backbone="efficientnet_lite3", non_negative=True, exportable=True, channels_last=False, align_corners=True,
        blocks={'expand': True}):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet_Fusion, self).__init__()

        use_pretrained = False if path else True
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
        self.pretrainedr, _ = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)

        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)

        
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # nn.ReLU(),
        )
        self.outputs = {}
       


    def forward(self, x,y):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)

        layer_1 = self.pretrained.layer1(x)
        self.outputs['layer_1'] = layer_1
        layer_2 = self.pretrained.layer2(layer_1)
        self.outputs['layer_2'] = layer_2
        layer_3 = self.pretrained.layer3(layer_2)
        self.outputs['layer_3'] = layer_3
        layer_4 = self.pretrained.layer4(layer_3)
        self.outputs['layer_4'] = layer_4

        layer_1R = self.pretrainedr.layer1(y)
        self.outputs['layer_1R'] = layer_1R
        layer_2R = self.pretrainedr.layer2(layer_1R)
        self.outputs['layer_2R'] = layer_2R
        layer_3R = self.pretrainedr.layer3(layer_2R)
        self.outputs['layer_3R'] = layer_3R
        layer_4R = self.pretrainedr.layer4(layer_3R)
        self.outputs['layer_4R'] = layer_4R

        layer_1_rn = self.scratch.layer1_rn(layer_1*layer_1R)
        layer_2_rn = self.scratch.layer2_rn(layer_2*layer_2R)
        layer_3_rn = self.scratch.layer3_rn(layer_3*layer_3R)
        layer_4_rn = self.scratch.layer4_rn(layer_4*layer_4R)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        return torch.squeeze(out, dim=1)
        return out


class MidasNet_Fusion_Ablation(nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=64, backbone="efficientnet_lite3", non_negative=True, exportable=True, channels_last=False, align_corners=True,
        blocks={'expand': True}):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet_Fusion_Ablation, self).__init__()

        use_pretrained = False if path else True
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
        self.pretrainedr, _ = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)

        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)

        
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # nn.ReLU(),
        )
        self.outputs = {}
       


    def forward(self, x,y):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)

        layer_1 = self.pretrained.layer1(x)
        self.outputs['layer_1'] = layer_1
        layer_2 = self.pretrained.layer2(layer_1)
        self.outputs['layer_2'] = layer_2
        layer_3 = self.pretrained.layer3(layer_2)
        self.outputs['layer_3'] = layer_3
        # layer_4 = self.pretrained.layer4(layer_3)
        # self.outputs['layer_4'] = layer_4

        layer_1R = self.pretrainedr.layer1(y)
        self.outputs['layer_1R'] = layer_1R
        layer_2R = self.pretrainedr.layer2(layer_1R)
        self.outputs['layer_2R'] = layer_2R
        layer_3R = self.pretrainedr.layer3(layer_2R)
        self.outputs['layer_3R'] = layer_3R
        # layer_4R = self.pretrainedr.layer4(layer_3R)
        # self.outputs['layer_4R'] = layer_4R

        layer_1_rn = self.scratch.layer1_rn(layer_1*layer_1R)
        layer_2_rn = self.scratch.layer2_rn(layer_2*layer_2R)
        layer_3_rn = self.scratch.layer3_rn(layer_3*layer_3R)
        # layer_4_rn = self.scratch.layer4_rn(layer_4*layer_4R)

        # path_4 = self.scratch.refinenet4(layer_4_rn)
        # path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_3 = self.scratch.refinenet3(layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        # path_2 = self.scratch.refinenet2(layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        return torch.squeeze(out, dim=1)
        return out




class Mono2StereoFusion(nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=64, backbone="efficientnet_lite3_fusion", non_negative=True, exportable=True, channels_last=False, align_corners=True,
        blocks={'expand': True}):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(Mono2StereoFusion, self).__init__()

        use_pretrained = False if path else True
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
        self.pretrainedr, _ = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)

        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)
        '''
        loading the base Model for monocular and freezing the weights of the model.
        '''
        self.baseModel = MidasNet_small('/content/drive/MyDrive/DDP/HDR/MiDaS/model-small-70d6b9c8.pt')
        for param in self.baseModel.parameters():
              param.requires_grad = False
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # nn.ReLU(),
        )
        self.outputs = {}
       


    def forward(self, x,y):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)
        x = self.baseModel(x)
        y = self.baseModel(y)
        layer_1 = self.pretrained.layer1(x)
        self.outputs['layer_1'] = layer_1
        layer_2 = self.pretrained.layer2(layer_1)
        self.outputs['layer_2'] = layer_2
        layer_3 = self.pretrained.layer3(layer_2)
        self.outputs['layer_3'] = layer_3
        layer_4 = self.pretrained.layer4(layer_3)
        self.outputs['layer_4'] = layer_4

        layer_1R = self.pretrainedr.layer1(y)
        self.outputs['layer_1R'] = layer_1R
        layer_2R = self.pretrainedr.layer2(layer_1R)
        self.outputs['layer_2R'] = layer_2R
        layer_3R = self.pretrainedr.layer3(layer_2R)
        self.outputs['layer_3R'] = layer_3R
        layer_4R = self.pretrainedr.layer4(layer_3R)
        self.outputs['layer_4R'] = layer_4R

        layer_1_rn = self.scratch.layer1_rn(layer_1*layer_1R)
        layer_2_rn = self.scratch.layer2_rn(layer_2*layer_2R)
        layer_3_rn = self.scratch.layer3_rn(layer_3*layer_3R)
        layer_4_rn = self.scratch.layer4_rn(layer_4*layer_4R)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        return torch.squeeze(out, dim=1)
        return out



class Mono2StereoFusion4blurred(nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=64, backbone="efficientnet_lite3_fusion", non_negative=True, exportable=True, channels_last=False, align_corners=True,
        blocks={'expand': True}):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(Mono2StereoFusion4blurred, self).__init__()

        use_pretrained = False if path else True
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=(15, 15), sigma=(1, 10))
        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
        self.pretrainedr, _ = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)

        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)
        '''
        loading the base Model for monocular and freezing the weights of the model.
        '''
        self.baseModel = MidasNet_small('/content/drive/MyDrive/DDP/HDR/MiDaS/model-small-70d6b9c8.pt')
        for param in self.baseModel.parameters():
              param.requires_grad = False
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # nn.ReLU(),
        )
        self.outputs = {}
       


    def forward(self, x,y):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)
        x = self.blur(self.baseModel(x))
        y = self.blur(self.baseModel(y))
        layer_1 = self.pretrained.layer1(x)
        self.outputs['layer_1'] = layer_1
        layer_2 = self.pretrained.layer2(layer_1)
        self.outputs['layer_2'] = layer_2
        layer_3 = self.pretrained.layer3(layer_2)
        self.outputs['layer_3'] = layer_3
        layer_4 = self.pretrained.layer4(layer_3)
        self.outputs['layer_4'] = layer_4

        layer_1R = self.pretrainedr.layer1(y)
        self.outputs['layer_1R'] = layer_1R
        layer_2R = self.pretrainedr.layer2(layer_1R)
        self.outputs['layer_2R'] = layer_2R
        layer_3R = self.pretrainedr.layer3(layer_2R)
        self.outputs['layer_3R'] = layer_3R
        layer_4R = self.pretrainedr.layer4(layer_3R)
        self.outputs['layer_4R'] = layer_4R

        layer_1_rn = self.scratch.layer1_rn(layer_1*layer_1R)
        layer_2_rn = self.scratch.layer2_rn(layer_2*layer_2R)
        layer_3_rn = self.scratch.layer3_rn(layer_3*layer_3R)
        layer_4_rn = self.scratch.layer4_rn(layer_4*layer_4R)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        return torch.squeeze(out, dim=1)
        return out

class Image_Mono2StereoFusion(nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=64, backbone="efficientnet_lite3_fusion_wImg", non_negative=True, exportable=True, channels_last=False, align_corners=True,
        blocks={'expand': True}):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(Image_Mono2StereoFusion, self).__init__()

        use_pretrained = False if path else True
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)
        self.pretrainedr, _ = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)

        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)
        '''
        loading the base Model for monocular and freezing the weights of the model.
        '''
        #self.baseModel = MidasNet_small('/content/drive/MyDrive/DDP/HDR/MiDaS/model-small-70d6b9c8.pt')
        self.baseModel = MidasNet_small('/content/drive/MyDrive/Dual Degree Project/MS Project/Consistent DE from Multi-Exposure Data/ALG2-Fusion1_extendedMIDAS_MultiExposureStereo/MiDaS/model-small-70d6b9c8.pt')

        for param in self.baseModel.parameters():
              param.requires_grad = False
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.LeakyReLU(negative_slope=0.01, inplace=False),
            # nn.ReLU(),
        )
        self.outputs = {}
       


    def forward(self, x,y):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)
        x_D = self.baseModel(x)
        y_D = self.baseModel(y)
        x = torch.cat([x,x_D],dim=1)
        y = torch.cat([y,y_D],dim=1)
        layer_1 = self.pretrained.layer1(x)
        self.outputs['layer_1'] = layer_1
        layer_2 = self.pretrained.layer2(layer_1)
        self.outputs['layer_2'] = layer_2
        layer_3 = self.pretrained.layer3(layer_2)
        self.outputs['layer_3'] = layer_3
        layer_4 = self.pretrained.layer4(layer_3)
        self.outputs['layer_4'] = layer_4

        layer_1R = self.pretrainedr.layer1(y)
        self.outputs['layer_1R'] = layer_1R
        layer_2R = self.pretrainedr.layer2(layer_1R)
        self.outputs['layer_2R'] = layer_2R
        layer_3R = self.pretrainedr.layer3(layer_2R)
        self.outputs['layer_3R'] = layer_3R
        layer_4R = self.pretrainedr.layer4(layer_3R)
        self.outputs['layer_4R'] = layer_4R

        layer_1_rn = self.scratch.layer1_rn(layer_1*layer_1R)
        layer_2_rn = self.scratch.layer2_rn(layer_2*layer_2R)
        layer_3_rn = self.scratch.layer3_rn(layer_3*layer_3R)
        layer_4_rn = self.scratch.layer4_rn(layer_4*layer_4R)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        return torch.squeeze(out, dim=1)
        return out





def fuse_model(m):
    prev_previous_type = nn.Identity()
    prev_previous_name = ''
    previous_type = nn.Identity()
    previous_name = ''
    for name, module in m.named_modules():
        if prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d and type(module) == nn.ReLU:
            # print("FUSED ", prev_previous_name, previous_name, name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name, name], inplace=True)
        elif prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d:
            # print("FUSED ", prev_previous_name, previous_name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name], inplace=True)
        # elif previous_type == nn.Conv2d and type(module) == nn.ReLU:
        #    print("FUSED ", previous_name, name)
        #    torch.quantization.fuse_modules(m, [previous_name, name], inplace=True)

        prev_previous_type = previous_type
        prev_previous_name = previous_name
        previous_type = type(module)
        previous_name = name