from nnunet.network_architecture.neural_network import SegmentationNetwork as SN
from monai.networks.nets.vit import ViT
from torch import nn
model_patch_size = [32,128,128]
model_batch_size = 12
model_num_pool_op_kernel_sizes = [[2, 2]]
class custom_net(SN):

    def __init__(self, num_classes):
        super(custom_net, self).__init__()
        self.params = {'content': None}
        self.conv_op = nn.Conv3d
        self.do_ds = False
        self.num_classes = num_classes
        
		######## self.model 设置自定义网络 by Sleeep ########
        self.model = UNet_3D(1, num_classes)
        ######## self.model 设置自定义网络 by Sleeep ########
        
        self.name = "UNet_3D"

    def forward(self, x):
        x = x.permute(0, 1, 3, 4, 2)
        out = self.model(x)
        out = out.permute(0, 1, 4, 2, 3)
        if self.do_ds:
            return [out, ]
        else:
            return out


def create_model():

    return custom_net(num_classes=3)



import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet_3D(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.left_conv_1 = double_conv(channel_in, 64)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_2 = double_conv(64, 128)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_3 = double_conv(128, 256)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_4 = double_conv(256, 512)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_5 = double_conv(512, 1024)

        self.deconv_1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.right_conv_1 = double_conv(1024, 512)
        self.deconv_2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.right_conv_2 = double_conv(512, 256)
        self.deconv_3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.right_conv_3 = double_conv(256, 128)
        self.deconv_4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.right_conv_4 = double_conv(128, 64)
        self.right_conv_5 = nn.Conv3d(64, channel_out, (3,3,3), padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        x = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(x)
        x = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(x)
        x = self.pool_3(feature_3)

        feature_4 = self.left_conv_4(x)
        x = self.pool_4(feature_4)

        x = self.left_conv_5(x)

        # 2：进行解码过程
        x = self.deconv_1(x)
        # 特征拼接、
        x = torch.cat((feature_4, x), dim=1)
        x = self.right_conv_1(x)

        x = self.deconv_2(x)
        x = torch.cat((feature_3, x), dim=1)
        x = self.right_conv_2(x)

        x = self.deconv_3(x)

        x = torch.cat((feature_2, x), dim=1)
        x = self.right_conv_3(x)

        x= self.deconv_4(x)
        x = torch.cat((feature_1, x), dim=1)
        x = self.right_conv_4(x)

        x = self.right_conv_5(x)
        # out = self.sigmoid(out)

        return x


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch), #归一化层
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x