from nnunet.network_architecture.neural_network import SegmentationNetwork as SN
from monai.networks.nets.vit import ViT
from torch import nn
model_patch_size = [32,128,128]
model_batch_size = 4
model_num_pool_op_kernel_sizes = [[2, 2]]
class custom_net(SN):

    def __init__(self, num_classes):
        super(custom_net, self).__init__()
        self.params = {'content': None}
        self.conv_op = nn.Conv3d
        self.do_ds = False
        self.num_classes = num_classes
        
		######## self.model 设置自定义网络 by Sleeep ########
        self.model = ViT(in_channels=1, patch_size=16, num_classes=num_classes,img_size=model_patch_size)
        ######## self.model 设置自定义网络 by Sleeep ########
        
        self.name = "ViT"

    def forward(self, x):
        x = x.permute(0, 1, 3, 4, 2)
        out = self.model(x)['out']
        out = out.permute(0, 1, 4, 2, 3)
        if self.do_ds:
            return [out, ]
        else:
            return out


def create_model():

    return custom_net(num_classes=3)

