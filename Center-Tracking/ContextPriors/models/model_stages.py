import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models.backbones.resnet_vd import ResNet_vd
from paddleseg.models.layers.layer_libs import ConvBNReLU
from paddleseg.models.layers.layer_libs import AuxLayer


class AggregationModule(nn.Layer):
    """Aggregation Module"""
    
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
    ):
        super(AggregationModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = kernel_size // 2

        self.reduce_conv = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        
        self.t1 = nn.Conv2D(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=out_channels,
        )
        self.t2 = nn.Conv2D(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=out_channels,
        )
        
        self.p1 = nn.Conv2D(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=out_channels,
        )
        self.p2 = nn.Conv2D(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=out_channels,
        )
        self.norm = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward function."""
        x = self.reduce_conv(x)
        x1 = self.t1(x)
        x1 = self.t2(x1)
        
        x2 = self.p1(x)
        x2 = self.p2(x2)
        
        out = self.relu(self.norm(x1 + x2))
        return out


class CPNet(nn.Layer):
    def __init__(self, prior_channels, proir_size, am_kernel_size, pretrained=None, groups=1, ):
        super().__init__()

        self.in_channels = 2048
        self.channels = 256
        self.backbone = ResNet_vd(50, pretrained=pretrained)

        self.prior_channels = prior_channels
        self.prior_size = paddle.to_tensor([proir_size, proir_size])
        self.aggregation = AggregationModule(self.in_channels, prior_channels,
                                             am_kernel_size)
        self.prior_conv = nn.Sequential(
            nn.Conv2D(
                self.prior_channels,
                paddle.prod(self.prior_size),
                1,
                padding=0,
                groups=groups), nn.BatchNorm2D(paddle.prod(self.prior_size)))
        self.intra_conv = ConvBNReLU(
            self.prior_channels, self.prior_channels, 1, padding=0, stride=1)
        self.inter_conv = ConvBNReLU(
            self.prior_channels,
            self.prior_channels,
            1,
            padding=0,
            stride=1,
        )

        self.bottleneck = ConvBNReLU(
            self.in_channels + self.prior_channels * 2,
            self.channels,
            3,
            padding=1,
        )
        # aux辅助函数 !!!out_channels设置为种类数
        self.aux = AuxLayer(
            in_channels=self.in_channels,
            out_channels=150, inter_channels=self.channels)
    
    def forward(self, inputs):
        # inputs B H w C_0
        H = inputs.shape[2]
        W = inputs.shape[3]
        if H != 480 or W != 480:
            inputs = F.interpolate(inputs, (480, 480),
                                   mode='bilinear',
                                   align_corners=True)
        conv1, conv2, conv3, conv4 = self.backbone(inputs)
        batch_size, channels, height, width = conv4.shape
        assert self.prior_size[0] == height and self.prior_size[1] == width

        # B H w C
        value = self.aggregation(conv4)

        # B H W (H*W)
        context_prior_map = self.prior_conv(value)
        
        # B (H*W) (H*W)
        context_prior_map = context_prior_map.reshape(
            (batch_size, paddle.prod(self.prior_size), -1))
        
        # B (H*W) (H*W)
        context_prior_map = paddle.transpose(context_prior_map, (0, 2, 1))
        context_prior_map = F.sigmoid(context_prior_map)
        inter_context_prior_map = 1 - context_prior_map
        
        value = value.reshape((batch_size, self.prior_channels, -1))
        value = paddle.transpose(value, (0, 2, 1))
        
        intra_context = paddle.bmm(context_prior_map, value)

        intra_context = intra_context / paddle.prod(self.prior_size)
        
        intra_context = intra_context.transpose((0, 2, 1))
        intra_context = intra_context.reshape((batch_size, self.prior_channels,
                                               self.prior_size[0],
                                               self.prior_size[1]))
        intra_context = self.intra_conv(intra_context)

        inter_context = paddle.bmm(inter_context_prior_map, value)
        inter_context = inter_context / paddle.prod(self.prior_size)
        inter_context = inter_context.transpose((0, 2, 1))
        inter_context = inter_context.reshape((batch_size, self.prior_channels,
                                               self.prior_size[0],
                                               self.prior_size[1]))
        inter_context = self.inter_conv(inter_context)

        cp_outs = paddle.concat([conv4, intra_context, inter_context], axis=1)
        output = self.bottleneck(cp_outs)
        output = F.interpolate(output, (H, W),
                               mode='bilinear',
                               align_corners=True)
        aux = self.aux(conv4)
        aux = F.interpolate(aux, size=(H, W), mode='bilinear', align_corners=True)
        return output, aux, context_prior_map

# backbonepath = None
# model = CPNet(proir_size=96, am_kernel_size=11, groups=1, prior_channels=256, pretrained=backbonepath)
# info = paddle.summary(model, (1, 3, 768, 768))
# print(info)
