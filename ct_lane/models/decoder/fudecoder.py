import torch
import torch.nn as nn
import torch.nn.functional as F
from models.builder import DECODERS

class FuSampling(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(FuSampling, self).__init__()
        self.s = scale_factor
        self.conv_w = nn.Conv2d(in_channels, out_channels * self.s * self.s, 1, bias=False)

    def forward(self, input):
        # grow channels for scale_factor ** 2 times
        output = self.conv_w(input.clone())
        n, c, h, w = output.size()
        # expend height and width from channels dimension
        output = output.permute(0, 3, 2, 1).contiguous()
        output = output.view(n, w, h * self.s, c // self.s)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(n, h * self.s, w * self.s, c // (self.s * self.s))
        # resize to (b,oc,s*w,s*h)
        output = output.permute(0, 3, 1, 2).contiguous()

        return output

class FeatureFused(nn.Module):
    """Module for fused features"""

    def __init__(self, in_channels, inter_channels=48, norm_layer=nn.BatchNorm2d):
        super(FeatureFused, self).__init__()
        self.convs = nn.ModuleList()
        for in_c in in_channels:
            self.convs.append(nn.Sequential(
            nn.Conv2d(in_c, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True)
            ))

    def forward(self, c_list):
        assert len(c_list)>0
        size = c_list[-1].size()[2:]
        for i in range(len(c_list)-1):
            c_list[i] = self.convs[i](F.interpolate(
                c_list[i], size, mode='bilinear', align_corners=True))
        fused_feature = torch.cat(c_list, dim=1)
        return fused_feature

class FuHead(nn.Module):
    def __init__(self, lf_in_channels, in_channels,
    lf_inter_channels=48, fusampler_channels=256, norm_layer=nn.BatchNorm2d):
        super(FuHead, self).__init__()
        self.fuse = FeatureFused(lf_in_channels, lf_inter_channels, norm_layer=norm_layer)
        fuse_in_channels = lf_inter_channels*(len(lf_in_channels))+in_channels
        self.block = nn.Sequential(
            nn.Conv2d( fuse_in_channels, fusampler_channels, 3, padding=1, bias=False),
            norm_layer(fusampler_channels),
            nn.ReLU(True),
            nn.Conv2d(fusampler_channels, fusampler_channels, 3, padding=1, bias=False),
            norm_layer(fusampler_channels),
            nn.ReLU(True)
        )

    def forward(self, x_list):
        fused_feature = self.fuse(x_list)
        out = self.block(fused_feature)
        return out

@DECODERS.registerModule
class FuDecoder(nn.Module):
    def __init__(self, num_class, scale_factor, 
        lf_in_channels: list, in_channels, fusampler_channels=256, norm_layer='BatchNorm2d', cfg=None) -> None:
        super(FuDecoder, self).__init__()
        self.cfg = cfg
        self.norm_layer = getattr(nn, norm_layer)
        self.head = FuHead(lf_in_channels, in_channels, fusampler_channels, norm_layer=self.norm_layer)
        self.dupsample = FuSampling(fusampler_channels, num_class, scale_factor=scale_factor)

        self.dropout = nn.Dropout2d(p=0.1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.T = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
            # nn.AvgPool2d(4,4),
            nn.Upsample(size=(1, 1), mode='bilinear', align_corners=True),
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, in_channels*2),
            nn.Dropout(0.1),
            nn.Linear(in_channels*2, num_class-1),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(nn.Linear(num_class, num_class-1),nn.Sigmoid())

    def forward(self, input, low_feature):
        # temp result for exists lanes(only using on culane)
        exists = self.T(input.clone())

        low_feature.append(input)
        output = self.head(low_feature)
        output = self.dropout(output)
        output = self.dupsample(output)
        output = self.softmax(output)

        return output, exists

if __name__ == '__main__':
    from torchsummary import summary
    t_net = FuDecoder(num_class=6, scale_factor=8, lf_in_channels=[16, 16], in_channels=512)
    llf_1 = torch.rand(1, 16, 16, 32)
    llf_2 = torch.rand(1, 16, 8, 16)
    x = torch.rand(1, 512, 4, 8)

    print(t_net(x, [llf_1, llf_2]).shape)

