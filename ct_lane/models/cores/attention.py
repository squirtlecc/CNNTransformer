import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mish import Mish

# python3.x
from functools import reduce


class Attention(nn.Module):
    """
        the attention need to scale feature to small size for transformer.
        and to get a small feature map can using convs and linear to get it.
        maybe can design a martix to transform the feature map to small size,
        like word hidden space embedding. ("bchw,hiwj->bcij", x, y)
    """
    def __init__(self, in_channels, out_channels,
            multi_head=3, down_scale=1, attention_times=1, input_shape=None):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention_times = attention_times
        self.attention_shape = torch.Size([-1, in_channels, 16, 32])
        self.down_scale = down_scale
        self.input_shape = input_shape
        self._flag = True

        self.attention_blocks = self.getAttentionBlocks(
            attention_times, multi_head)
        self.down_conv, self.up_conv = self.getScaleConv()
        
        self.position_encodding = PositionEncodingSine()

        # fixed all module in init instead of in forward if has input_shape.
        if input_shape is None:
            self.to_attention = self.getEncoderConv()
            # get_attention to ceate this modules in forward.
            self.position_embedding = self.getPositionEmbedding()
            self.over_attention = self.getDecoderConv()
        else:
            self.initModuleBlocks(input_shape, down_scale)


    def initModuleBlocks(self, input_shape, down_scale):
        _init_block = torch.zeros((1,1,input_shape[0]//down_scale, input_shape[1]//down_scale))
        self.getEncoderConv(_init_block)
        self.getPositionEmbedding(_init_block)
        self.getDecoderConv(_init_block)
        self.position_encodding = PositionEncodingSine(_init_block.shape[-1]*_init_block.shape[-2])
        self._flag = False


    def getAttentionBlocks(self, attention_times=1, multi_head=1):
        self.attention_blocks = nn.ModuleList()
        for _ in range(attention_times):
            self.attention_blocks.append(AttentionBlock(
                self.in_channels, self.in_channels,
                ratio=2, multi_head=multi_head,
                in_shape=(self.attention_shape[-2], self.attention_shape[-1])))
        return self.attention_blocks

    def getScaleConv(self):
        downsample_conv = []
        upsample_conv = []
        for _ in range(self.down_scale):
            downsample_conv.append(nn.Conv2d(
                self.in_channels, self.in_channels,
                kernel_size=3, stride=2, padding=1, bias=False))
            downsample_conv.append(nn.BatchNorm2d(self.in_channels))

            upsample_conv.append(nn.ConvTranspose2d(
                self.in_channels, self.in_channels,
                kernel_size=2, stride=2, padding=0, bias=False))
        upsample_conv.append(nn.Conv2d(
            self.in_channels, self.out_channels, 3, 1, 1))

        return nn.Sequential(*downsample_conv), nn.Sequential(*upsample_conv)

    def getEncoderConv(self, fixed_data=None, hidden_features=1024*2):
        # only include the linear conv that from scale conv to 16x16 linear conv.
        if fixed_data is None:
            if hasattr(self, 'to_attention'):
                return self.to_attention
            # not test yet
            to_attention = []
            to_attention = nn.Sequential(*to_attention)
            return to_attention
        # start 2 to skip the batch and channels dimension.
        in_features = reduce(lambda x, y:x*y, list(fixed_data.shape[-2:]))
        out_features = reduce(lambda x, y:x*y, list(self.attention_shape[-2:]))


        to_attention = [
            nn.Flatten(start_dim=-2),
            nn.Linear(in_features, hidden_features),
            nn.Linear(hidden_features, out_features),
            NReshape(*list(self.attention_shape[1:])),
        ]
        self.to_attention = nn.Sequential(*to_attention).to(fixed_data.device)

        return self.to_attention

    def getPositionEmbedding(self, fixed_data=None):
        # the position embedding only add to the scaled features.
        # maybe need test add before scale convs.
        if fixed_data is None:
            return nn.Sequential()
        if not self.position_embedding:
            pos_module = PositionEmbedding(
                fixed_data.shape[1:]).to(fixed_data.device)
            self.position_embedding = pos_module
        return self.position_embedding

    def getDecoderConv(self, fixed_data=None, hidden_features=1024*2):
        if fixed_data is None:
            return nn.Sequential()
        # start 2 to skip the batch and channels dimension.
        in_features, out_features = 1, 1
        for f in fixed_data.shape[-2:]:
            out_features *= f
        for f in self.attention_shape[-2:]:
            in_features *= f
        decoder_list = []
        
        if not self.over_attention:
            decoder_list = [
                nn.Flatten(start_dim=-2),
                nn.Linear(in_features, hidden_features),
                nn.Linear(hidden_features, out_features),
                NReshape(*list(fixed_data.shape[1:])),
            ]

            self.over_attention = nn.Sequential(*decoder_list).to(fixed_data.device)

        return self.over_attention
        
    def fixedToAttentionLayer(self, fixed):
        if self._flag:
            self._flag = False
            _fixed = self.down_conv(fixed)
            self.getEncoderConv(_fixed)
            self.getPositionEmbedding(_fixed)
            self.getDecoderConv(_fixed)


    def forward(self, input):
        self.fixedToAttentionLayer(input)
        down_scale_input = self.down_conv(input)

        position_input = self.position_embedding(down_scale_input)
        attention_input = self.to_attention(position_input)

        attention_output = attention_input.clone()
        for attention_block in self.attention_blocks:
            attention_output = attention_block(attention_output) + attention_output

        output = self.over_attention(attention_output)
        output = output 

        output = self.up_conv(output)
        return output


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
        ratio=1, multi_head=1, in_shape=(16, 16)):
        super(AttentionBlock, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.ratio = ratio
        self.hidden_dim = in_channels // ratio
        self.multi_head = multi_head
        self.divide = math.sqrt(in_shape[-1]*in_shape[-2])
        self.atn_shape = in_shape
        self.conv_h_1x1 = nn.Conv2d(self.in_c, self.hidden_dim, 1, 1, 0)


        self.query, self.key, self.value = \
            self.getMultiHeadQKV()

        self.softmax = nn.Softmax(dim=-1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.layernorm = nn.LayerNorm([
            self.hidden_dim, in_shape[-2], in_shape[-2]])
        self.batchnorm = nn.BatchNorm2d(self.hidden_dim)

        self.fusion_conv = self.getFusionConv(multi_head)

        self.add_and_norm = nn.Sequential(
            nn.LayerNorm([self.in_c, in_shape[-2], in_shape[-2]]),
            nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False),
        )
        self.M = nn.Parameter(torch.FloatTensor(in_shape[-2], in_shape[-2]), requires_grad=True)

        self.gamma = nn.Parameter(torch.zeros(multi_head))


    def getFusionConv(self, multi_head):

        fusion_conv = [
            nn.Conv2d(self.in_c+self.hidden_dim*multi_head, self.in_c+self.hidden_dim, 1, 1, 0),
            nn.LayerNorm([self.in_c+self.hidden_dim, self.atn_shape[-2], self.atn_shape[-1]]),
            nn.Conv2d(self.in_c+self.hidden_dim, self.in_c+self.hidden_dim, 1, 1, 0),
            nn.Conv2d(self.in_c+self.hidden_dim, self.out_c, 1, 1, 0),
        ]
        self.fusion_conv = nn.Sequential(*fusion_conv)
        return self.fusion_conv


    def getMultiHeadQKV(self):
        query = nn.ModuleList()
        key = nn.ModuleList()
        value = nn.ModuleList()

        for _ in range(self.multi_head):
            query.append(nn.Sequential(
                nn.Conv2d(self.in_c, self.hidden_dim*2, 1, 1, 0, bias=False),
                nn.Conv2d(self.hidden_dim*2, self.hidden_dim*2, 1, 1, 0, bias=False),
                nn.Softmax(dim=1),
                nn.Conv2d(self.hidden_dim*2, self.hidden_dim, 1, 1, 0, bias=False),

            ))
            key.append(nn.Sequential(
                nn.Conv2d(self.in_c, self.hidden_dim*2, 1, 1, 0, bias=False),
                nn.Softmax(dim=1),
                nn.Conv2d(self.hidden_dim*2, self.hidden_dim*2, 1, 1, 0, bias=False),
                nn.Conv2d(self.hidden_dim*2, self.hidden_dim, 1, 1, 0, bias=False),
            ))
            value.append(nn.Sequential(
                nn.Conv2d(self.in_c, self.hidden_dim*2, 1, 1, 0, bias=False),
                nn.Conv2d(self.hidden_dim*2, self.hidden_dim*2, 1, 1, 0, bias=False),
                nn.ReLU(),
                nn.Conv2d(self.hidden_dim*2, self.hidden_dim, 1, 1, 0, bias=False),
            ))
        return query, key, value

    def forward(self, input):
        multi_concat = input.clone()
        for i in range(self.multi_head):
            q = self.query[i](input)
            k = self.key[i](input)
            v = self.value[i](input)

            # attention matrix
            self_score = q @ k.transpose(-1, -2) / (self.divide + 1e-8)
            soft_addressing = self.sigmoid(self.batchnorm(self_score))
            self_attention = soft_addressing @ v
            multi_concat = torch.cat([multi_concat, self_attention], dim=1)
        return self.fusion_conv(multi_concat)

class PositionEncodingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = 10000

    def forward(self, x):
        if(len(x.shape) == 4):
            self.num_pos_feats = x.shape[-1]*x.shape[-2]
        mask = torch.ones((x.shape[:-2]), dtype=torch.int8, device=x.device)
        # mask = torch.ones_like(x, dtype=torch.int8, device=x.device)
        c_embed = mask.cumsum(-1, dtype=torch.float32)

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_c = c_embed[..., None] * dim_t
        # print(pos_c.shape)
        pos_c = torch.stack(
            (pos_c[..., 0::2].sin(), pos_c[..., 1::2].cos()),
            dim=-1).flatten(-2)
        # print(f"pos_c.shape: {pos_c.shape}")
        
        pos = pos_c.reshape(x.shape)
        # print(pos[0,0,...])
        return pos+x


class PositionEmbedding(nn.Module):
    def __init__(self, shape=(64, 64)):
        super(PositionEmbedding, self).__init__()
        if len(shape) >= 3:
            self.shape = torch.Size([-1, shape[-3], shape[-2], shape[-1]])
        if len(shape) == 2:
            self.shape = torch.Size([-1, -1, shape[-2], shape[-1]])

        self.position_embedding = nn.Parameter(F.tanh(torch.FloatTensor(shape)), requires_grad=True)
    def forward(self, input):
        output = self.position_embedding + input
        return output


class NReshape(nn.Module):
    def __init__(self, *args):
        super(NReshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.shape[0],)+self.shape)


if __name__ == "__main__":

    x = torch.zeros(128, 16, 32).bool()
    posi = PositionEncodingSine(16*32)
    t = posi(x)
    print(t[100,...])
    print(t.shape)