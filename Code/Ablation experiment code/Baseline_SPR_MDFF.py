import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
# 双输入的VIT
# DEVICE = "cuda:1"  # device

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels,padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def choquet_integral_batch(inputs, fuzzy_measure):
    """
    计算给定输入和模糊测度的Choquet积分（批量处理）。

    :param inputs: 形状为 (B, C, H, W, 2) 的张量，表示输入值。
    :param fuzzy_measure: 形状为 (2,) 的张量，表示模糊测度。
    :return: 形状为 (B, C, H, W) 的张量，表示Choquet积分的结果。
    """
    # 对输入值和对应的模糊测度进行降序排序
    sorted_inputs, indices = torch.sort(inputs, dim=-1, descending=True)
    fuzzy_measure = fuzzy_measure.to(inputs.device)
    sorted_fuzzy_measure = fuzzy_measure[indices]

    # 计算模糊测度的差分
    diffs = torch.cat([sorted_fuzzy_measure[..., :1], sorted_fuzzy_measure[..., 1:] - sorted_fuzzy_measure[..., :-1]],
                      dim=-1)

    # 计算Choquet积分
    choquet_value = torch.sum(sorted_inputs * diffs, dim=-1)

    return choquet_value


def fuse_feature_maps(feature_map1, feature_map2, fuzzy_measure):
    """
    使用Choquet模糊积分融合两个特征图。

    :param feature_map1: 第一个特征图，形状为 (B, C, H, W)。
    :param feature_map2: 第二个特征图，形状为 (B, C, H, W)。
    :param fuzzy_measure: 模糊测度，形状为 (2,)。
    :return: 融合后的特征图，形状为 (B, C, H, W)。
    """
    # 将两个特征图堆叠在一起，形状为 (B, C, H, W, 2)
    inputs = torch.stack([feature_map1, feature_map2], dim=-1)

    # 计算Choquet积分
    fused_feature_map = choquet_integral_batch(inputs, fuzzy_measure)

    return fused_feature_map

class Module2(nn.Module):
    def __init__(self, channels):
        super(Module2,self).__init__()
        self.conv = DepthwiseSeparableConv(channels*2,channels)
        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = DoubleConv3(channels,channels)
    def forward(self, feature1, feature2):
        fuzzy_measure = torch.tensor([0.5, 0.5])
        feature_dim2 = torch.cat([feature1, feature2], dim=2)
        feature_dim3 = torch.cat([feature1, feature2], dim=3)

        feature_dim3_trasposed = feature_dim3.permute(0,1,3,2)
        feature_fuse1 = torch.cat([feature_dim2,feature_dim3_trasposed],dim=3)
        feature_dim2_trasposed = feature_dim2.permute(0,1,3,2)
        feature_fuse2 = torch.cat([feature_dim2_trasposed,feature_dim3],dim=2)
        feature_fuse1 = self.conv1(feature_fuse1)
        feature_fuse1 = self.pool(feature_fuse1)
        feature_fuse2 = self.conv1(feature_fuse2)
        feature_fuse2 = self.pool(feature_fuse2)
        feature_fuse = torch.cat([feature_fuse1,feature_fuse2],dim=1)
        feature_fuse = self.conv(feature_fuse)
        feature_fuse3 = fuse_feature_maps(feature1, feature_fuse, fuzzy_measure)

        feature_dim1 = torch.cat([feature2, feature_fuse3], dim=1)

        feature_dim1 = feature_dim1.float()

        return feature_dim1
class MinPooling2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(MinPooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # 使用 unfold 展开窗口
        unfolded = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # 计算每个窗口的最小值
        min_pooled = unfolded.contiguous().view(*unfolded.size()[:-2], -1).min(dim=-1)[0]
        return min_pooled


class DoubleConv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.double_conv(x)


#===============
# A_Former_v0
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = pair(img_size)
        patch_size = pair(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2):
        x1 = self.norm(x1)
        x2 = self.norm(x2)

        qkv_1 = self.to_qkv(x1).chunk(3, dim=-1)
        qkv_2 = self.to_qkv(x2).chunk(3, dim=-1)
        _, k_1, v_1 = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv_1)
        q_2, _, _ = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv_2)

        dots = torch.matmul(q_2, k_1.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads,
                          dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        for attn, ff in self.layers:
            x1 = attn(x1, x2) + x1
            x1 = ff(x1) + x1
        x = self.norm(x1)
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes=None, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0., new_shape=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.to_patch_embedding = PatchEmbed(img_size=pair(
            image_size), patch_size=pair(patch_size), in_c=channels, embed_dim=dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        if num_classes is not None:
            self.conv_head = nn.Conv2d(dim, num_classes, kernel_size=1)
        else:
            self.conv_head = nn.Identity()
        if new_shape is not None:
            self.upsample = nn.Upsample(size=(new_shape, new_shape), mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()

    def forward(self, x1, x2):
        x2 = self.to_patch_embedding(x2)
        x1 = self.to_patch_embedding(x1)

        b, n, _ = x1.shape
        b, n, _ = x2.shape

        x1 += self.pos_embedding
        x1 = self.dropout(x1)
        x2 += self.pos_embedding
        x2 = self.dropout(x2)

        x = self.transformer(x1, x2)

        # Converting tokens back to spatial dimensions
        new_dim = int(n ** 0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=new_dim, w=new_dim)

        # Upsample to original image size
        x = self.upsample(x)
        x = self.conv_head(x)

        return x

#==============================


class KU_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super(KU_Block, self).__init__()
        self.conv2 = nn.Conv2d(in_c // 4, in_c, kernel_size=1, padding=0, stride=2)
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        xs = torch.chunk(x, 4, dim=1)
        x1, x2, x3, x4 = xs
        b, c, h, w = x1.shape
        new_h = h * 2
        new_w = w * 2

        # 使用输入张量的数据类型和设备创建新张量
        out = torch.zeros(b, c, new_h, new_w, dtype=x1.dtype, device=x1.device)

        out[:, :, 0::2, 0::2] = x1
        out[:, :, 0::2, 1::2] = x2
        out[:, :, 1::2, 0::2] = x3
        out[:, :, 1::2, 1::2] = x4

        # 确保所有操作都在相同的数据类型下进行
        out = out.to(x.dtype)

        out = self.conv2(out)
        out = self.conv3_1(out)
        return out

class Baseline_SPR_MDFF(nn.Module):
    def __init__(self):
        super(Baseline_SPR_MDFF, self).__init__()

        # encoder of UNet
        self.dconv_k3_1 = DoubleConv3(3, 64)
        self.dconv_k3_2 = DoubleConv3(64, 128)
        self.dconv_k3_3 = DoubleConv3(128, 256)
        self.dconv_k3_4 = DoubleConv3(256, 512)

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.downsample = nn.MaxPool3d(2)

        # encoder of A_VIT
        self.encoder_f_1 = ViT(image_size=(224, 224), patch_size=(16, 16), num_classes=64, dim=196, depth=1, heads=12,
                               mlp_dim=256, channels = 64, dim_head=64, dropout=0.1, emb_dropout=0.1, new_shape=224)
        self.encoder_f_2 = ViT(image_size=(224, 224), patch_size=(8, 8), num_classes=128, dim=196, depth=1, heads=12,
                                mlp_dim=256, channels = 64, dim_head=64, dropout=0.1, emb_dropout=0.1, new_shape=112)
        self.encoder_f_3 = ViT(image_size=(112, 112), patch_size=(4, 4), num_classes=256, dim=196, depth=1, heads=12,
                                mlp_dim=256, channels = 128, dim_head=64, dropout=0.1, emb_dropout=0.1, new_shape=56)
        self.encoder_f_4 = ViT(image_size=(56, 56), patch_size=(2, 2), num_classes=512, dim=196, depth=1, heads=12,
                                mlp_dim=256, channels = 256, dim_head=64, dropout=0.1, emb_dropout=0.1, new_shape=28)
        self.K_block_1 = KU_Block(64, 64)
        self.K_block_2 = KU_Block(64, 64)
        self.K_block_3 = KU_Block(128, 128)
        self.K_block_4 = KU_Block(256, 256)


        # conv_k5 -> f
        self.conv_k5_1 = DoubleConv3(64, 64)
        self.conv_k5_2 = DoubleConv3(128, 128)
        self.conv_k5_3 = DoubleConv3(256, 256)
        self.conv_k5_4 = DoubleConv3(512, 512)

        #decoder
        self.conv_d_4 = DoubleConv3(1024, 512)
        self.conv_d_3 = DoubleConv3(1024, 256)
        self.conv_d_2 = DoubleConv3(512, 128)
        self.conv_d_1 = DoubleConv3(256, 64)

        self.Module2_1 = Module2(512)
        self.Module2_2 = Module2(256)
        self.Module2_3 = Module2(128)

        # out
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)

        # up
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.minpool = MinPooling2D()


    def forward(self, x):
        x_f = x
        x_u = x

        # encoder of UNet
        x_u_1 = self.dconv_k3_1(x_u)
        x_u_1_m = self.minpool(x_u_1)
        x_u_2 = self.downsample(x_u_1)
        x_u_2 = x_u_2 - x_u_1_m
        x_u_2 = self.dconv_k3_2(x_u_2)
        x_u_2_m = self.minpool(x_u_2)
        x_u_3 = self.downsample(x_u_2)
        x_u_3 = x_u_3 - x_u_2_m
        x_u_3 = self.dconv_k3_3(x_u_3)
        x_u_3_m = self.minpool(x_u_3)
        x_u_4 = self.downsample(x_u_3)
        x_u_4 = x_u_4 - x_u_3_m
        x_u_4 = self.dconv_k3_4(x_u_4)

        # encoder of A_Former
        x_f = self.dconv_k3_1(x_f)
        x_f = self.K_block_1(x_f)
        x_f_1 = self.encoder_f_1(x_f, x_u_1)
        x_f_1 = self.K_block_2(x_f_1)
        x_f_2 = self.encoder_f_2(x_f_1, x_u_1)
        x_f_2 = self.K_block_3(x_f_2)
        x_f_3 = self.encoder_f_3(x_f_2, x_u_2)
        x_f_3 = self.K_block_4(x_f_3)
        x_f_4 = self.encoder_f_4(x_f_3, x_u_3)


        # f + u
        skip_1 = x_f_1 + x_u_1
        skip_2 = x_f_2 + x_u_2
        skip_3 = x_f_3 + x_u_3
        skip_4 = x_f_4 + x_u_4

        # dconv -> f
        x_f_1 = self.conv_k5_1(skip_1)
        x_f_2 = self.conv_k5_2(skip_2)
        x_f_3 = self.conv_k5_3(skip_3)
        x_f_4 = self.conv_k5_4(skip_4)

        # concat f and skip
        x_d_1 = torch.cat((x_f_1, skip_1), dim=1)
        x_d_2 = torch.cat((x_f_2, skip_2), dim=1)
        x_d_3 = torch.cat((x_f_3, skip_3), dim=1)
        x_d_4 = torch.cat((x_f_4, skip_4), dim=1)

        # decoder
        x_d_4 = self.conv_d_4(x_d_4)
        x_d_4 = self.up(x_d_4)
        # x_d_3 = torch.cat((x_d_3, x_d_4), dim=1)
        x_d_3 = self.Module2_1(x_d_4, x_d_3)

        x_d_3 = self.conv_d_3(x_d_3)
        x_d_3 = self.up(x_d_3)
        # x_d_2 = torch.cat((x_d_2, x_d_3), dim=1)
        x_d_2 = self.Module2_2(x_d_3, x_d_2)

        x_d_2 = self.conv_d_2(x_d_2)
        x_d_2 = self.up(x_d_2)
        # x_d_1 = torch.cat((x_d_1, x_d_2), dim=1)
        x_d_1 = self.Module2_3(x_d_2, x_d_1)

        x_d_1 = self.conv_d_1(x_d_1)


        # output
        x_out = self.conv_out(x_d_1)



        return x_out



