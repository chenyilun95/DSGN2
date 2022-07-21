import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

def convbn(in_planes,
           out_planes,
           kernel_size,
           stride,
           pad,
           dilation=1,
           gn=False,
           groups=32):
    if gn and out_planes % 32 != 0: 
        print('Cannot apply GN as the channels is not 32-divisible.')
        gn = False
    return nn.Sequential(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=dilation if dilation > 1 else pad,
                  dilation=dilation,
                  bias=False),
        nn.BatchNorm2d(out_planes) if not gn else nn.GroupNorm(
            groups, out_planes))


def convbn_3d(in_planes,
              out_planes,
              kernel_size,
              stride,
              pad,
              gn=False,
              groups=32):
    return nn.Sequential(
        nn.Conv3d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  padding=pad,
                  stride=stride,
                  bias=False),
        nn.BatchNorm3d(out_planes) if not gn else nn.GroupNorm(
            groups, out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride,
                 downsample,
                 pad,
                 dilation,
                 gn=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes, planes, 3, stride, pad, dilation, gn=gn),
            nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation, gn=gn)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class disparityregression(nn.Module):
    def __init__(self):
        super(disparityregression, self).__init__()

    def forward(self, x, depth):
        assert len(x.shape) == 4
        assert len(depth.shape) == 1
        out = torch.sum(x * depth[None, :, None, None], 1)
        return out


class hourglass(nn.Module):
    def __init__(self, inplanes, gn=False, planes_mul=[2, 2]):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_3d(inplanes,
                      inplanes * planes_mul[0],
                      kernel_size=3,
                      stride=2,
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * planes_mul[0],
                               inplanes * planes_mul[0],
                               kernel_size=3,
                               stride=1,
                               pad=1,
                               gn=gn)

        self.conv3 = nn.Sequential(
            convbn_3d(inplanes * planes_mul[0],
                      inplanes * planes_mul[1],
                      kernel_size=3,
                      stride=2,
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            convbn_3d(inplanes * planes_mul[1],
                      inplanes * planes_mul[1],
                      kernel_size=3,
                      stride=1,
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * planes_mul[1],
                               inplanes * planes_mul[0],
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * planes_mul[0]) if not gn else nn.GroupNorm(32, inplanes * planes_mul[0]))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * planes_mul[0],
                               inplanes,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes)
            if not gn else nn.GroupNorm(32, inplanes))  # +x

    def forward(self, x, presqu=None, postsqu=None):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu,
                          inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        if presqu is None and postsqu is None:
            return out
        else:
            return out, pre, post

class hourglass_bev(hourglass):
    def __init__(self, inplanes, gn=False):
        super(hourglass_bev, self).__init__(inplanes, gn)

        self.conv1 = nn.Sequential(
            convbn_3d(inplanes,
                      inplanes * 2,
                      kernel_size=3,
                      stride=(1, 2, 2),
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               stride=1,
                               pad=1,
                               gn=gn)

        self.conv3 = nn.Sequential(
            convbn_3d(inplanes * 2,
                      inplanes * 2,
                      kernel_size=3,
                      stride=(1, 2, 2),
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            convbn_3d(inplanes * 2,
                      inplanes * 2,
                      kernel_size=3,
                      stride=1,
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               padding=1,
                               output_padding=(0, 1, 1),
                               stride=(1, 2, 2),
                               bias=False),
            nn.BatchNorm3d(inplanes *
                           2) if not gn else nn.GroupNorm(32, inplanes *
                                                          2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2,
                               inplanes,
                               kernel_size=3,
                               padding=1,
                               output_padding=(0, 1, 1),
                               stride=(1, 2, 2),
                               bias=False),
            nn.BatchNorm3d(inplanes)
            if not gn else nn.GroupNorm(32, inplanes))  # +x


class hourglass2d(nn.Module):
    def __init__(self, inplanes, gn=False):
        super(hourglass2d, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes,
                   inplanes * 2,
                   kernel_size=3,
                   stride=2,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv2 = convbn(inplanes * 2,
                            inplanes * 2,
                            kernel_size=3,
                            stride=1,
                            pad=1,
                            dilation=1,
                            gn=gn)

        self.conv3 = nn.Sequential(
            convbn(inplanes * 2,
                   inplanes * 2,
                   kernel_size=3,
                   stride=2,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            convbn(inplanes * 2,
                   inplanes * 2,
                   kernel_size=3,
                   stride=1,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes *
                           2) if not gn else nn.GroupNorm(32, inplanes *
                                                          2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2,
                               inplanes,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes)
            if not gn else nn.GroupNorm(32, inplanes))  # +x

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu,
                          inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post

class upconv_module(nn.Module):
    def __init__(self, in_channels, up_channels, share_upconv=False, final_channels=None, kernel1=True):
        super(upconv_module, self).__init__()
        self.num_stage = len(in_channels) - 1
        self.conv = nn.ModuleList()
        self.redir = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            self.conv.append(
                convbn(in_channels[0] if stage_idx == 0 else up_channels[stage_idx - 1], up_channels[stage_idx], 3 if stage_idx != 0 or not kernel1 else 1, 1, 1 if stage_idx != 0 or not kernel1  else 0, 1)
            )
            self.redir.append(
                convbn(in_channels[stage_idx + 1], up_channels[stage_idx], 3, 1, 1, 1)
            )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.share_upconv = share_upconv
        if self.share_upconv:
            self.lastconv = nn.Conv2d(up_channels[stage_idx], final_channels[-1],
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False)
            self.rpnconv = nn.Conv2d(up_channels[stage_idx], final_channels[-1],
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False)

    def forward(self, feats):
        x = feats[0]
        for stage_idx in range(self.num_stage):
            x = self.conv[stage_idx](x)
            redir = self.redir[stage_idx](feats[stage_idx + 1])
            x = F.relu(self.up(x) + redir)
        
        if self.share_upconv:
            return self.lastconv(x), self.rpnconv(x)
        else:
            return x

class upconv_module_cat(nn.Module):
    def __init__(self, in_channels, up_channels, final_channels):
        super(upconv_module_cat, self).__init__()
        self.num_stage = len(in_channels)
        self.conv = nn.ModuleList()
        self.squeezeconv = nn.Conv2d(sum(up_channels), final_channels,
            kernel_size=1, padding=0, stride=1, bias=True)
        # nn.Sequential(
        #     convbn(sum(up_channels), final_channels, 1, 1, 0, 1))
        for stage_idx in range(self.num_stage):
            self.conv.append(nn.Sequential(
                convbn(in_channels[stage_idx], up_channels[stage_idx], 1, 1, 0, 1),
                nn.ReLU(inplace=True)))

    def forward(self, feats):
        feat_0 = F.interpolate(self.conv[0](feats[0]), scale_factor=4, mode='bilinear', align_corners=True)
        feat_1 = F.interpolate(self.conv[1](feats[1]), scale_factor=2, mode='bilinear', align_corners=True)
        feat_2 = self.conv[2](feats[2])
        cat_feats = torch.cat([feat_0, feat_1, feat_2], dim=1)
        x = self.squeezeconv(cat_feats)
        return x

class upconv_module_catk3(nn.Module):
    def __init__(self, in_channels, up_channels, final_channels, share_upconv=False):
        super(upconv_module_catk3, self).__init__()
        self.num_stage = len(in_channels)
        self.conv = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            self.conv.append(nn.Sequential(
                convbn(in_channels[stage_idx], up_channels[stage_idx], 3, 1, 1, 1),
                nn.ReLU(inplace=True)))
        self.squeezeconv = nn.Sequential(
            convbn(sum(up_channels), final_channels[0], 3, 1, 1, gn=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels[0], final_channels[1],
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=False))
        
        self.share_upconv = share_upconv
        if self.share_upconv:
            self.lastconv = nn.Conv2d(final_channels[1], final_channels[1],
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False)
            self.rpnconv = nn.Conv2d(final_channels[1], final_channels[1],
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False)

    def forward(self, feats):
        feat_0 = F.interpolate(self.conv[0](feats[0]), scale_factor=4, mode='bilinear', align_corners=True)
        feat_1 = F.interpolate(self.conv[1](feats[1]), scale_factor=2, mode='bilinear', align_corners=True)
        feat_2 = self.conv[2](feats[2])
        cat_feats = torch.cat([feat_0, feat_1, feat_2], dim=1)
        x = self.squeezeconv(cat_feats)

        if self.share_upconv:
            return self.lastconv(x), self.rpnconv(x)
        else:
            return x

class feature_extraction_neck(nn.Module):
    def __init__(self, cfg):
        super(feature_extraction_neck, self).__init__()

        self.cfg = cfg
        self.in_dims = cfg.in_dims
        self.with_upconv = cfg.with_upconv
        self.share_upconv = getattr(cfg, 'share_upconv', False)
        self.upconv_type = getattr(cfg, 'upconv_type', 'fpn')
        self.start_level = cfg.start_level
        self.cat_img_feature = cfg.cat_img_feature
        self.drop_psv = getattr(cfg, 'drop_psv', False)
        self.with_upconv_voxel = getattr(cfg, 'with_upconv_voxel', False)
        self.mono = getattr(cfg, 'mono', False)
        self.with_sem_neck = getattr(cfg, 'with_sem_neck', False)
        self.with_spp = getattr(cfg, 'with_spp', True)
        self.extra_sem = getattr(cfg, 'extra_sem', False)

        assert not self.share_upconv or (not self.drop_psv and self.with_upconv_voxel and self.with_upconv)
        assert not getattr(cfg, 'swap_feature', False)

        self.sem_dim = cfg.sem_dim
        self.stereo_dim = cfg.stereo_dim
        self.spp_dim = getattr(cfg, 'spp_dim', 32)
        if self.mono and not self.drop_psv:
            assert self.stereo_dim[-1] > 32

        concat_dim = sum(self.in_dims[self.start_level:])
        if self.with_spp:
            self.spp_branches = nn.ModuleList([
                nn.Sequential(
                    nn.AvgPool2d(s, stride=s),
                    convbn(self.in_dims[-1],
                        self.spp_dim,
                        1, 1, 0,
                        gn=cfg.GN,
                        groups=min(32, self.spp_dim)),
                    nn.ReLU(inplace=True))
                for s in [(64, 64), (32, 32), (16, 16), (8, 8)]])
            concat_dim += self.spp_dim * len(self.spp_branches)

        if self.with_upconv and not self.drop_psv:
            assert self.start_level == 2
            if self.upconv_type == 'fpn':
                self.up_dims = getattr(cfg, 'up_dims', [64, 32])
                self.kernel1 = getattr(cfg, 'kernel1', True)
                self.upconv_module = upconv_module([concat_dim, self.in_dims[1], self.in_dims[0]], self.up_dims, share_upconv=self.share_upconv, final_channels=(self.stereo_dim[-2], self.stereo_dim[-1]), kernel1=self.kernel1)
            elif self.upconv_type == 'cat':
                self.upconv_module = upconv_module_cat([concat_dim, self.in_dims[1], self.in_dims[0]], [128, 32, 32], final_channels=self.stereo_dim[-1])
            elif self.upconv_type == 'catk3':
                self.upconv_module = upconv_module_catk3([concat_dim, self.in_dims[1], self.in_dims[0]], [128, 32, 32], final_channels=(self.stereo_dim[-2], self.stereo_dim[-1]), share_upconv=self.share_upconv)
            else:
                raise ValueError('Invalid upconv type.')
            stereo_dim = 32
        else:
            stereo_dim = concat_dim
            assert self.start_level >= 1

        if not self.drop_psv and not self.share_upconv:
            if (self.with_upconv and self.upconv_type != 'fpn') or self.share_upconv:
                self.lastconv = nn.Identity()
            else:
                self.lastconv = nn.Sequential(
                    convbn(stereo_dim, self.stereo_dim[0], 3, 1, 1, gn=cfg.GN),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.stereo_dim[0], self.stereo_dim[1],
                            kernel_size=1,
                            padding=0,
                            stride=1,
                            bias=False))
        if not self.share_upconv:
            if self.cat_img_feature or self.with_sem_neck:
                if self.with_upconv_voxel:
                    assert self.start_level == 2
                    if self.upconv_type == 'fpn':
                        self.up_dims = getattr(cfg, 'up_dims', [64, 32])
                        self.kernel1 = getattr(cfg, 'kernel1', True)
                        self.upconv_module_voxel = upconv_module([concat_dim, self.in_dims[1], self.in_dims[0]], self.up_dims, kernel1=self.kernel1)
                        self.rpnconv = nn.Sequential(
                            convbn(self.up_dims[-1], self.sem_dim[0], 3, 1, 1, 1, gn=cfg.GN),
                            nn.ReLU(inplace=True),
                            convbn(self.sem_dim[0], self.sem_dim[1], 3, 1, 1, gn=cfg.GN),
                            nn.ReLU(inplace=True)
                        )
                        if self.extra_sem:
                            self.extra_rpnconv = nn.Sequential(
                                convbn(self.up_dims[-1], self.sem_dim[0], 3, 1, 1, 1, gn=cfg.GN),
                                nn.ReLU(inplace=True),
                                convbn(self.sem_dim[0], self.sem_dim[1], 3, 1, 1, gn=cfg.GN),
                                nn.ReLU(inplace=True)
                            )
                    elif self.upconv_type == 'cat':
                        self.upconv_module_voxel = upconv_module_cat([concat_dim, self.in_dims[1], self.in_dims[0]], [128, 32, 32], final_channels=self.sem_dim[-1])
                    elif self.upconv_type == 'catk3':
                        self.upconv_module_voxel = upconv_module_catk3([concat_dim, self.in_dims[1], self.in_dims[0]], [128, 32, 32], final_channels=(self.sem_dim[-2], self.sem_dim[-1]))
                    else:
                        raise ValueError('Invalid upconv type.')
                else:
                    self.rpnconv = nn.Sequential(
                        convbn(concat_dim, self.sem_dim[0], 3, 1, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True),
                        convbn(self.sem_dim[0], self.sem_dim[1], 3, 1, 1, gn=cfg.GN),
                        nn.ReLU(inplace=True)
                    )

    def forward(self, feats, right=False):
        feat_shape = tuple(feats[self.start_level].shape[2:])
        assert len(feats) == len(self.in_dims)

        concat_features = feats[self.start_level:]
        if self.with_spp:
            spp_branches = []
            for branch_module in self.spp_branches:
                x = branch_module(feats[-1])
                x = F.interpolate(
                    x, feat_shape,
                    mode='bilinear',
                    align_corners=True)
                spp_branches.append(x)
            concat_features.extend(spp_branches)

        concat_feature = torch.cat(concat_features, 1)
        stereo_feature = concat_feature

        if not self.drop_psv:
            if self.with_upconv:
                if self.share_upconv:
                    stereo_feature, sem_feature = self.upconv_module([stereo_feature, feats[1], feats[0]])
                    return stereo_feature, sem_feature
                else:
                    stereo_feature = self.upconv_module([stereo_feature, feats[1], feats[0]])

            stereo_feature = self.lastconv(stereo_feature)

        if self.with_upconv_voxel and (self.cat_img_feature or self.with_sem_neck):
            sem_feature = self.upconv_module_voxel([concat_feature, feats[1], feats[0]])
        else:
            sem_feature = concat_feature
    
        if not self.with_upconv_voxel or self.upconv_type == 'fpn':
            if self.cat_img_feature or self.with_sem_neck:
                if self.extra_sem and not right:
                    extra_sem_feature = self.extra_rpnconv(sem_feature)
                sem_feature = self.rpnconv(sem_feature)
                if self.extra_sem and not right:
                    return stereo_feature, (sem_feature, extra_sem_feature)
            else:
                sem_feature = None

        return stereo_feature, sem_feature
