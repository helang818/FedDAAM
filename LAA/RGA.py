import torch
from torch import nn

class RGA(nn.Module):
    def __init__(self, in_ch, in_sp, spa_ratio=8):
        super(RGA, self).__init__()
        self.in_ch = in_ch
        self.in_sp = in_sp
        # self.use_spatial = use_spatial

        # self.inter_sp = in_sp // spa_ratio
        self.inter_sp = in_sp
        self.inter_ch = in_ch // spa_ratio
        # Embedding functions for original features

        self.gx_spatial = nn.Sequential(
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.inter_ch,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.inter_ch),
            nn.ReLU()
        )

        # Embedding functions for relation features

        self.gg_spatial = nn.Sequential(
            nn.Conv3d(in_channels=self.in_sp * 2, out_channels=self.inter_sp,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.inter_sp),
            nn.ReLU()
        )


        self.theta_spatial = nn.Sequential(
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.inter_ch,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.inter_ch),
            nn.ReLU()
        )
        self.phi_spatial = nn.Sequential(
            nn.Conv3d(in_channels=self.in_ch, out_channels=self.inter_ch,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.inter_ch),
            nn.ReLU()
        )


    def forward(self, g1, g2, debug=False):
        # b, c, h, w = x.size()
        b, c, t, h, w = g1.size()
        # spatial attention
        # 1.降低通道数，生成转置矩阵，即b, c, h, w-> b,h*w, c
        theta_xs = self.theta_spatial(g1)  # 降低通道数1
        if debug:
            print('theta_xs.shape', theta_xs.shape)
        phi_xs = self.phi_spatial(g2)  # 降低通道数2
        if debug:
            print('phi_xs.shape', phi_xs.shape)
        # 2.降低通道数，将b, c, t,h, w->b, c,t, h*w->b, t,c, h*w
        theta_xs = theta_xs.view(b, self.inter_ch, t, -1)  # 通过view，将b, c,t, h, w->b, c,t, h*w
        if debug:
            print('theta_xs.shape', theta_xs.shape)
        theta_xs = theta_xs.permute(0, 2, 3, 1)  # 将b, c,t, h*w -> b,t,h*w, c  # 转置矩阵，也就是T
        if debug:
            print('theta_xs.shape', theta_xs.shape)
        phi_xs = phi_xs.view(b, self.inter_ch, t, -1)
        phi_xs = phi_xs.permute(0, 2, 1, 3)
        if debug:
            print('phi_xs.shape', phi_xs.shape)

        # 3.计算相关系数
        Gs = torch.matmul(theta_xs, phi_xs)  # 计算相关系数
        if debug:
            print('Gs.shape', Gs.shape)
        # 4.改变形状，生成每个特征对别的特征的影响系数矩阵Gs_out和受别的特征影响系数矩阵Gs_joint
        Gs_in = Gs.permute(0, 3, 1, 2).view(b, h * w, t, h, w)
        if debug:
            print('Gs_in.shape', Gs_in.shape)

        Gs_out = Gs.permute(0, 2, 1, 3).view(b, h * w, t, h, w)
        if debug:
            print('Gs_out.shape', Gs_out.shape)

        # 5.拼接相关矩阵，并降维
        Gs_joint = torch.cat((Gs_in, Gs_out), 1)  # 拼接相关矩阵
        if debug:
            print('Gs_joint.shape', Gs_joint.shape)
        Gs_joint = self.gg_spatial(Gs_joint)  # 通道降维
        if debug:
            print('Gs_joint.shape', Gs_joint.shape)
        # Gs_joint = torch.sigmoid(Gs_joint)
        return Gs_joint



class RGA_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, s_ratio=8, d_ratio=8):
        super(RGA_Module, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial
        # self.use_spatial = use_spatial
        self.RGA = RGA(self.in_channel, self.in_spatial//4, spa_ratio=s_ratio)

        num_channel_s = 3*self.in_spatial //4
        # print(num_channel_s)
        # print(num_channel_s // down_ratio)
        self.W_spatial = nn.Sequential(
            nn.Conv3d(in_channels=num_channel_s, out_channels=num_channel_s // d_ratio,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(num_channel_s // d_ratio),
            nn.ReLU(),
            nn.Conv3d(in_channels=num_channel_s // d_ratio, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(1)  # TODO
        )

    def forward(self, x, debug=False):

        _, _, _, H, W = x.size()

        # 计算每个部分的新空间维度大小
        new_H = H // 2
        new_W = W // 2

        # 使用张量切片操作划分成四个部分
        part1 = x[:, :, :, :new_H, :new_W]
        part2 = x[:, :, :, :new_H, new_W:]
        part3 = x[:, :, :, new_H:, :new_W]
        part4 = x[:, :, :, new_H:, new_W:]

        g1= torch.cat((self.RGA(part1, part2),
                       self.RGA(part1, part3),
                       self.RGA(part1, part4)), 1)
        if debug:
            print('g1',g1.shape)
        g2 = torch.cat((self.RGA(part2, part1),
                        self.RGA(part2, part3),
                        self.RGA(part2, part4)), 1)
        if debug:
            print('g2', g2.shape)
        g3 = torch.cat((self.RGA(part3, part1),
                        self.RGA(part3, part2),
                        self.RGA(part3, part4)), 1)
        if debug:
            print('g3', g3.shape)
        g4 = torch.cat((self.RGA(part4, part1),
                        self.RGA(part4, part2),
                        self.RGA(part4, part3)), 1)
        if debug:
            print('g4', g4.shape)
        # att1
        W_g1 = self.W_spatial(g1)
        part1 = torch.sigmoid(W_g1.expand_as(part1)) * part1
        if debug:
            print('W_ys.shape', W_g1.shape)# 权重维度拓展到和X一致再相乘
            print('out.shape', part1.shape)
        # att2
        W_g2 = self.W_spatial(g2)
        part2 = torch.sigmoid(W_g2.expand_as(part2)) * part2
        if debug:
            print('W_ys.shape', W_g2.shape)  # 权重维度拓展到和X一致再相乘
            print('out.shape', part2.shape)
        W_g3 = self.W_spatial(g3)
        part3 = torch.sigmoid(W_g3.expand_as(part3)) * part3
        if debug:
            print('W_ys.shape', W_g3.shape)  # 权重维度拓展到和X一致再相乘
            print('out.shape', part3.shape)
        W_g4 = self.W_spatial(g4)
        part4 = torch.sigmoid(W_g4.expand_as(part4)) * part4
        if debug:
            print('W_ys.shape', W_g4.shape)  # 权重维度拓展到和X一致再相乘
            print('out.shape', part4.shape)
        part13 = torch.cat((part1, part3), dim=-2)
        part24 = torch.cat((part2, part4), dim=-2)
        merged_tensor = torch.cat((part13, part24), dim=-1)

        return merged_tensor


if __name__ == '__main__':
    import torch

    s_ratio = 8
    c_ratio = 8
    d_ratio = 8
    height = 112
    width = 112
    # model = RGA_Module(128, (height // 4) * (width // 4),  s_ratio=s_ratio, d_ratio=d_ratio)
    # # print(model)
    # input = torch.randn(8, 128, 8, 32, 32)
    # model = RGA_Module(256, (height // 8) * (width // 8), s_ratio=s_ratio, d_ratio=d_ratio)
    # input = torch.randn(8, 256, 8, 16, 16)
    model = RGA_Module(512, (height // 8) * (width // 8), s_ratio=s_ratio, d_ratio=d_ratio)
    input = torch.randn(8, 512, 8, 8, 8)
    output = model(input)
    print(output.shape)
