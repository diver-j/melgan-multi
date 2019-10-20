import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d
from torch.nn.utils import weight_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(in_channels, out_channels, 3, 1, dilation=1,
                               padding=get_padding(3, 1))),
            weight_norm(Conv1d(in_channels, out_channels, 3, 1, dilation=3,
                               padding=get_padding(3, 3))),
            weight_norm(Conv1d(in_channels, out_channels, 3, 1, dilation=9,
                               padding=get_padding(3, 9))),
        ])

    def forward(self, x):
        for l in self.convs:
            x_p = x
            x = l(x)
            x = F.leaky_relu(x)
            x = x + x_p
        return x


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_pre = weight_norm(Conv1d(80, 512, 7, 1, padding=3))
        self.ups = nn.ModuleList([
            weight_norm(ConvTranspose1d(512, 256, 16, 8, padding=4)),
            weight_norm(ConvTranspose1d(256, 128, 16, 8, padding=4)),
            weight_norm(ConvTranspose1d(128, 64, 4, 2, padding=1)),
            weight_norm(ConvTranspose1d(64, 32, 4, 2, padding=1))
        ])
        self.resblocks = nn.ModuleList([
            ResBlock(256, 256),
            ResBlock(128, 128),
            ResBlock(64, 64),
            ResBlock(32, 32)
        ])
        self.conv_post = weight_norm(Conv1d(32, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(4):
            x = self.ups[i](x)
            x = self.resblocks[i](x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_pre = weight_norm(Conv1d(1, 16, 15, 1, padding=7))
        self.grouped_convs = nn.ModuleList([
            weight_norm(Conv1d(16, 64, 41, 1, groups=4, padding=20)),
            weight_norm(Conv1d(64, 256, 41, 1, groups=16, padding=20)),
            weight_norm(Conv1d(256, 1024, 41, 1, groups=64, padding=20)),
            weight_norm(Conv1d(1024, 1024, 41, 1, groups=256, padding=20)),
        ])
        self.conv_post1 = weight_norm(Conv1d(1024, 1024, 5, 1, padding=2))
        self.conv_post2 = weight_norm(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        x = self.conv_pre(x)
        x = F.leaky_relu(x)
        fmap.append(x)
        for l in self.grouped_convs:
            x = l(x)
            x = F.leaky_relu(x)
            fmap.append(x)
        x = self.conv_post1(x)
        fmap.append(x)
        x = self.conv_post2(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            Discriminator(),
            Discriminator(),
            Discriminator(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=get_padding(4)),
            AvgPool1d(4, 4, padding=get_padding(4))
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i == 0:
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
            else:
                y_d_r, fmap_r = d(self.meanpools[i-1](y))
                y_d_g, fmap_g = d(self.meanpools[i-1](y_hat))
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*10


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        real_loss = torch.mean((torch.ones_like(dg[0]) - dr[0])**2)
        generated_loss = torch.mean((dg[0])**2)
        total_disc_loss = real_loss + generated_loss
        loss += total_disc_loss
        r_losses.append(real_loss)
        g_losses.append(generated_loss)

    return loss, r_losses, g_losses


def generator_loss(disc_generated_outputs):
    loss = 0
    for dg in disc_generated_outputs:
        loss += torch.mean((torch.ones_like(dg[0]) - dg[0])**2)

    return loss

