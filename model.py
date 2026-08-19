 #  67: VGG16 + U-Net


import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class VGG16_UNet(nn.Module):

    def __init__(self, out_channels=1):

        super().__init__()

        # ====================================
        # PRETRAINED VGG16
        # ====================================

        vgg = vgg16(
            weights=VGG16_Weights.DEFAULT
        )

        features = vgg.features


        # ====================================
        # VGG16 ENCODER BLOCKS
        # ====================================

        self.enc1 = nn.Sequential(
            *features[0:4]
        )
        # Output: 64 channels


        self.enc2 = nn.Sequential(
            *features[5:9]
        )
        # Output: 128 channels


        self.enc3 = nn.Sequential(
            *features[10:16]
        )
        # Output: 256 channels


        self.enc4 = nn.Sequential(
            *features[17:23]
        )
        # Output: 512 channels


        self.enc5 = nn.Sequential(
            *features[24:30]
        )
        # Output: 512 channels


        # ====================================
        # POOLING
        # ====================================

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # ====================================
        # BOTTLENECK
        # ====================================

        self.bottleneck = DoubleConv(
            512,
            512
        )


        # ====================================
        # DECODER 5
        # ====================================

        self.up5 = nn.ConvTranspose2d(
            512,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec5 = DoubleConv(
            1024,
            512
        )


        # ====================================
        # DECODER 4
        # ====================================

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            768,
            256
        )


        # ====================================
        # DECODER 3
        # ====================================

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            384,
            128
        )


        # ====================================
        # DECODER 2
        # ====================================

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            192,
            64
        )


        # ====================================
        # DECODER 1
        # ====================================

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            96,
            32
        )


        # ====================================
        # FINAL SEGMENTATION HEAD
        # ====================================

        self.final = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1
        )


    def forward(self, x):

        # ====================================
        # ENCODER
        # ====================================

        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        e5 = self.enc5(p4)
        p5 = self.pool(e5)


        # ====================================
        # BOTTLENECK
        # ====================================

        b = self.bottleneck(p5)


        # ====================================
        # DECODER + SKIP CONNECTIONS
        # ====================================

        d5 = self.up5(b)

        d5 = torch.cat(
            [d5, e5],
            dim=1
        )

        d5 = self.dec5(d5)


        d4 = self.up4(d5)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)


        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)


        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)


        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)


        # ====================================
        # OUTPUT
        # ====================================

        out = self.final(d1)

        return out