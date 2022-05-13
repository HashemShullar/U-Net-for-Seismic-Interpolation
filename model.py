import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class Dialated_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dialated_Conv, self).__init__()

        if out_channels in [64, 128]:

            self.conv1 = nn.Conv2d(in_channels, int(out_channels/4), 3, stride=1, padding=1, dilation=1, padding_mode='reflect', bias=False)
            self.conv2 = nn.Conv2d(in_channels, int(out_channels/4), 3, stride=1, padding=2, dilation=2, padding_mode='reflect', bias=False)
            self.conv3 = nn.Conv2d(in_channels, int(out_channels/4), 3, stride=1, padding=4, dilation=4, padding_mode='reflect', bias=False)
            self.conv4 = nn.Conv2d(in_channels, int(out_channels/4), 3, stride=1, padding=6, dilation=6, padding_mode='reflect', bias=False)

        else:
            self.conv1 = nn.Conv2d(in_channels, int(out_channels/4), 3, stride=1, padding=1, dilation=1,  bias=False)
            self.conv2 = nn.Conv2d(in_channels, int(out_channels/4), 3, stride=1, padding=1, dilation=1,  bias=False)
            self.conv3 = nn.Conv2d(in_channels, int(out_channels/4), 3, stride=1, padding=1, dilation=1,  bias=False)
            self.conv4 = nn.Conv2d(in_channels, int(out_channels/4), 3, stride=1, padding=1, dilation=1,  bias=False)


    def forward(self, x):

        concat_conv0  = self.conv1(x)
        concat_conv1  = self.conv2(x)
        concat_conv2  = self.conv3(x)
        concat_conv3  = self.conv4(x)

        concat_conv  = torch.cat((concat_conv0, concat_conv1, concat_conv2, concat_conv3), dim=1)

        return concat_conv





class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()

        self.contract = nn.ModuleList()
        self.expand   = nn.ModuleList()

        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # Contracting part
        for feature in features:
            self.contract.append(nn.Sequential(
            Dialated_Conv(in_channels, feature),
            nn.BatchNorm2d(feature),
            nn.ReLU(inplace=True),
            Dialated_Conv(feature, feature),
            nn.BatchNorm2d(feature),
            nn.ReLU(inplace=True)))


            in_channels = feature

        # Expanding part
        for feature in reversed(features):

            self.expand.append(nn.Conv2d(2*feature, feature, 3, 1, 1, bias=False))

            self.expand.append(nn.Sequential(
            nn.Conv2d(2*feature, feature, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feature),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature, feature, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feature),
            nn.ReLU(inplace=True)))




        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], 2*features[-1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(2*features[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*features[-1], 2*features[-1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(2*features[-1]),
            nn.ReLU(inplace=True)
        )


        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        indicies         = []

        for down in self.contract:
            x = down(x)
            skip_connections.append(x)
            x, ids = self.pool(x)
            indicies.append(ids)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        indicies         = indicies[::-1]

        for idx in range(0, len(self.expand), 2):
            x = self.expand[idx](x)
            x = self.unpool(x, indicies[idx//2])
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.expand[idx+1](concat_skip)

        return self.out(x)

