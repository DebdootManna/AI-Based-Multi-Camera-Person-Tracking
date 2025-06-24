"""
OSNet implementation for person re-identification.
Lightweight implementation of OSNet for ReID.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

__all__ = ['osnet_x1_0']

model_urls = {
    'osnet_x1_0': 'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY',
    'osnet_x0_75': 'https://drive.google.com/uc?id=1uwA9fElHqjt4zLClHrsVadscbRtv9Tqt',
    'osnet_x0_5': 'https://drive.google.com/uc?id=16DGLbZukvVYgINws8uUhdeJUMxoWPSKj',
    'osnet_x0_25': 'https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs',
    'osnet_ibn_x1_0': 'https://drive.google.com/uc?id=1sSwDLI4fce-57jD7b8Q1w09rQH1hLTlE',
}


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                             padding, bias=False, groups=groups)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Conv1x1(nn.Module):
    """1x1 convolution + batch norm + ReLU."""
    
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0,
            bias=False, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""
    
    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )
    
    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""
    
    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = ConvLayer(mid_channels, mid_channels, 1)
        self.conv2b = ConvLayer(mid_channels, mid_channels, 3, padding=1)
        self.conv2c = ConvLayer(mid_channels, mid_channels, (1, 3), padding=(0, 1))
        self.conv2d = ConvLayer(mid_channels, mid_channels, (3, 1), padding=(1, 0))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1(mid_channels, out_channels, stride=1)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1(in_channels, out_channels, stride=1)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)
    
    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class OSNet(nn.Module):
    """Omni-Scale Network."""
    
    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, 
                 loss='softmax', IN=False, **kwargs):
        super(OSNet, self).__init__()
        
        self.num_classes = num_classes
        self.loss = loss
        
        # Convolutional layers
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True,
            IN=IN
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=True,
            IN=IN
        )
        self.conv5 = Conv1x1(channels[3], feature_dim)
        
        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Initialize weights
        self._init_params()
    
    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, IN):
        layers = []
        
        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))
        
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )
        
        return nn.Sequential(*layers)
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
    
    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        
        x = self.conv5(x)
        v = F.adaptive_avg_pool2d(x, 1)
        v = v.view(v.size(0), -1)
        
        if not self.training:
            return v
            
        y = self.classifier(v)
        
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def osnet_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    """
    OSNet model with width multiplier x1.0
    """
    model = OSNet(
        num_classes=num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs
    )
    
    if pretrained:
        # Load pretrained weights
        model_url = model_urls['osnet_x1_0']
        try:
            state_dict = load_state_dict_from_url(model_url, progress=True)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
    
    return model


def osnet_x0_75(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    """
    OSNet model with width multiplier x0.75
    """
    model = OSNet(
        num_classes=num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[48, 192, 288, 384],
        loss=loss,
        **kwargs
    )
    
    if pretrained:
        # Load pretrained weights
        model_url = model_urls['osnet_x0_75']
        try:
            state_dict = load_state_dict_from_url(model_url, progress=True)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
    
    return model


def osnet_x0_5(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    """
    OSNet model with width multiplier x0.5
    """
    model = OSNet(
        num_classes=num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
        loss=loss,
        **kwargs
    )
    
    if pretrained:
        # Load pretrained weights
        model_url = model_urls['osnet_x0_5']
        try:
            state_dict = load_state_dict_from_url(model_url, progress=True)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
    
    return model


def osnet_x0_25(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    """
    OSNet model with width multiplier x0.25
    """
    model = OSNet(
        num_classes=num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        loss=loss,
        **kwargs
    )
    
    if pretrained:
        # Load pretrained weights
        model_url = model_urls['osnet_x0_25']
        try:
            state_dict = load_state_dict_from_url(model_url, progress=True)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
    
    return model
