#! /usr/bin/python3
# -*- coding: utf8 -*-

import os, time, random, re, glob
from os.path import expanduser
from pathlib import Path
import math
import random
import numpy
from PIL import Image, ImageMath, ImageFilter, ImageOps
from io import BytesIO
import torch
import torch._dynamo
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
import cv2

# from torchsummary import summary






# TanhExp activation function.
class TanhExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            result = torch.where(x > 20, x, x * torch.tanh(torch.exp(x)))
            ctx.save_for_backward(x)
            return result

    @staticmethod
    def backward(ctx, grad_output):
        # GPUが利用可能ならGPUを利用する。
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        x = ctx.saved_tensors[0]
        one = torch.tensor([1.0], device=torch.device(device))
        x = torch.where(x > 20, one, torch.tanh(torch.exp(x)) - x * torch.exp(x) * (torch.square(torch.tanh(torch.exp(x))) - 1.0))
        return grad_output * x

class TanhExp(torch.nn.Module):
    def forward(self, x):
        return TanhExpFunction.apply(x)





class ModelOfGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__() # 親クラスである torch.nn.Module の __init__ を継承する。
        # weightの既定の初期化はKaiming Heの初期化。

        numberOfChannels1 = 64 # recommended value: 480
        numberOfChannels2 = 64 # recommended value: 384
        numberOfGroups = 1
        self.numberOfResidualBlocks1 = 8 # recommended value: 20
        self.numberOfResidualBlocks2 = 8 # recommended value: 8
        index = 0
        layersList = []

        # Input
        layersList.append(
            torch.nn.PixelUnshuffle(downscale_factor=4))

        layersList.append(
            torch.nn.Conv2d(in_channels=3 * 4 * 4, out_channels=numberOfChannels1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='replicate'))

        # Normalizationは入力データを平均0、標準偏差1となるように変換する。従って、出力には負の値及び1.0より大きな値を含む。これにより入力データの情報利用効率を上げる。
        # Normalizationは学習可能なパラメーターgamma, betaを持つ。gammaはスケール、betaはバイアスである。
        # affine=Trueでパラメーターを学習する。この時、直前の層ではbias=Falseとする。
        layersList.append(
            torch.nn.GroupNorm(num_groups=int(numberOfChannels1 / (3 * 4 * 4)), num_channels=numberOfChannels1, eps=1e-05, affine=True))

        layersList.append(TanhExp())


        # Residual Blocks
        for i in range(self.numberOfResidualBlocks1):
            for j in range(self.numberOfResidualBlocks2):
                layersList.append(
                    torch.nn.Conv2d(in_channels=numberOfChannels1, out_channels=numberOfChannels2, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=numberOfGroups, bias=True, padding_mode='replicate'))

                layersList.append(TanhExp())

                layersList.append(
                    torch.nn.Conv2d(in_channels=numberOfChannels2, out_channels=numberOfChannels1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=numberOfGroups, bias=True, padding_mode='replicate'))

                layersList.append(TanhExp())

            layersList.append(
                torch.nn.Conv2d(in_channels=numberOfChannels1, out_channels=numberOfChannels1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=numberOfGroups, bias=True, padding_mode='replicate'))

            layersList.append(TanhExp())

        layersList.append(
            torch.nn.Conv2d(in_channels=numberOfChannels1, out_channels=numberOfChannels1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=numberOfGroups, bias=True, padding_mode='replicate'))

        layersList.append(TanhExp())
        # Residual Blocks end


        # Output
        layersList.append(
            torch.nn.Conv2d(in_channels=numberOfChannels1, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

        layersList.append(
            torch.nn.PixelShuffle(upscale_factor=2))

        layersList.append(TanhExp())

        layersList.append(
            torch.nn.Conv2d(in_channels=int(512 / 4), out_channels=512, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

        layersList.append(
            torch.nn.PixelShuffle(upscale_factor=2))

        layersList.append(TanhExp())

        layersList.append(
            torch.nn.Conv2d(in_channels=int(512 / 4), out_channels=3, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='replicate'))

        self.layers = torch.nn.ModuleList(layersList)
        self.layers.apply(self._initializeWeights)


    def forward(self, x):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            i = 0

            # Input
            xOrigin = x
            x = self.layers[i](x) # PixelUnshuffle
            i += 1
            x = self.layers[i](x) # Conv2d
            i += 1
            x = self.layers[i](x) # GroupNorm
            i += 1
            x = self.layers[i](x) # TanhExp
            i += 1


            # Residual Blocks
            x0 = x
            for j in range(self.numberOfResidualBlocks1):
                x1 = x
                for j in range(self.numberOfResidualBlocks2):
                    h = self.layers[i](x) # Conv2d
                    i += 1
                    h = self.layers[i](h) # TanhExp
                    i += 1
                    h = self.layers[i](h) # Conv2d
                    i += 1
                    h = self.layers[i](h) # TanhExp
                    i += 1
                    x = x + h
                x = self.layers[i](x) # Conv2d
                i += 1
                x = self.layers[i](x) # TanhExp
                i += 1
                x = x + x1
            x = self.layers[i](x) # Conv2d
            i += 1
            x = self.layers[i](x) # TanhExp
            i += 1
            x = x + x0
            # Residual Blocks end


            # Output
            x = self.layers[i](x) # Conv2d
            i += 1
            x = self.layers[i](x) # PixelShuffle
            i += 1
            x = self.layers[i](x) # TanhExp
            i += 1
            x = self.layers[i](x) # Conv2d
            i += 1
            x = self.layers[i](x) # PixelShuffle
            i += 1
            x = self.layers[i](x) # TanhExp
            i += 1
            x = self.layers[i](x) # Conv2d
            i += 1
            x = x + xOrigin

            return x


    def _initializeWeights(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
             #torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu', generator=None) # default.
             torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(16), mode='fan_in', nonlinearity='leaky_relu', generator=None)
             #m.bias.data.fill_(0.01)
