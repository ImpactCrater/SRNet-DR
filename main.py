#! /usr/bin/python3
# -*- coding: utf8 -*-

import os, time, random, re, glob
from pathlib import Path
import math
import random
import numpy as np
from PIL import Image, ImageMath, ImageFilter, ImageOps
from io import BytesIO
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import pytorch_ssim
from config import config
# from torchsummary import summary


###====================== HYPER-PARAMETERS ===========================###
## File Format
saveFileFormat = config.saveFileFormat

## Mini Batch
miniBatchSize = config.miniBatchSize

## Learning Rate of RAdam
learningRate = config.learningRate

## Weight Decay of RAdam
weightDecay = config.weightDecay

## Training
nEpoch = config.nEpoch

## Number of Iterations of the Step to Save
nIterationOfStepToSave = config.nIterationOfStepToSave

## Paths
samplesPath = config.samplesPath
checkpointPath = config.checkpointPath
validationHRImagePath = config.validationHRImagePath
trainingHRImagePath = config.trainingHRImagePath
lRImagePath = config.lRImagePath





class ImageFromDirectory(Dataset):
    imageExtensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"] # 拡張子のリストをクラス変数として宣言。

    def __init__(self, imageDirectory, mode):
        if imageDirectory == False:
            self.imageDirectory = os.getcwd()
        else:
            self.imageDirectory = imageDirectory

        self.mode = mode

        # 画像ファイルのパスのリストを取得する。
        self.imagePathsList = self._getImagePaths()

    def __getitem__(self, index):
        # index 番目のデータが要求された時にそれを返す。
        path = self.imagePathsList[index]

        # 画像ファイルをPillow形式で読み込む。
        image = Image.open(path)

        transformToTensor = transforms.Compose([transforms.ToTensor()])
        if self.mode == "training" or self.mode == "validation":
            # 画像データに変換を施す。
            imageHR, imageLR = self._preprocess(image)
            imageHR = transformToTensor(imageHR)
            imageLR = transformToTensor(imageLR)

            return imageHR, imageLR

        elif self.mode == "sr":
            if image.width % 4 == 0:
                newWidth = image.width
            else:
                newWidth = 4 * (image.width // 4 + 1)

            if image.height % 4 == 0:
                newHeight = image.height
            else:
                newHeight = 4 * (image.height // 4 + 1)

            newImage = Image.new(image.mode, (newWidth, newHeight), (0,0,0))
            newImage.paste(image, (0, 0))
            imageLR = transformToTensor(newImage)

            return imageLR

    def updateDataList(self):
        # 画像ファイルのパスのリストを新たに取得する。
        self.imagePathsList = self._getImagePaths()

    def _getImagePaths(self):
        # 指定したディレクトリー内の画像ファイルのパスのリストを取得する。
        imageDirectory = Path(self.imageDirectory)
        imagePathsList = [
            p for p in sorted(imageDirectory.glob("**/*")) if p.suffix in ImageFromDirectory.imageExtensions]

        return imagePathsList

    def _preprocess(self, image):
        # Data augmentation
             # x = random.randint(a, b); a <= x <= b (x; int)
             # x = random.uniform(a, b); a <= x <= b (x; float)
        minSize = image.width if image.width < image.height else image.height
        randomSize = random.randint(388, minSize)
        left = random.randint(0, image.width - randomSize)
        top = random.randint(0, image.height - randomSize)
        right = left + randomSize
        bottom = top + randomSize
        imageHR = image.crop((left, top, right, bottom))
        imageHR = ImageOps.autocontrast(image, 0.01) # auto contrast, 0.01% cut-off
        imageHR = imageHR.filter(ImageFilter.UnsharpMask(radius=0.5, percent=400, threshold=0))
        imageHR = imageHR.resize((388, 388), Image.BICUBIC)
        h, s, v = imageHR.convert("HSV").split()
        randomValue = random.randint(-16, 16)
        hShifted = h.point(lambda x: (x + randomValue) % 255 if (x + randomValue) % 255 >= 0 else 255 - (x + randomValue)) # rotate hue
        imageHR = Image.merge("HSV", (hShifted, s, v)).convert("RGB")
        if random.randint(0, 1) == 1:
            imageHR = ImageOps.mirror(imageHR)

        # Deterioration
        imageLR = imageHR.copy()
        randomSize = math.floor(random.uniform(0.0, 1.0) * random.uniform(0.0, 1.0) * random.uniform(0.0, 1.0) * random.uniform(0.0, 1.0) * (388 - 97) + 97)
        imageLR = imageLR.resize((randomSize, randomSize), Image.BICUBIC)
        randomRadius = random.uniform(0.0, 1.0)
        imageLR = imageLR.filter(ImageFilter.GaussianBlur(randomRadius))

        randomStrength = random.uniform(0.0, 0.1)
        width, height = imageLR.size
        r, g, b = imageLR.split()
        noiseImage = Image.effect_noise((width, height), 255) # Generate Gaussian noise
        r = Image.blend(r, noiseImage, randomStrength)
        noiseImage = Image.effect_noise((width, height), 255)
        g = Image.blend(g, noiseImage, randomStrength)
        noiseImage = Image.effect_noise((width, height), 255)
        b = Image.blend(b, noiseImage, randomStrength)
        imageLR = Image.merge("RGB", (r, g, b))

        randomQuality = random.randint(5, 100)
        imageFile = BytesIO()
        imageLR.save(imageFile, 'webp', quality=randomQuality)
        imageLR = Image.open(imageFile)
        imageLR = imageLR.resize((388, 388), Image.BICUBIC)

        left = random.randint(0, 3)
        top = random.randint(0, 3)
        right = left + 384
        bottom = top + 384
        imageHR = imageHR.crop((left, top, right, bottom))
        imageLR = imageLR.crop((left, top, right, bottom))
        return imageHR, imageLR

    def __len__(self):
        # len(datasetインスタンス)でディレクトリー内の画像ファイルの数を返す。

        return len(self.imagePathsList)






# TanhExp activation function.
class TanhExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
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






class Model(torch.nn.Module):
    def __init__(self):
        super().__init__() # 親クラスである torch.nn.Module の __init__ を継承する。
        # weightの初期化はKaiming Heの初期化で自動的に成される。

        nChannels1 = 384
        nChannels2 = 384
        nGroups1 = 1
        nGroups2 = 1
        self.nResidualBlocks1 = 8 # 8
        self.nResidualBlocks2 = 16 # 16
        index = 0
        layersList = []

        layersList.append(
            torch.nn.PixelUnshuffle(downscale_factor=4))

        layersList.append(
            torch.nn.Conv2d(in_channels=3 * 4 * 4, out_channels=nChannels1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='replicate'))

        layersList.append(
            torch.nn.GroupNorm(num_groups=int(nChannels1 / 24), num_channels=nChannels1, eps=1e-05, affine=True))

        layersList.append(TanhExp())

        # Residual Blocks
        for j in range(self.nResidualBlocks2):
            for i in range(self.nResidualBlocks1):
                layersList.append(
                    torch.nn.Conv2d(in_channels=nChannels1, out_channels=nChannels2, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=nGroups1, bias=True, padding_mode='replicate'))

                layersList.append(TanhExp())

                layersList.append(
                    torch.nn.Conv2d(in_channels=nChannels2, out_channels=nChannels1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=nGroups2, bias=True, padding_mode='replicate'))

                layersList.append(TanhExp())

            layersList.append(
                torch.nn.Conv2d(in_channels=nChannels1, out_channels=nChannels1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

            layersList.append(TanhExp())

        layersList.append(
            torch.nn.Conv2d(in_channels=nChannels1, out_channels=nChannels1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

        layersList.append(TanhExp())
        # Residual Blocks end

        layersList.append(
            torch.nn.Conv2d(in_channels=nChannels1, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

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

        layersList.append(torch.nn.Sigmoid())

        self.layers = torch.nn.ModuleList(layersList)

    def forward(self, x):
        i = 0

        x = self.layers[i](x) # PixelShuffle
        i += 1
        x = self.layers[i](x) # Conv2d
        i += 1
        x = self.layers[i](x) # GroupNorm
        i += 1
        x = self.layers[i](x) # TanhExp
        i += 1

        # Residual Blocks
        x1 = x
        for k in range(self.nResidualBlocks2):
            x0 = x
            for j in range(self.nResidualBlocks1):
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
            x = x + x0
        x = self.layers[i](x) # Conv2d
        i += 1
        x = self.layers[i](x) # TanhExp
        i += 1
        x = x + x1
        # Residual Blocks end

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
        x = self.layers[i](x) # Sigmoid

        return x





def train():
    print("Checking the existence of the directory for saving generated images.")
    saveDirectoryGenerated = samplesPath + "generated"
    if os.path.isdir(saveDirectoryGenerated):
        print("O.K.")
    else:
        print("Making the directory.")
        os.mkdir(saveDirectoryGenerated)

    print("Now processing...")
    # GPUが利用可能ならGPUを利用する。
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark=True
    else:
        device = "cpu"

    # モデルのインスタンスを作成する。
    model = Model()
    if os.path.isfile(checkpointPath + "model.pth"):
        model.load_state_dict(torch.load(checkpointPath + "model.pth", map_location=torch.device(device)))
    model = model.to(device)

    # オプティマイザーを作成する。
    optimizer = torch.optim.RAdam(params=model.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weightDecay)

    # 損失関数のインスタンスを作成する。
    lossFunction = pytorch_ssim.SSIM(window_size=11) # torch.nn.Moduleを継承している。

    # 評価用画像の Dataset を作成する。
    datasetValidation = ImageFromDirectory(validationHRImagePath, "validation")

    # 評価用画像の DataLoader を作成する。
    dataloaderValidation = DataLoader(datasetValidation, batch_size=miniBatchSize, pin_memory=True, shuffle=False, num_workers=0, drop_last=False)

    # 評価用画像を保存する。
    miniBatchLRList = []
    i = 0
    for miniBatchHR, miniBatchLR in dataloaderValidation:
        utils.save_image(miniBatchLR, saveDirectoryGenerated + "/" + str(i) + "-Reference0LR.png", nrow=16)
        utils.save_image(miniBatchHR, saveDirectoryGenerated + "/" + str(i) + "-Reference1HR.png", nrow=16)
        miniBatchLRList.append(miniBatchLR)
        i += 1

    # 学習用画像の Dataset を作成する。
    datasetTrain = ImageFromDirectory(trainingHRImagePath, "training")

    # 学習用画像の DataLoader を作成する。
    dataloaderTraining = DataLoader(datasetTrain, batch_size=miniBatchSize, pin_memory=True, shuffle=True, num_workers=0, drop_last=True)
    for epoch in range(0, nEpoch):
        epochTime = time.time()
        totalLoss, step = 0, 0

        datasetTrain.updateDataList() # データセットのリストを更新する。

        nImagesTrain = len(datasetTrain)
        nStep = math.floor(nImagesTrain / miniBatchSize)
        print("The dataset has been updated.")
        print("Number of Images: {} Number of Steps: {}".format(nImagesTrain, nStep))

        i = 0
        previousTime = time.time()
        for miniBatchHR, miniBatchLR in dataloaderTraining:
            miniBatchHR = miniBatchHR.to(device)
            miniBatchLR = miniBatchLR.to(device)
            model.train() # training モードに設定する。
            miniBatchGenerated = model(miniBatchLR) # 画像データをモデルに入力する。
            del miniBatchLR
            loss = lossFunction(miniBatchGenerated, miniBatchHR) # 生成画像と正解画像との間の損失を計算させる。
            del miniBatchHR
            del miniBatchGenerated
            optimizer.zero_grad(set_to_none=True) # 勾配を削除により初期化する。
            loss.backward() # 誤差逆伝播により勾配を計算させる。
            optimizer.step() # パラメーターを更新させる。

            totalLoss += float(loss)
            nowTime = time.time()
            print("Epoch: {:2d} Step: {:4d} Time: {:4.2f} Loss: {:.8f}".format(
                  epoch, step, nowTime - previousTime, loss)) # 損失値を表示させる。
            del loss
            previousTime = nowTime

            step += 1

            # Validationを実行する。
            if i % nIterationOfStepToSave == 0:
                model.eval() # evaluation モードに設定する。
                with torch.no_grad(): # 以下のスコープ内では勾配計算をさせない。
                    j = 0
                    for miniBatchLR in miniBatchLRList:
                        miniBatchLR = miniBatchLR.to(device)
                        miniBatchGenerated = model(miniBatchLR) # 評価用画像データのミニバッチをモデルに入力する。
                        utils.save_image(miniBatchGenerated, saveDirectoryGenerated + "/" + str(j) + "-" + str(epoch) + "-" + str(i) + ".png", nrow=16)
                        torch.save(model.state_dict(), checkpointPath + "model.pth") # モデル データを保存する。
                        j += 1

            i += 1

        print("Epoch[{:2d}/{:2d}] Time: {:4.2f} loss: {:.8f}".format(
            epoch, nEpoch, time.time() - epochTime, totalLoss / nStep))








def sr():
    print("Now processing...")

    # GPUが利用可能ならGPUを利用する。
    if torch.cuda.is_available():
      device = "cuda"
    else:
      device = "cpu"

    # モデルのインスタンスを作成する。
    model = Model()
    if os.path.isfile(checkpointPath + "model.pth"):
        model.load_state_dict(torch.load(checkpointPath + "model.pth", map_location=torch.device('cpu')))
    model = model.to(device)

    # summary(model,(3,384,384))


    # 入力画像の Dataset を作成する。
    datasetSR = ImageFromDirectory(lRImagePath, "sr")

    # 入力画像の DataLoader を作成する。
    dataloaderSR = DataLoader(datasetSR, batch_size=1, pin_memory=True, shuffle=False, num_workers=0, drop_last=False)

    saveDirectoryGenerated = samplesPath + "sr"

    nImagesSR = len(datasetSR)
    nStep = math.floor(nImagesSR / miniBatchSize)
    print("Number of Images: {}".format(nImagesSR))

    i = 0
    for miniBatchLR in dataloaderSR:
        model.eval() # evaluation モードに設定する。
        with torch.no_grad(): # 以下のスコープ内では勾配計算をさせない。
            stepTime = time.time()
            miniBatchLR = miniBatchLR.to(device)
            miniBatchGenerated = model(miniBatchLR) # 画像データをモデルに入力する。
            utils.save_image(miniBatchGenerated, saveDirectoryGenerated + "/" + str(i) + ".png", nrow=1)

            i += 1
            print("{}: Time: {:4.2f}".format(i, time.time() - stepTime))

    print("Done.")





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, sr')

    args = parser.parse_args()

    mode = args.mode

    if mode == 'train':
        train()
    elif mode == 'sr':
        sr()
    else:
        raise Exception("Unknow --mode")
