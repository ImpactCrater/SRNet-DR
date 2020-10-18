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


###====================== HYPER-PARAMETERS ===========================###
## File Format
saveFileFormat = config.saveFileFormat

## Mini Batch
miniBatchSize = config.miniBatchSize

## Adam
learningRate = config.learningRate

## Training
nEpoch = config.nEpoch
noiseLevel = config.noiseLevel

## Paths
samplesPath = config.samplesPath
checkpointPath = config.checkpointPath
weightFilePath = os.path.join(checkpointPath, 'g.h5')
validationHRImagePath = config.validationHRImagePath
trainingHRImagePath = config.trainingHRImagePath
evaluationImagePath = config.evaluationImagePath
enlargementLRImagePath = config.enlargementLRImagePath





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
            imageHR = self._preprocess(image)
            imageLR = self._downsampleAndDeteriorate(imageHR)
            imageHR = transformToTensor(imageHR)
            imageLR = transformToTensor(imageLR)

            return imageHR, imageLR

        elif self.mode == "evaluation":
            imageHR = self._enlarge(image)
            imageHR = transformToTensor(imageHR)
            imageLR = transformToTensor(image)

            return imageHR, imageLR

        elif self.mode == "enlargement":
            imageLR = transformToTensor(image)

            return imageLR

    def updateDataList(self):
        # 画像ファイルのパスのリストを新たに取得する。
        self.imagePathsList = self._getImagePaths()

    def _getImagePaths(self):
        # 指定したディレクトリー内の画像ファイルのパスのリストを取得する。
        imageDirectory = Path(self.imageDirectory)
        imagePathsList = [
            p for p in imageDirectory.glob("**/*") if p.suffix in ImageFromDirectory.imageExtensions]

        return imagePathsList

    def _enlarge(self, image):
        image = image.resize((image.width * 4, image.height * 4), Image.BICUBIC) # resize
        return image

    def _preprocess(self, image):
        minSize = image.width if image.width < image.height else image.height
        randomSize = random.randrange(384, minSize)
        left = random.randrange(0, image.width - randomSize)
        top = random.randrange(0, image.height - randomSize)
        right = left + randomSize
        bottom = top + randomSize
        image = image.crop((left, top, right, bottom))
        image = ImageOps.autocontrast(image, 0.01) # auto contrast, 0.01% cut-off
        image = image.filter(ImageFilter.UnsharpMask(radius = 0.5, percent = 400, threshold = 0)) # unsharp mask
        image = image.resize((384, 384), Image.BICUBIC) # resize
        h, s, v = image.convert("HSV").split()
        randomValue = random.randint(-16, 16)
        hShifted = h.point(lambda x: (x + randomValue) % 255 if (x + randomValue) % 255 >= 0 else 255 - (x + randomValue)) # change hue
        image = Image.merge("HSV", (hShifted, s, v)).convert("RGB")
        randomValue = random.randint(0, 1)
        if randomValue == 0:
            image = ImageOps.mirror(image)

        return image

    def _downsampleAndDeteriorate(self, image):
        image = image.resize((96, 96), Image.BICUBIC)
        randomRadius = random.random()
        image = image.filter(ImageFilter.GaussianBlur(randomRadius))
        q = random.randint(noiseLevel, 100)
        imageFile = BytesIO()
        image.save(imageFile, 'webp', quality=q)
        image = Image.open(imageFile)
        return image

    def __len__(self):
        # len(datasetインスタンス)でディレクトリー内の画像ファイルの数を返す。

        return len(self.imagePathsList)






class Swish(torch.nn.Module): # Swish activation function.
    def forward(self, x):
        return x * torch.sigmoid(x)





class Model(torch.nn.Module):
    def __init__(self):
        super().__init__() # 親クラスである torch.nn.Module の __init__ を継承する。
        # weightの初期化はKaiming Heの初期化で自動的に成される。

        nChannels = 128
        self.nResidualBlocks1 = 8
        self.nResidualBlocks2 = 16
        index = 0
        layersList = []

        layersList.append(
            torch.nn.Conv2d(in_channels=3, out_channels=nChannels * 1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

        layersList.append(Swish())

        layersList.append(
            torch.nn.GroupNorm(num_groups=int(nChannels * 1 / 8), num_channels=nChannels * 1, eps=1e-05, affine=True))

        layersList.append(Swish())

        # Residual Blocks
        for j in range(self.nResidualBlocks2):
            for i in range(self.nResidualBlocks1):
                layersList.append(
                    torch.nn.Conv2d(in_channels=nChannels * 1, out_channels=nChannels * 2, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

                layersList.append(Swish())

                layersList.append(
                    torch.nn.Conv2d(in_channels=nChannels * 2, out_channels=nChannels * 1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

                layersList.append(Swish())

            layersList.append(
                torch.nn.Conv2d(in_channels=nChannels * 1, out_channels=nChannels, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

            layersList.append(Swish())

        layersList.append(
            torch.nn.Conv2d(in_channels=nChannels * 1, out_channels=nChannels, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

        layersList.append(Swish())
        # Residual Blocks end

        layersList.append(
            torch.nn.Conv2d(in_channels=nChannels * 1, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

        layersList.append(
            torch.nn.PixelShuffle(upscale_factor=2))

        layersList.append(Swish())

        layersList.append(
            torch.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='replicate'))

        layersList.append(torch.nn.PixelShuffle(upscale_factor=2))

        layersList.append(Swish())

        layersList.append(
            torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='replicate'))

        layersList.append(torch.nn.Tanh())

        self.Layers = torch.nn.ModuleList(layersList)

    def forward(self, x):
        i = 0

        x = self.Layers[i](x) # Conv2d
        i += 1
        x = self.Layers[i](x) # Swish
        i += 1
        x = self.Layers[i](x) # GroupNorm
        i += 1
        x = self.Layers[i](x) # Swish
        i += 1

        # Residual Blocks
        x1 = x
        for k in range(self.nResidualBlocks2):
            x0 = x
            for j in range(self.nResidualBlocks1):
                h = self.Layers[i](x) # Conv2d
                i += 1
                h = self.Layers[i](h) # Swish
                i += 1
                h = self.Layers[i](h) # Conv2d
                i += 1
                h = self.Layers[i](h) # Swish
                i += 1
                x = x + h
            x = self.Layers[i](x) # Conv2d
            i += 1
            x = self.Layers[i](x) # Swish
            i += 1
            x = x + x0
        x = self.Layers[i](x) # Conv2d
        i += 1
        x = self.Layers[i](x) # Swish
        i += 1
        x = x + x1
        # Residual Blocks end

        x = self.Layers[i](x) # Conv2d
        i += 1
        x = self.Layers[i](x) # PixelShuffle
        i += 1
        x = self.Layers[i](x) # Swish
        i += 1
        x = self.Layers[i](x) # Conv2d
        i += 1
        x = self.Layers[i](x) # PixelShuffle
        i += 1
        x = self.Layers[i](x) # Swish
        i += 1
        x = self.Layers[i](x) # Conv2d
        i += 1
        x = self.Layers[i](x) # Tanh

        return x





def train():
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

    # オプティマイザーを作成する。
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # SSIM損失関数のインスタンスを作成する。
    ssimLossFunction = pytorch_ssim.SSIM(window_size=11) # torch.nn.Moduleを継承している。

    saveDirectoryGenerated = samplesPath + "generated"

    # 評価用画像の Dataset を作成する。
    datasetValidation = ImageFromDirectory(validationHRImagePath, "validation")

    # 評価用画像の DataLoader を作成する。
    dataloaderValidation = DataLoader(datasetValidation, batch_size=miniBatchSize, shuffle=False, num_workers=0, drop_last=False)

    # 評価用画像を保存する。
    miniBatchLRList = []
    i = 0
    for miniBatchHR, miniBatchLR in dataloaderValidation:
        utils.save_image(miniBatchHR, saveDirectoryGenerated + "/" + str(i) + "-HR.png", nrow=16)
        utils.save_image(miniBatchLR, saveDirectoryGenerated + "/" + str(i) + "-LR.png", nrow=16)
        miniBatchLRList.append(miniBatchLR)
        i += 1

    # 学習用画像の Dataset を作成する。
    datasetTrain = ImageFromDirectory(trainingHRImagePath, "training")

    # 学習用画像の DataLoader を作成する。
    dataloaderTraining = DataLoader(datasetTrain, batch_size=miniBatchSize, shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(0, nEpoch):
        epochTime = time.time()
        totalSSIMRGBLoss, step = 0, 0

        datasetTrain.updateDataList() # データセットのリストを更新する。

        nImagesTrain = len(datasetTrain)
        nStep = math.floor(nImagesTrain / miniBatchSize)
        print("The dataset has been updated.")
        print("Number of Images: {} Number of Steps: {}".format(nImagesTrain, nStep))

        i = 0
        for miniBatchHR, miniBatchLR in dataloaderTraining:

            stepTime = time.time()
            miniBatchHR = miniBatchHR.to(device)
            miniBatchLR = miniBatchLR.to(device)
            model.train() # training モードに設定する。
            miniBatchGenerated = model(miniBatchLR) # 画像データをモデルに入力する。
            ssimRGBLoss = torch.pow(1 - ssimLossFunction(miniBatchGenerated, miniBatchHR), 2) # 生成画像と正解画像との間の損失を計算させる。
            optimizer.zero_grad() # 勾配を初期化する。
            ssimRGBLoss.backward() # 誤差逆伝播により勾配を計算させる。
            optimizer.step() # パラメーターを更新させる。

            totalSSIMRGBLoss += float(ssimRGBLoss)
            print("Epoch: {:2d} Step: {:4d} Time: {:4.2f} SSIM_RGB_Loss: {:.8f}".format(
                  epoch, step, time.time() - stepTime, ssimRGBLoss)) # SSIM損失値を表示させる。

            step += 1

            # Validationを実行する。
            if i % 20 == 0:
                model.eval() # evaluation モードに設定する。
                with torch.no_grad(): # 以下のスコープ内では勾配計算をさせない。
                    j = 0
                    for miniBatchLR in miniBatchLRList:
                        miniBatchGenerated = model(miniBatchLR) # 評価用画像データのミニバッチをモデルに入力する。
                        utils.save_image(miniBatchGenerated, saveDirectoryGenerated + "/" + str(j) + "-" + str(epoch) + "-" + str(i) + ".png", nrow=16)
                        torch.save(model.to("cpu").state_dict(), checkpointPath + "model.pth") # モデル データを保存する。
                        j += 1

            i += 1

        print("Epoch[{:2d}/{:2d}] Time: {:4.2f} SSIM_RGB_Loss: {:.8f}".format(
            epoch, nEpoch, time.time() - epochTime, totalSSIMRGBLoss / nStep))







def evaluate():
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


    # 入力画像の Dataset を作成する。
    datasetEvaluation = ImageFromDirectory(evaluationImagePath, "evaluation")

    # 入力画像の DataLoader を作成する。
    dataloaderEvaluation = DataLoader(datasetEvaluation, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    saveDirectoryGenerated = samplesPath + "evaluated"

    nImagesEvaluation = len(datasetEvaluation)
    nStep = math.floor(nImagesEvaluation / miniBatchSize)
    print("Number of Images: {}".format(nImagesEvaluation))

    i = 0
    for miniBatchHR, miniBatchLR in dataloaderEvaluation:
        model.eval() # evaluation モードに設定する。
        with torch.no_grad(): # 以下のスコープ内では勾配計算をさせない。
            stepTime = time.time()
            miniBatchHR = miniBatchHR.to(device)
            miniBatchLR = miniBatchLR.to(device)
            miniBatchGenerated = model(miniBatchLR) # 画像データをモデルに入力する。
            utils.save_image(miniBatchHR, saveDirectoryGenerated + "/" + str(i) + "-Bicubic.png", nrow=16)
            utils.save_image(miniBatchGenerated, saveDirectoryGenerated + "/" + str(i) + "-Generated.png", nrow=16)

            i += 1
            print("{}: Time: {:4.2f}".format(i, time.time() - stepTime))

    print("Done.")







def enlarge():
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


    # 入力画像の Dataset を作成する。
    datasetEnlargement = ImageFromDirectory(enlargementLRImagePath, "enlargement")

    # 入力画像の DataLoader を作成する。
    dataloaderEnlargement = DataLoader(datasetEnlargement, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    saveDirectoryGenerated = samplesPath + "enlarged"

    nImagesEnlargement = len(datasetEnlargement)
    nStep = math.floor(nImagesEnlargement / miniBatchSize)
    print("Number of Images: {}".format(nImagesEnlargement))

    i = 0
    for miniBatchLR in dataloaderEnlargement:
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

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate, enlarge')

    args = parser.parse_args()

    mode = args.mode

    if mode == 'train':
        train()
    elif mode == 'evaluate':
        evaluate()
    elif mode == 'enlarge':
        enlarge()
    else:
        raise Exception("Unknow --mode")
