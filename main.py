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
import AdaBelief
import cv2
import model

# from torchsummary import summary


#python3 ~/SRNet-DR/main.py
#python3 ~/SRNet-DR/main.py --mode=sr


## Paths

homePath = expanduser("~")
# homePath = "/mnt/disks/SRNet-DR"

# Checkpoint Location
checkpointPath = homePath + '/SRNet-DR/checkpoint/'

# Samples Location
samplesPath = homePath + '/SRNet-DR/samples/'

# Validation Set Location
validationHRImagePath = homePath + '/SRNet-DR/HRImage-Validation/'

# Train Set Location
trainingHRImagePath = homePath + '/SRNet-DR/HRImage-Training/'
#trainingHRImagePath = homePath + '/SRNet-DR/HRImage-Validation/'

# Super-Resolution Set Location
lRImagePath = homePath + '/SRNet-DR/LRImage-SR/'
#lRImagePath = homePath + '/SRNet-DR/temp/'

# Save File Format
saveFileFormat = '.png'


## Hyper-Parameters

# Batch
miniBatchSize = 1

# Learning Rate
learningRate = 1e-6 # 1e-6 # モデルのパラメーター数が多いほど、またデータ数が多いほど、小さな学習率にする。

# Weight Decay
weightDecay = 1e-8 # 1e-8

# Training
nEpoch = 800

# Number of Iterations of the Step to Save
nIterationOfStepToSave = 400 # 10 or 400 or 1000

torch._dynamo.verbose=True
torch._dynamo.suppress_errors = True


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
        if image.mode == "RGBA":
            r, g, b, a = image.split()
            image = Image.merge("RGB", (r, g, b))

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
        h, s, v = imageHR.convert("HSV").split()
        v = ImageOps.autocontrast(v, cutoff=0.01, preserve_tone=False) # preserve_tone=True: Convert to grayscale.
        randomValue = random.randint(-16, 16)
        hShifted = h.point(lambda x: (x + randomValue) % 255 if (x + randomValue) % 255 >= 0 else 255 - (x + randomValue)) # rotate hue
        imageHR = Image.merge("HSV", (hShifted, s, v)).convert("RGB")
        imageHR = imageHR.filter(ImageFilter.UnsharpMask(radius=0.5, percent=400, threshold=0))
        imageHR = imageHR.resize((388, 388), Image.Resampling.BICUBIC)
        if random.randint(0, 1) == 1:
            imageHR = ImageOps.mirror(imageHR)

        # Deterioration
        imageLR = imageHR.copy()
        randomSize = math.floor(random.uniform(0.0, 1.0) * random.uniform(0.0, 1.0) * random.uniform(0.0, 1.0) * random.uniform(0.0, 1.0) * (388 - 97) + 97)
        imageLR = imageLR.resize((randomSize, randomSize), Image.Resampling.BICUBIC)
        randomRadius = random.uniform(0.0, 1.0) # (0.0, 1.0)
        imageLR = imageLR.filter(ImageFilter.GaussianBlur(randomRadius))

        randomStrength = random.uniform(0.0, 0.075) # (0.0, 0.075)
        width, height = imageLR.size
        r, g, b = imageLR.split()
        noiseImage = Image.effect_noise((width, height), 255) # Generate Gaussian noise
        r = Image.blend(r, noiseImage, randomStrength)
        noiseImage = Image.effect_noise((width, height), 255)
        g = Image.blend(g, noiseImage, randomStrength)
        noiseImage = Image.effect_noise((width, height), 255)
        b = Image.blend(b, noiseImage, randomStrength)
        imageLR = Image.merge("RGB", (r, g, b))

        randomQuality = random.randint(5, 100) # (5, 100)
        imageFile = BytesIO()
        imageLR.save(imageFile, 'webp', quality=randomQuality)
        imageLR = Image.open(imageFile)
        imageLR = imageLR.resize((388, 388), Image.Resampling.BICUBIC)

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












class GeneratorLossFunction(torch.nn.Module):
    def __init__(self):
        super(GeneratorLossFunction, self).__init__()

    def _calculateSsim(self, image1, image2):
        # image; [miniBatches, channels, height, width]
        _, numberOfChannels, height, width = image1.size()

        # SSIM: Structural Similarity
        # 入力は0.0から1.0の範囲に正規化済み。
        # 局所領域全体の類似度が高いほど1.0に近く、類似度が低いほど0.0に近く、逆相関では-1.0に近い。
        blurer = transforms.GaussianBlur(kernel_size=11, sigma=2.0)

        mean1 = blurer(image1)
        mean2 = blurer(image2)
        # 0.0から1.0

        deviation1 = image1 - mean1
        deviation2 = image2 - mean2
        # -1.0から1.0

        standardDeviation1 = torch.sqrt(blurer((deviation1) ** 2))
        standardDeviation2 = torch.sqrt(blurer((deviation2) ** 2))
        # 0.0から1.0

        # standardDeviation1_standardDeviation2 = standardDeviation1 * standardDeviation2
        # 0.0から1.0

        covariance = blurer(deviation1 * deviation2)
        # 局所領域全体の相関が強いほど1.0に近く、相関が弱いほど-1.0に近い。

        smallConstant1 = 0.01 ** 2
        smallConstant2 = 0.03 ** 2
        # smallConstant3 = smallConstant2 / 2

        ratioMapOfMean = (2.0 * mean1 * mean2 + smallConstant1) / (mean1 ** 2 + mean2 ** 2 + smallConstant1)
        # 類似度が高いほど1.0に近く、類似度が低いほど0.0に近い。

        # ratioMapOfStandardDeviation = (2.0 * standardDeviation1_standardDeviation2 + smallConstant2) / (standardDeviation1 ** 2 + standardDeviation2 ** 2 + smallConstant2)
        # 類似度が高いほど1.0に近く、類似度が低いほど0.0に近い。

        # correlationCoefficientMap = (covariance + smallConstant3) / (standardDeviation1_standardDeviation2 + smallConstant3)
        # 局所領域全体の類似度が高いほど1.0に近く、類似度が低いほど0.0に近く、逆相関では-1.0に近い。

        # ssimMap = ratioMapOfMean * ratioMapOfStandardDeviation * correlationCoefficientMap
        ssimMap = ratioMapOfMean * ((2.0 * covariance + smallConstant2) / (standardDeviation1 ** 2 + standardDeviation2 ** 2 + smallConstant2))
        # 式の変形による簡略化。
        return ssimMap


    def _calculateFeatureSsim(self, image1, image2):

        # GPUが利用可能ならGPUを利用する。
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.benchmark=True
        else:
            device = "cpu"
        # Kernel

        _, numberOfChannels, height, width = image1.size()

        kernel1 = torch.FloatTensor([[ 1,  0,  0], 
                                     [ 0, -2,  0], 
                                     [ 0,  0,  1]]).to(device, non_blocking=True)

        kernel2 = torch.FloatTensor([[ 0,  1,  0], 
                                     [ 0, -2,  0], 
                                     [ 0,  1,  0]]).to(device, non_blocking=True)

        kernel3 = torch.FloatTensor([[ 0,  0,  1], 
                                     [ 0, -2,  0], 
                                     [ 1,  0,  0]]).to(device, non_blocking=True)

        kernel4 = torch.FloatTensor([[ 0,  0,  0], 
                                     [ 1, -2,  1], 
                                     [ 0,  0,  0]]).to(device, non_blocking=True)

        kernel1_Expanded = kernel1.expand(numberOfChannels, 1, 3, 3).contiguous()
        image1_1 = torch.nn.functional.conv2d(image1, kernel1_Expanded, stride=1, padding=1, groups=numberOfChannels)
        image2_1 = torch.nn.functional.conv2d(image2, kernel1_Expanded, stride=1, padding=1, groups=numberOfChannels)

        kernel2_Expanded = kernel2.expand(numberOfChannels, 1, 3, 3).contiguous()
        image1_2 = torch.nn.functional.conv2d(image1, kernel2_Expanded, stride=1, padding=1, groups=numberOfChannels)
        image2_2 = torch.nn.functional.conv2d(image2, kernel2_Expanded, stride=1, padding=1, groups=numberOfChannels)

        kernel3_Expanded = kernel3.expand(numberOfChannels, 1, 3, 3).contiguous()
        image1_3 = torch.nn.functional.conv2d(image1, kernel3_Expanded, stride=1, padding=1, groups=numberOfChannels)
        image2_3 = torch.nn.functional.conv2d(image2, kernel3_Expanded, stride=1, padding=1, groups=numberOfChannels)

        kernel4_Expanded = kernel4.expand(numberOfChannels, 1, 3, 3).contiguous()
        image1_4 = torch.nn.functional.conv2d(image1, kernel4_Expanded, stride=1, padding=1, groups=numberOfChannels)
        image2_4 = torch.nn.functional.conv2d(image2, kernel4_Expanded, stride=1, padding=1, groups=numberOfChannels)

        featureSsimMap1 = self._calculateSsim(image1_1, image2_1)
        featureSsimMap2 = self._calculateSsim(image1_2, image2_2)
        featureSsimMap3 = self._calculateSsim(image1_3, image2_3)
        featureSsimMap4 = self._calculateSsim(image1_4, image2_4)

        return (featureSsimMap1 + featureSsimMap2 + featureSsimMap3 + featureSsimMap4) / 4


    def forward(self, image1, image2):
        ssimMap = self._calculateSsim(image1, image2)
        ssimLoss = ((torch.sqrt(1 + torch.pow((1 - ssimMap) * 1024, 2)) - 1) / 1024).mean()

        featureSsimMap = self._calculateFeatureSsim(image1, image2)
        featureSsimLoss = ((torch.sqrt(1 + torch.pow((1 - featureSsimMap) * 1024, 2)) - 1) / 1024).mean()

        return ssimLoss, featureSsimLoss






def displayImage(miniBatchGenerated, miniBatchHR):

    with torch.no_grad(): # 以下のスコープ内では勾配計算をさせない。
        # GPUが利用可能ならGPUを利用する。
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.benchmark=True
        else:
            device = "cpu"

        height = miniBatchHR.size()[2]
        width = miniBatchHR.size()[3]
        inputImage = torch.zeros((1, 3, height, width * 2)).to(device, non_blocking=True) # NCHW RGB [0., 1.]
        inputImage[0, :, :, :width] = miniBatchHR[0, :, :, :]
        inputImage[0, :, :, width:] = miniBatchGenerated[0, :, :, :]
        inputImage = inputImage.to("cpu")

        image = numpy.zeros((3, height, width * 2)) # CHW BGR [0., 1.]
        image[0, :, :] = inputImage[0, 2, :, :] # Blue
        image[1, :, :] = inputImage[0, 1, :, :] # Green
        image[2, :, :] = inputImage[0, 0, :, :] # Red
        image = numpy.transpose(image, (1, 2, 0)) # HWC BGR

        image = image.clip(min=0.0, max=1.0)
        image = (image * 255).astype(numpy.uint8)
        cv2.imshow('Result', image)
        keyPressed =cv2.waitKey(10)
    return






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
    #print(torch.cuda.device_count())
    # 分散処理システムを利用する。
    os.environ['MASTER_ADDR'] = 'localhost' # 処理を実行するマスター マシンのIPアドレスを指定する。
    os.environ['MASTER_PORT'] = '12355' # 利用する空きポート番号を指定する。
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    # 通信バックエンドの1つであるNCCLバックエンドは、CUDAのtensorを対象にした集合演算に最適化された実装となっている。
    # rank: distributed process number, world_size: number of distributed processes

    # チェック ポイントのディレクトリーを確認する。
    print("Checking the existence of the directory for the checkpoint.")
    if os.path.isdir(checkpointPath):
        print("O.K.")
    else:
        print("Making the directory.")
        os.mkdir(checkpointPath)

    # 生成モデルのインスタンスを作成する。
    modelOfGenerator = model.ModelOfGenerator()
    if os.path.isfile(checkpointPath + "modelOfGenerator.pth"):
        modelOfGenerator.load_state_dict(torch.load(checkpointPath + "modelOfGenerator.pth", map_location=torch.device(device)))
    else:
        print("modelOfGenerator.pth will be created.")

    print(modelOfGenerator)

    modelOfGenerator.to(device, non_blocking=True) # モデルのデータをdeviceに置く。

    # モデルを細切れにして順伝播でCPU Offloadingするように設定する。
    modelOfGenerator = FullyShardedDataParallel(modelOfGenerator, cpu_offload=CPUOffload(offload_params=True))

    # オプティマイザーを作成する。
    optimizerOfGenerator = AdaBelief.AdaBelief(modelOfGenerator.parameters(), lr=learningRate, weight_decay=weightDecay, rectify=False)

    # 損失関数のインスタンスを作成する。
    generatorLossFunction = GeneratorLossFunction() # torch.nn.Moduleを継承している。

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
    count = 0
    for epoch in range(0, nEpoch):
        step = 0

        datasetTrain.updateDataList() # データセットのリストを更新する。

        nImagesTrain = len(datasetTrain)
        nStep = math.floor(nImagesTrain / miniBatchSize)
        print("The dataset has been updated.")
        print("Number of Images: {} Number of Steps: {}".format(nImagesTrain, nStep))

        previousTime = time.time()
        for miniBatchHR, miniBatchLR in dataloaderTraining:
            miniBatchHR = miniBatchHR.to(device, non_blocking=True)
            miniBatchLR = miniBatchLR.to(device, non_blocking=True)

            # Train Generator
            modelOfGenerator.train() # training モードに設定する。
            miniBatchGenerated = modelOfGenerator(miniBatchLR) # 画像を生成モデルに入力して生成画像を得る。
            del miniBatchLR
            ssimLoss, featureSsimLoss = generatorLossFunction(miniBatchGenerated, miniBatchHR) # 損失を計算させる。
            displayImage(miniBatchGenerated, miniBatchHR)
            del miniBatchGenerated
            generatorLoss = ssimLoss + featureSsimLoss
            optimizerOfGenerator.zero_grad(set_to_none=True) # 勾配を削除により初期化する。
            generatorLoss.backward() # 誤差逆伝播により勾配を計算させる。
            optimizerOfGenerator.step() # パラメーターを更新させる。
            ssimLossValue = float(ssimLoss)
            featureSsimLossValue = float(featureSsimLoss)
            del ssimLoss
            del featureSsimLoss
            del generatorLoss

            nowTime = time.time()
            print("Epoch: {:2d} Count: {:4d} Time: {:4.2f} ssimLoss: {:.8f} featureSsimLoss: {:.8f}".format(
                  epoch, count, nowTime - previousTime, ssimLossValue, featureSsimLossValue)) # 損失値を表示させる。
            previousTime = nowTime

            count += 1
            step += 1



            # Validationを実行する。
            if count % nIterationOfStepToSave == 0 and count != 0:
                modelOfGenerator.eval() # evaluation モードに設定する。
                with torch.no_grad(): # 以下のスコープ内では勾配計算をさせない。
                    i = 0
                    for miniBatchLR in miniBatchLRList:
                        miniBatchLR = miniBatchLR.to(device, non_blocking=True)
                        miniBatchGenerated = modelOfGenerator(miniBatchLR) # 評価用画像データのミニバッチをモデルに入力する。
                        utils.save_image(miniBatchGenerated, saveDirectoryGenerated + "/" + str(i) + "-" + str(epoch) + "-" + str(step) + ".png", nrow=16)
                        torch.save(modelOfGenerator.state_dict(), checkpointPath + "modelOfGenerator.pth") # モデル データを保存する。
                        i += 1








def sr():
    print("Now processing...")

    # GPUが利用可能ならGPUを利用する。
    if torch.cuda.is_available():
      device = "cuda"
    else:
      device = "cpu"

    # モデルのインスタンスを作成する。
    modelOfGenerator = model.ModelOfGenerator()
    if os.path.isfile(checkpointPath + "modelOfGenerator.pth"):
        modelOfGenerator.load_state_dict(torch.load(checkpointPath + "modelOfGenerator.pth", map_location=torch.device(device)))

    # モデルのデータをdeviceに置く。
    modelOfGenerator.to(device, non_blocking=True)


    # summary(modelOfGenerator,(3,384,384))


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
        modelOfGenerator.eval() # evaluation モードに設定する。
        with torch.no_grad(): # 以下のスコープ内では勾配計算をさせない。
            stepTime = time.time()
            miniBatchLR = miniBatchLR.to(device, non_blocking=True)
            miniBatchGenerated = modelOfGenerator(miniBatchLR) # 画像データをモデルに入力する。
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
