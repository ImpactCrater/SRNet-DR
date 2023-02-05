from easydict import EasyDict as edict
from os.path import expanduser

config = edict()

# Home Path
config.homePath = expanduser("~")

# Checkpoint Location
config.checkpointPath = config.homePath + '/SRNet-DR/checkpoint/'

# Samples Location
config.samplesPath = config.homePath + '/SRNet-DR/samples/'

# Save File Format
config.saveFileFormat = '.png'

# Batch
config.miniBatchSize = 1

# Learning Rate of RAdam
config.learningRate = 2e-6 # モデルのパラメーター数が多いほど、またデータ数が多いほど、小さな学習率にする。

# Weight Decay of RAdam
config.weightDecay = 1e-9

# Training
config.nEpoch = 800

# Number of Iterations of the Step to Save
config.nIterationOfStepToSave = 1000

# Validation Set Location
config.validationHRImagePath = config.homePath + '/SRNet-DR/HRImage-Validation/'

# Train Set Location
config.trainingHRImagePath = config.homePath + '/SRNet-DR/HRImage-Training/'

# Super-Resolution Set Location
config.lRImagePath = config.homePath + '/SRNet-DR/LRImage-SR/'
