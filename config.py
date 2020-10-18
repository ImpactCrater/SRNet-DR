from easydict import EasyDict as edict
from os.path import expanduser

config = edict()

# Home Path
config.homePath = expanduser("~")

# Checkpoint Location
config.checkpointPath = config.homePath + '/SRNet-D/checkpoint/'

# Samples Location
config.samplesPath = config.homePath + '/SRNet-D/samples/'

# Save File Format
config.saveFileFormat = '.png'

# Batch
config.miniBatchSize = 2

# Adam
config.learningRate = 2e-5

# Training
config.nEpoch = 100

# Noise Reduction
# WebP compression level; 1 to 100 (smaller value adds more noise)
config.noiseLevel = 10

# Validation Set Location
config.validationHRImagePath = config.homePath + '/SRNet-D/HRImage-Validation/'

# Train Set Location
config.trainingHRImagePath = config.homePath + '/SRNet-D/HRImage-Training/'

# Test Set Location
config.evaluationImagePath = config.homePath + '/SRNet-D/LRImage-Evaluation/'

# Tnlargement Set Location
config.enlargementLRImagePath = config.homePath + '/SRNet-D/LRImage-Enlargement/'
