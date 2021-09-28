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

# Adam
config.learningRate = 2e-5

# Training
config.nEpoch = 800

# Number of Iterations of the Step to Save
config.nIterationOfStepToSave = 4000

# Noise Reduction
# WebP compression level; 1 to 100 (smaller value adds more noise)
config.noiseLevel = 5

# Validation Set Location
config.validationHRImagePath = config.homePath + '/SRNet-DR/HRImage-Validation/'

# Train Set Location
config.trainingHRImagePath = config.homePath + '/SRNet-DR/HRImage-Training/'

# Test Set Location
config.evaluationImagePath = config.homePath + '/SRNet-DR/LRImage-Evaluation/'

# Tnlargement Set Location
config.enlargementLRImagePath = config.homePath + '/SRNet-DR/LRImage-Enlargement/'
