from easydict import EasyDict as edict
import json
from os.path import expanduser

config = edict()
config.TRAIN = edict()
config.VALID = edict()

# home path
config.home_path = expanduser("~")

# checkpoint location
config.checkpoint_path = config.home_path + '/SRNet-D/checkpoint/'

# samples location
config.samples_path = config.home_path + '/SRNet-D/samples/'

# save file format
config.save_file_format = '.png'

# Adam
config.TRAIN.sample_batch_size = 25
config.TRAIN.batch_size = 9
config.TRAIN.learning_rate = 1e-4

# training
config.TRAIN.n_epoch = 100

# noise reduction
# WebP compression level; 1 to 100 (smaller value adds more noise)
config.TRAIN.noise_level = 40

# train set location
config.TRAIN.hr_img_path = config.home_path + '/SRNet-D/HRImage_Training/'

# test set location
config.VALID.hr_img_path = config.home_path + '/SRNet-D/HRImage_Validation/'
config.VALID.eval_img_path = config.home_path + '/SRNet-D/LRImage_Evaluation/'
config.VALID.eval_img_name_regx = '/1\.(bmp|png|webp|jpg)'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
