from yacs.config import CfgNode as CN

_C = CN()


###System settings###
_C.SYSTEM = CN()
#Number of workers
_C.SYSTEM.NUM_WORKERS = 4
#Random seed number
_C.SYSTEM.RANDOM_SEED = 1


###Dataset parameters###
_C.DATASET = CN()
# Name of datset: 'CUB', 'coco', 'flickr', 'FLO'
_C.DATASET.NAME = 'CUB'
# Dataset path
_C.DATASET.PATH = PATH
# Image mean and std
_C.DATASET.IMG_MEAN = (0.48145466, 0.4578275, 0.40821073)
_C.DATASET.IMG_STD = (0.26862954, 0.26130258, 0.27577711)
# Image size
_C.DATASET.IMG_SIZE = 224
# Augmentation
_C.DATASET.IMG_MASKING_PROB = 0.0
_C.DATASET.TEXT_MASKING_PROB = 0.0
# Batch size
_C.DATASET.BATCH_SIZE = 128

###Train parameters###
_C.TRAIN = CN()
#Number of training epochs
_C.TRAIN.NUM_EPOCHS = 250
#Number of epochs after which validation to be performed
_C.TRAIN.VAL_EPOCHS = 10
#Learning rate
_C.TRAIN.LEARNING_RATE = 0.00001
#Loss weighting
_C.TRAIN.LOSS_WEIGHT = 200



### Model Parameters ###
_C.MODEL = CN()
# VLM model 'CLIP' or 'BLIP'
_C.MODEL.VLM = 'CLIP'
# Image encoder backbone for CLIP
_C.MODEL.VLM_BACKBONE = 'ViT-B/32'
# VLM model weights
_C.MODEL.VLM_WEIGHTS = None
# Embedding dimension from VLM
_C.MODEL.EMB_DIM = 512
#Number of inducing points
_C.MODEL.NUM_INDUCING_PTS = 250
#Latent space dimension
_C.MODEL.LATENT_DIM = 10
#Checkpoint directory path
_C.MODEL.CHECKPOINT_DIR = 'checkpoint/'

def get_cfg_defaults():

    return _C.clone()
