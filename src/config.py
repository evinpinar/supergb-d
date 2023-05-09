from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.dataset = '../datasets/TOD/'
_C.batchsize= 2
_C.numworkers= 1
_C.numepochs= 15
_C.numcat= 12
_C.outputdir = 'logdir/'
_C.model_channels = (256, 256)
_C.gnn_layers = 0
_C.dataloader_sample = 128
_C.neg_sample =0.5
_C.dino = False
_C.xyz = True
_C.normals = True
_C.rgb = True
_C.depth_normalize = 0
_C.dino_normalize = True
_C.rgb_normalize = False

_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = "Adam"
_C.OPTIMIZER.BASE_LR = 0.001

_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = "StepLR"
_C.SCHEDULER.StepLR = CN()
_C.SCHEDULER.StepLR.gamma = 0.5
_C.SCHEDULER.StepLR.step_size = 15
_C.SCHEDULER.CLIP_LR = 1e-5
_C.SCHEDULER.MAX_EPOCH = 20

_C.LOG_PERIOD= 10
_C.VAL_PERIOD= 1
_C.CHECKPOINT_PERIOD= 1


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()


def purge_cfg(cfg):
    """Purge configuration for clean logs.
    The rules to purge is:
    1. If a CfgNode has 'TYPE' attribute, its CfgNode children the key of which do not contain 'TYPE' will be removed.
    Args:
        cfg (CfgNode): input config
    """
    target_key = cfg.get('TYPE', None)
    removed_keys = []
    for k, v in cfg.items():
        if isinstance(v, CN):
            if target_key is not None and (k != target_key):
                removed_keys.append(k)
            else:
                purge_cfg(v)

    for k in removed_keys:
        del cfg[k]
