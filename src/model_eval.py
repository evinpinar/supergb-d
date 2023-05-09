#%matplotlib inline
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU

import torch
from torch.utils.data import DataLoader
import torch.optim
# tensorboard stuff
from torch.utils.tensorboard import SummaryWriter

from data.data_loader import Tabletop_Object_Dataset_Preprocessed
from src.model_network import MergerNetwork


def main(args):

    TOD_filepath = args.dataset
    tb_dir = args.outputdir  # 'logdir_selective'
    num_cat = args.numcat
    ckpt = args.ckpt

    print("file path: ", TOD_filepath, "tb_dir: ", tb_dir)
    print("num category: ", num_cat)

    data_loading_params['tb_directory'] = tb_dir + 'tbdir/'
    data_loading_params['tb_dir'] = tb_dir
    data_loading_params['flush_secs'] = 10

    print("categories used: ", len(data_loading_params['use_categories']), " : ", data_loading_params['use_categories'])

    tb_writer = SummaryWriter(data_loading_params['tb_directory'],
                              flush_secs=data_loading_params['flush_secs'])

    dataset_test_open = Tabletop_Object_Dataset_Preprocessed(TOD_filepath + 'training_set/', TOD_filepath + 'features',
                                                             'test_open', data_loading_params, dino=cfg.dino, depth_normalize=cfg.depth_normalize,
                                                             rgb=cfg.rgb, xyz=cfg.xyz, normals=cfg.normals, extreme=cfg.extreme, dino_normalize=cfg.dino_normalize,
                                                             rgb_normalize=cfg.rgb_normalize)

    print("open test dataset len", len(dataset_test_open))

    feat_dim = 18
    if cfg.rgb and cfg.xyz and cfg.normals:
        feat_dim = 18
    if not cfg.rgb:
        feat_dim -= 6
    if not cfg.xyz:
        feat_dim -= 6
    if not cfg.normals:
        feat_dim -= 6
    if cfg.dino:
        feat_dim += 12
    if cfg.extreme:
        if cfg.dino and cfg.xyz:
            feat_dim = 4 # extreme case of z and dino6
        else:
            feat_dim = 2 # only z or only dino 6

    print("model input size: ", feat_dim)

    model = MergerNetwork(feat_dim=feat_dim, mlp_channels=args.model_channels, dropout_prob=0)

    filename = tb_dir + str(ckpt) + '_checkpoint.pth'
    print("Loading model from: ", filename)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    model.cuda()

    test_dl = DataLoader(dataset=dataset_test_open,
                         batch_size=1,
                         shuffle=False,
                         num_workers=1)

    model.eval()

    from model_train import test


    output_dir = tb_dir + '/open/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Eval on open data")
    test(test_dl, model, output_dir, tb_writer, args.inferseg)




if __name__ == '__main__':

    from src.config import *
    from model_train import parse_args

    args = parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.numworkers = args.numworkers
    purge_cfg(cfg)
    cfg.freeze()
    print(cfg)

    main(cfg)

