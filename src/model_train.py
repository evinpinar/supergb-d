

#%matplotlib inline
#os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU

import argparse
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
# tensorboard stuff
from torch.utils.tensorboard import SummaryWriter

import src.evaluation as evaluation
import src.util.utilities as util_
from model_network import Tabletop_Object_Dataset_Preprocessed, MergerNetwork, infer_seg


iter_num = 1

def train(data_loader, model, optimizer, tb_writer):
    global iter_num

    batch_time = util_.AverageMeter()
    data_time = util_.AverageMeter()
    total_losses = util_.AverageMeter()

    model.train()
    for i, batch in enumerate(data_loader):

        pairwise_patch_features = batch['pairwise_patch_features']
        gt_pairs = batch['gt_pairs']

        pairwise_patch_features = pairwise_patch_features.cuda()
        gt_pairs = gt_pairs.cuda()

        optimizer.zero_grad()
        preds = model(pairwise_patch_features)
        loss = F.binary_cross_entropy_with_logits(preds, gt_pairs.unsqueeze(1))

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        N = pairwise_patch_features.shape[0] # batch size
        num_pairs = pairwise_patch_features.shape[2]
        total_losses.update(loss.item(), N*num_pairs) # record average edge labeling loss


        # tensorboard writer
        # Record information every x iterations
        if iter_num%500==0:
            print("loss at iter ", iter_num, ": ", loss.item())

            #iter_num = epoch*len(data_loader.dataset) + i

            info = {'iter_num': iter_num,
                    'Batch Time': round(batch_time.avg, 3),
                    'Data Time': round(data_time.avg, 3),
                    'loss': round(total_losses.avg, 7),
                    }

            # Tensorboard stuff
            tb_writer.add_scalar('Total Loss', info['loss'], iter_num)
            tb_writer.add_scalar('Time/per iter', info['Batch Time'], iter_num)
            tb_writer.add_scalar('Time/data fetch', info['Data Time'], iter_num)

            # Reset meters
            batch_time = util_.AverageMeter()
            data_time = util_.AverageMeter()
            total_losses = util_.AverageMeter()

        iter_num += 1


def eval(data_loader, model, tb_writer):

    #global iter_num

    total_losses = util_.AverageMeter()
    g_prec, g_rec, g_f, g_acc = util_.AverageMeter(), util_.AverageMeter(), util_.AverageMeter(), util_.AverageMeter()
    model.eval()
    for i, batch in enumerate(data_loader):
        pairwise_patch_features = batch['pairwise_patch_features']
        gt_pairs = batch['gt_pairs']

        pairwise_patch_features = pairwise_patch_features.cuda()
        gt_pairs = gt_pairs.cuda()

        preds = model(pairwise_patch_features)

        batch_id = 0
        num_pairs = batch['num_pairs'][batch_id].item()

        loss = F.binary_cross_entropy_with_logits(preds[:,:,:num_pairs], gt_pairs.unsqueeze(1)[:,:,:num_pairs])
        total_losses.update(loss.item(), num_pairs) # record average edge labeling loss


        gt_pairs = gt_pairs.detach().cpu().squeeze(1)[batch_id]
        result = torch.round(torch.sigmoid(preds.detach())).cpu().squeeze(1)[batch_id]
        graph_metrics = evaluation.graph_metrics(result, gt_pairs, num_pairs)
        g_prec.update(graph_metrics[0], 1)
        g_rec.update(graph_metrics[1], 1)
        g_f.update(graph_metrics[2], 1)
        g_acc.update(graph_metrics[3], 1)


    tot_loss = round(total_losses.avg, 7)
    print("Eval total loss: ", tot_loss)

    tb_writer.add_scalar('Val Loss', tot_loss, iter_num)

    return tot_loss


def test(data_loader, model, tb_dir, tb_writer, infer_seg_tp=False, infer_seg_pix=False, threshold=[], analyse_graph=False):
    # global iter_num
    print("test called for iter: ", iter_num)

    model.eval()
    eval_total = {
        'Objects F-measure': [],
        'Objects Precision': [],
        'Objects Recall': [],
        'Boundary F-measure': [],
        'Boundary Precision': [],
        'Boundary Recall': [],
        'obj_detected': [],
        'obj_detected_075': [],
        'obj_gt': [],
        'obj_detected_075_percentage': [],
        'Graph Precision': [],
        'Graph Recall': [],
        'Graph F-measure': [],
        'Graph Accuracy': [],
    }
    mean_avg_prec, mean_avg_recall, mean_avg_f = [], [], []
    cats_f = {}
    cats_prec = {}
    cats_rec = {}
    boundary_cats_f = {}
    boundary_cats_prec = {}
    boundary_cats_rec = {}
    for i in range(27):
        cats_f[i] = []
        cats_prec[i] = []
        cats_rec[i] = []
        boundary_cats_f[i] = []
        boundary_cats_prec[i] = []
        boundary_cats_rec[i] = []

    graph_analysis = {}
    total_losses = util_.AverageMeter()
    for i, batch in enumerate(data_loader):
        if i%100==0:
            print("data: ", i)

        pairwise_patch_features = batch['pairwise_patch_features']
        gt_pairs = batch['gt_pairs']

        pairwise_patch_features = pairwise_patch_features.cuda()
        gt_pairs = gt_pairs.cuda()

        preds = model(pairwise_patch_features)

        batch_id = 0

        num_pairs = batch['num_pairs'][batch_id].item()

        loss = F.binary_cross_entropy_with_logits(preds[:, :, :num_pairs], gt_pairs.unsqueeze(1)[:, :, :num_pairs])
        N = pairwise_patch_features.shape[0]
        total_losses.update(loss.item(), num_pairs) # record average edge labeling loss


        result = torch.round(torch.sigmoid(preds[batch_id].detach())).cpu().squeeze()

        seg_masks_infer = infer_seg(batch, result, batch_id, infer_seg_tp, infer_seg_pix, analyse_graph)
        if analyse_graph:
            seg_masks = seg_masks_infer[0]
            graph_analysis[i] = seg_masks_infer[1] # graph analysis
        else:
            seg_masks = seg_masks_infer

        foreground_labels = util_.torch_to_numpy(batch['foreground_labels'])  # Shape: [N x H x W]

        semseg_labels = util_.torch_to_numpy(batch['semseg_labels'])  # Shape: [N x H x W]
        if len(threshold) == 0:
            eval_metrics = evaluation.multilabel_category_metrics(seg_masks, foreground_labels[0], semseg_labels[0])
        else:
            metric_results, eval_metrics = evaluation.multilabel_category_metrics_threshold(seg_masks,
                                                                                        foreground_labels[0],
                                                                                        semseg_labels[0], threshold)


        for key, _ in eval_total.items():
            if key in ['Graph Precision', 'Graph Recall', 'Graph F-measure', 'Graph Accuracy']:
                continue
            eval_total[key].append(eval_metrics[key])

        avg_prec, avg_rec, avg_f = [],[],[]
        for k, v in eval_metrics['Cats F-measure'].items():
            cats_f[k].append(eval_metrics['Cats F-measure'][k])
            avg_f.append(eval_metrics['Cats F-measure'][k])
            cats_prec[k].append(eval_metrics['Cats Precision'][k])
            avg_prec.append(eval_metrics['Cats Precision'][k])
            cats_rec[k].append(eval_metrics['Cats Recall'][k])
            avg_rec.append(eval_metrics['Cats Recall'][k])
            boundary_cats_f[k].append(eval_metrics['Cats Boundary F-measure'][k])
            boundary_cats_prec[k].append(eval_metrics['Cats Boundary Precision'][k])
            boundary_cats_rec[k].append(eval_metrics['Cats Boundary Recall'][k])

        mean_avg_recall.append(np.nanmean(avg_rec))
        mean_avg_prec.append(np.nanmean(avg_prec))
        mean_avg_f.append(np.nanmean(avg_f))

        batch_id = 0
        gt_pairs = batch['gt_pairs'].detach().cpu().squeeze(1)[batch_id]
        num_pairs = batch['num_pairs'][batch_id].item()
        result = torch.round(torch.sigmoid(preds.detach())).cpu().squeeze(1)[batch_id]
        graph_metrics = evaluation.graph_metrics(result, gt_pairs, num_pairs)
        eval_total['Graph Precision'].append(graph_metrics[0])
        eval_total['Graph Recall'].append(graph_metrics[1])
        eval_total['Graph F-measure'].append(graph_metrics[2])
        eval_total['Graph Accuracy'].append(graph_metrics[3])


    tot_loss = round(total_losses.avg, 7)
    print(data_loader.dataset.train_or_test, " total loss: ", tot_loss)

    tb_writer.add_scalar(data_loader.dataset.train_or_test + ' Loss', tot_loss, iter_num)

    eval_mean = {}
    for key, _ in eval_total.items():
        eval_mean[key] = round(np.mean(eval_total[key]) * 100, 2)

    alist = ['Objects Precision', 'Objects Recall', 'Objects F-measure', 'Boundary Precision', 'Boundary Recall',
             'Boundary F-measure', 'obj_detected', 'obj_detected_075', 'obj_gt', 'obj_detected_075_percentage',
             'Graph Precision', 'Graph Recall', 'Graph F-measure', 'Graph Accuracy']
    for ar in alist:
        print(f"{ar: >10}", end="\t")
    print()
    for ar in alist:
        ang = eval_mean[ar]
        print(f"{ang:15}", end="\t")
    print()

    mean_prec, mean_rec, mean_f = [], [], []
    boundary_mean_prec, boundary_mean_rec, boundary_mean_f = [], [], []
    for i in range(1, 26):
        if len(cats_f[i]) != 0:
            mean_f.append(round(np.nanmean(cats_f[i]) * 100, 2))
            mean_prec.append(round(np.nanmean(cats_prec[i]) * 100, 2))
            mean_rec.append(round(np.nanmean(cats_rec[i]) * 100, 2))
            boundary_mean_f.append(round(np.nanmean(boundary_cats_f[i]) * 100, 2))
            boundary_mean_prec.append(round(np.nanmean(boundary_cats_prec[i]) * 100, 2))
            boundary_mean_rec.append(round(np.nanmean(boundary_cats_rec[i]) * 100, 2))
        else:
            mean_f.append(np.nan)
            mean_prec.append(np.nan)
            mean_rec.append(np.nan)
            boundary_mean_f.append(np.nan)
            boundary_mean_prec.append(np.nan)
            boundary_mean_rec.append(np.nan)

    evaluation.save_table_cats(tb_dir + str(iter_num)+ "_" + data_loader.dataset.train_or_test + '_metrics.tsv', eval_mean,
                               [mean_prec, mean_rec, mean_f, boundary_mean_prec, boundary_mean_rec, boundary_mean_f],
                               data_loading_params['use_categories'], graph=True)

    np.save(tb_dir + str(iter_num) + "_eval_all_images.npy", eval_total)
    if analyse_graph:
        np.save(tb_dir + str(iter_num) + "_graph_analyse_pairs.npy", graph_analysis)
    prec, rec, fscore = eval_mean['Objects Precision'], eval_mean['Objects Recall'], eval_mean['Objects F-measure']

    return prec, rec, fscore


def main(args):

    TOD_filepath = args.dataset
    batch_size = args.batchsize
    num_epochs = args.numepochs
    tb_dir = args.outputdir
    num_cat = args.numcat

    print("file path: ", TOD_filepath, "tb_dir: ", tb_dir)
    print("num category: ", num_cat)

    data_loading_params['tb_directory'] = tb_dir + 'tbdir/'
    data_loading_params['tb_dir'] = tb_dir
    data_loading_params['flush_secs'] = 10

    print("categories used: ", len(data_loading_params['use_categories']), " : ", data_loading_params['use_categories'])

    tb_writer = SummaryWriter(data_loading_params['tb_directory'],
                              flush_secs=data_loading_params['flush_secs'])

    dataset_train_val = Tabletop_Object_Dataset_Preprocessed(TOD_filepath + 'training_set/', TOD_filepath + 'features', 'train',
                                                             data_loading_params, dataloader_sample=cfg.dataloader_sample,
                                                             dino=cfg.dino, neg_sample=cfg.neg_sample, depth_normalize=cfg.depth_normalize,
                                                             rgb=cfg.rgb, xyz=cfg.xyz, normals=cfg.normals, extreme=cfg.extreme, dino_normalize=cfg.dino_normalize,
                                                             rgb_normalize=cfg.rgb_normalize)
    dataset_val_val = Tabletop_Object_Dataset_Preprocessed(TOD_filepath + 'training_set/', TOD_filepath + 'features',
                                                             'val', data_loading_params, dino=cfg.dino, depth_normalize=cfg.depth_normalize,
                                                           rgb=cfg.rgb, xyz=cfg.xyz, normals=cfg.normals, extreme=cfg.extreme, dino_normalize=cfg.dino_normalize,
                                                           rgb_normalize=cfg.rgb_normalize)

    print("train dataset len", len(dataset_train_val))
    print("val dataset len", len(dataset_val_val))

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

    train_dl = DataLoader(dataset=dataset_train_val,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=args.numworkers,
                    drop_last=False)

    val_dl = DataLoader(dataset=dataset_val_val,
                          batch_size=1,
                          shuffle=False,
                          num_workers=args.numworkers)

    model = MergerNetwork(feat_dim=feat_dim, mlp_channels=args.model_channels, dropout_prob=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.OPTIMIZER.BASE_LR)
    from src.lr_scheduler import build_scheduler
    scheduler = build_scheduler(cfg, optimizer)

    val_every = 2

    model = model.cuda()

    best_prec, best_rec, best_fscore = 0, 0, 0
    best_prec_it, best_rec_it, best_fscore_it = 0, 0, 0

    for epoch_iter in range(num_epochs):
        print("epoch: ", epoch_iter)
        train(train_dl, model, optimizer, tb_writer)
        scheduler.step()

        # Save the best model
        if epoch_iter % val_every == 0:

            checkpoint = {
                'iter_num': iter_num,
                'epoch_num': epoch_iter,
                'model': model.state_dict(),
            }
            checkpoint['optimizer'] = optimizer.state_dict()
            filename = tb_dir + "/" + str(iter_num) + '_checkpoint.pth'
            print("Saving model to: ", filename)
            torch.save(checkpoint, filename)

            print("Testing the model!... ")
            prec, rec, fscore = test(val_dl, model, tb_dir, tb_writer, infer_seg_tp=False, analyse_graph=True)

            if prec >= best_prec:
                best_prec = prec
                best_prec_it = iter_num

            if rec >= best_rec:
                best_rec = rec
                best_rec_it = iter_num

            if fscore >= best_fscore:
                best_fscore = fscore
                best_fscore_it = iter_num

    print("Best checkpoints: ")
    print('Best precision: ', best_prec, " iter: ", best_prec_it)
    print('Best recall: ', best_rec, " iter: ", best_rec_it)
    print('Best fscore: ', best_fscore, " iter: ", best_fscore_it)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Learning Training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='run.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('--numworkers', default=1, type=int, required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    from src.config import *

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.numworkers = args.numworkers
    purge_cfg(cfg)
    cfg.freeze()
    print(cfg)

    shutil.copy(args.config_file, cfg.outputdir+'/config.yaml')

    main(cfg)

