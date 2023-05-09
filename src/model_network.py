
#%matplotlib inline
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim

import numpy as np
import matplotlib.pyplot as plt

import src.evaluation as evaluation
import src.util.utilities as util_

from skimage.color import label2rgb

from data.data_loader import Tabletop_Object_Dataset_Preprocessed


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)

def set_bn(module, momentum):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


def xavier_uniform(module):
    if module.weight is not None:
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)

class Conv1d(nn.Module):
    """Applies a 1D convolution over an input signal composed of several input planes
    optionally followed by batch normalization and relu activation.
    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (nn.Module, optional): relu activation module
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

        self.reset_parameters()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self, init_fn=None):
        """default initialization"""
        if init_fn is not None:
            init_fn(self.conv)
        if self.bn is not None:
            init_bn(self.bn)

class SharedMLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 ndim=1,
                 dropout_prob=0.0,
                 bn=True,
                 bn_momentum=0.1):
        """Multilayer perceptron shared on resolution (1D or 2D)
        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            ndim (int): the number of dimensions to share
            dropout_prob (float or None): dropout ratio
            bn (bool): whether to use batch normalization
            bn_momentum (float)
        """
        super(SharedMLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]
        self.ndim = ndim

        if ndim == 1:
            mlp_module = Conv1d
        else:
            raise ValueError('SharedMLP only supports ndim=(1, 2).')

        for ind, out_channels in enumerate(mlp_channels):
            self.append(mlp_module(in_channels, out_channels, 1, relu=True, bn=bn, bn_momentum=bn_momentum))
            in_channels = out_channels

        # Do not use modules due to ModuleList.
        assert dropout_prob >= 0.0
        self.dropout_prob = dropout_prob

    def forward(self, x):
        for module in self:
            assert isinstance(module, (Conv1d))
            x = module(x)
            if self.training and self.dropout_prob > 0.0:
                if self.ndim == 1:
                    x = F.dropout(x, p=self.dropout_prob, training=True)
                elif self.ndim == 2:
                    x = F.dropout2d(x, p=self.dropout_prob, training=True)
                else:
                    raise ValueError('SharedMLP only supports ndim=(1, 2).')
        return x

    def reset_parameters(self, init_fn=None):
        for module in self:
            assert isinstance(module, (Conv1d))
            module.reset_parameters(init_fn)

    def extra_repr(self):
        return 'dropout_prob={}'.format(self.dropout_prob) if self.dropout_prob > 0.0 else ''

class MergerNetwork(nn.Module):

    ### TODO: Replace this with a triplet network / graformer etc

    def __init__(self, feat_dim=12, mlp_channels=(128, 256, 256, 1), dropout_prob=0.5):
        super(MergerNetwork, self).__init__()
        self.feat_dim = feat_dim

        self.mlp_shared = SharedMLP(feat_dim, mlp_channels, dropout_prob=dropout_prob)
        self.reset_parameters()

    def forward(self, pairwise_patch_features):

        # input: batch, num_pairs, dim*2
        # output: batch, num_pairs

        x_out = self.mlp_shared(pairwise_patch_features)

        return x_out

    def reset_parameters(self):
        # xavier_uniform(self.ins_logit)
        self.mlp_shared.reset_parameters(xavier_uniform)
        set_bn(self, momentum=0.01)

def infer_seg(batch, result, batch_id, prec=False, pixcount=False, analyse_graph=False):
    patch_pairs = batch['patch_pairs']
    patch_binary_masks = batch['patch_binary_masks']

    patch_binary_masks = patch_binary_masks[batch_id].detach().cpu().numpy()
    patch_pairs = patch_pairs[batch_id].detach().cpu().numpy()
    num_patches = batch['num_patches'][batch_id].item()
    num_pairs = batch['num_pairs'][batch_id].item()
    gt_pairs = batch['gt_pairs'].detach().cpu().squeeze(1)[batch_id]
    analysis = {'tp': [], 'fp': [], 'fn': [], 'tn': []}

    # print("num patches: ", num_patches)
    # print("num pairs: ", num_pairs)
    g = Graph(num_patches)
    debug_pairs = 300
    for ind in range(num_pairs): #num_pairs
        pi, pj = int(patch_pairs[ind][0]), int(patch_pairs[ind][1])
        if result[ind] == 1:
            if prec: # add an edge only if it is also 1 in gt
                if result[ind] == batch['gt_pairs'][0][ind]:
                    g.addEdge(pi, pj)
            elif pixcount: # add an edge only if both patch sizes are bigger than a threshold
                pi_mask = patch_binary_masks[int(pi)].astype(bool)
                pj_mask = patch_binary_masks[int(pj)].astype(bool)
                if pi_mask.sum() > 10 and pj_mask.sum() > 10:
                    g.addEdge(pi, pj)
            else:
                g.addEdge(pi, pj)
        if analyse_graph:
            pi_mask = patch_binary_masks[int(pi)].astype(bool)
            pj_mask = patch_binary_masks[int(pj)].astype(bool)
            pair_total_pixels = tuple([pi_mask.sum(), pj_mask.sum()])
            pred_edge = result[ind]
            gt_edge = gt_pairs[ind]
            # true positive
            if gt_edge == 1 and pred_edge == 1:
                analysis['tp'].append(pair_total_pixels)
            # false positive
            elif gt_edge == 0 and pred_edge == 1:
                analysis['fp'].append(pair_total_pixels)
            # false negative
            elif gt_edge == 1 and pred_edge == 0:
                analysis['fn'].append(pair_total_pixels)
            # false negative
            elif gt_edge == 0 and pred_edge == 0:
                analysis['tn'].append(pair_total_pixels)

    cc = g.connectedComponents()
    # print("Following are connected components: ", len(cc))

    new_mask = np.zeros((len(cc), patch_binary_masks.shape[1], patch_binary_masks.shape[2]))
    for i, comp in enumerate(cc):
        for c in comp:
            new_mask[i] = np.logical_or(new_mask[i], patch_binary_masks[c])

    # print(new_mask.shape)
    # colored_lbls = label2rgb(np.argmax(new_mask, axis=0).astype(int))

    seg_masks = np.argmax(new_mask, axis=0).astype(int)

    if analyse_graph:
        return seg_masks, analysis

    return seg_masks


class Graph:

    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]

    def DFSUtil(self, temp, v, visited):

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp

    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)

    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc



if __name__ == '__main__':

    from src.config import data_loading_params

    data_loading_params['tb_dir'] = "logdir_overfit/"
    base_dir = '../datasets/TOD/'
    dataset = Tabletop_Object_Dataset_Preprocessed(base_dir + 'training_set/', base_dir + 'features', 'val', data_loading_params)

    batch_size=4
    dl = DataLoader(dataset=dataset,
               batch_size=batch_size,
               shuffle=False,
               num_workers=1,
               drop_last=True)

    batch = next(iter(dl))

    pairwise_patch_features = batch['pairwise_patch_features']
    gt_pairs = batch['gt_pairs']


    feat_dim = pairwise_patch_features.shape[1]
    print("feature dim: ", feat_dim)

    model = MergerNetwork(feat_dim=feat_dim, dropout_prob=0)  # can be MLP, graph, transformer


    model = model.cuda()
    pairwise_patch_features = pairwise_patch_features.cuda()
    gt_pairs = gt_pairs.cuda()

    print("train...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for i in range(10000):
        optimizer.zero_grad()
        preds = model(pairwise_patch_features)
        loss = F.binary_cross_entropy_with_logits(preds, gt_pairs.unsqueeze(1))
        #loss = F.binary_cross_entropy(preds, gt_pairs.unsqueeze(1))
        if i % 1000 == 0:
            print("i: ", i, " ", loss)
        loss.backward()
        optimizer.step()

    ##### EVALUATE

    dataset = Tabletop_Object_Dataset_Preprocessed(base_dir + 'training_set/', base_dir + 'features', 'val',
                                                   data_loading_params)

    batch_size = 4
    dl = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=1,
                    drop_last=True)

    batch = next(iter(dl))

    model.eval()

    pairwise_patch_features = batch['pairwise_patch_features']
    gt_pairs = batch['gt_pairs']
    pairwise_patch_features = pairwise_patch_features.cuda()
    gt_pairs = gt_pairs.cuda()
    preds = model(pairwise_patch_features)

    print("preds: ", preds.shape, preds.squeeze().shape)

    for batch_id in range(batch_size):

        result = torch.round(torch.sigmoid(preds[batch_id].detach())).cpu().squeeze()

        seg_masks = infer_seg(batch, result, batch_id)+2
        # end = time.time()
        # print("inferring segmentation fully: ", end - start)
        foreground_labels = util_.torch_to_numpy(batch['foreground_labels']) +2 # Shape: [N x H x W]

        # eval_metrics = evaluation.multilabel_metrics(seg_masks, foreground_labels[0])

        semseg_labels = util_.torch_to_numpy(batch['semseg_labels'])  # Shape: [N x H x W]

        print("num pred ins: ", np.unique(seg_masks), "num gt inst: ", np.unique(foreground_labels[batch_id]))
        eval_metrics = evaluation.multilabel_category_metrics(seg_masks, foreground_labels[batch_id], semseg_labels[batch_id])

        print(eval_metrics['Cats Precision'], eval_metrics['Cats Recall'])

        colored_lbls = label2rgb(seg_masks+1)
        plt.imshow(colored_lbls)
        plt.show()


