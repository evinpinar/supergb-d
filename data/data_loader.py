import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json

from numpy.random import default_rng
from src.util import utilities as util_
from data import data_augmentation

NUM_VIEWS_PER_SCENE = 7

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2


###### Some utilities #####

def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


############# Tabletop Object Dataset Preprocessed #############

class Tabletop_Object_Dataset_Preprocessed(Dataset):
    """ Data loader for Tabletop Object Dataset
    """

    def __init__(self, base_dir, feat_dir, train_or_test, config, dataloader_sample=None, dino=False, neg_sample=0.5, depth_normalize=0, rgb_normalize = False, rgb=True, xyz=True, normals=True, extreme=False, fine=False, dino_normalize=True):
        self.base_dir = base_dir
        self.feat_dir = feat_dir
        self.config = config
        self.train_or_test = train_or_test
        self.dataloader_sample = dataloader_sample
        self.dino=dino
        self.rgb=rgb
        self.xyz=xyz
        self.normals=normals
        self.neg_sample=neg_sample
        self.depth_normalize=depth_normalize
        self.rgb_normalize=rgb_normalize
        self.dino_normalize=dino_normalize
        self.extreme=extreme
        self.fine=fine

        if train_or_test == 'train':
            selected_img_ids = np.load('train_ids.npy')
            self.selected_imgs = selected_img_ids
        elif train_or_test == 'val':
            val_ids = np.load('val_ids.npy')
            if len(val_ids) > 2000:
                randm = np.random.choice(val_ids, size=2000, replace=False)
                val_ids = list(randm)
            self.selected_imgs = val_ids
        elif train_or_test == 'test_open':
            selected_img_ids = np.load('test_open_ids.npy')
            self.selected_imgs = selected_img_ids

        self.len = len(self.selected_imgs)

        self.name = 'TableTop'

        if 'v6' in self.base_dir:
            global OBJECTS_LABEL
            OBJECTS_LABEL = 4

        self.seen_categories = ['ashcan', 'bag', 'can', 'mug', 'cellular_telephone', 'cap', 'camera', 'clock', 'loudspeaker',
                           'washer', 'microwave', 'pillow'],
        self.full_categories = []

    def __len__(self):
        return self.len

    def process_rgb(self, rgb_img):
        """ Process RGB image
                - random color warping
        """
        rgb_img = rgb_img.astype(np.float32)

        rgb_img = data_augmentation.standardize_image(rgb_img)

        return rgb_img

    def process_depth(self, depth_img):
        """ Process depth channel
                TODO: CHANGE THIS
                - change from millimeters to meters
                - cast to float32 data type
                - add random noise
                - compute xyz ordered point cloud
        """

        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)

        return depth_img

    def extract_rgb_feats(self, rgb_img, patch_binary_masks, patch_pairs, num_pairs, dino=None):

        num_patches = patch_binary_masks.shape[0]
        patch_rgb_feats = torch.zeros((num_patches, 3))
        for i in range(num_patches):
            masked_rgb = rgb_img[:, patch_binary_masks[i]==1]
            rgb_mean = torch.mean(masked_rgb, axis=1)
            patch_rgb_feats[i] = rgb_mean

        rgb_feats = torch.zeros((6, num_pairs)) #[feat, num_pairs]
        for p in range(num_pairs):
            pi, pj = patch_pairs[p]
            rgb_feats[:, p] = torch.cat((patch_rgb_feats[pi], patch_rgb_feats[pj]), axis=0) # should be a 6 dim vector

        # return: num_patch_pairs x 3 (mean of rgb channels

        return rgb_feats

    def read_features(self, scene_dir, view_num):

        scene_num = scene_dir.split("/")[-2]

        feat_file = self.feat_dir + "/" + scene_num + "/" + f"feats_{view_num:05d}_"
        feat_file_rgb = self.feat_dir +"_rgb" + "/" + scene_num + "/" + f"feats_{view_num:05d}_"
        feat_file_dino = self.feat_dir + "_dino" + "/" + scene_num + "/" + f"feats_{view_num:05d}_"


        num_patches = np.load(feat_file + "num_patches.npy")[0]
        num_pairs = np.load(feat_file + "num_pairs.npy")[0]
        num_gt_pairs = np.load(feat_file + "num_gt_pairs.npy")[0]
        patch_pairs = np.load(feat_file + "patch_pairs.npy")
        patch_binary_masks = np.load(feat_file + "patch_binary_masks.npz")['arr_0']
        pairwise_patch_features = np.load(feat_file + "pairwise_patch_features.npy")
        gt_pairs = np.load(feat_file + "gt_pairs.npy")
        rgb_feats = np.load(feat_file_rgb + "rgb_feats.npy")
        if self.dino:
            dino_feats = np.load(feat_file_dino + 'dino.npy')

        pairwise_patch_features = data_augmentation.array_to_tensor(pairwise_patch_features)  # Shape: [feat, num_pairs]
        gt_pairs = data_augmentation.array_to_tensor(gt_pairs)  # [num_pairs]
        rgb_feats = data_augmentation.array_to_tensor(rgb_feats)
        # patch_pairs = torch.from_numpy(patch_pairs).float() # Shape: [num_pairs]
        # Shape: [num_pairs]
        # patch_binary_masks = data_augmentation.array_to_tensor(patch_binary_masks) # Shape: [num_patches, H, W]
        patch_binary_masks = torch.from_numpy(patch_binary_masks).float()
        patch_pairs = data_augmentation.array_to_tensor(np.array(patch_pairs))

        pairwise_patch_features = torch.cat(
            (pairwise_patch_features[:6], rgb_feats[:3], pairwise_patch_features[6:], rgb_feats[3:]), dim=0)

        # center_x, center_y, depth, normal_x, normal_y, normal_z, r, g, b
        # normalize values:
        pairwise_patch_features[0, :] /= 480  # x dim
        pairwise_patch_features[9, :] /= 480  # x dim
        pairwise_patch_features[1, :] /= 640  # y dim
        pairwise_patch_features[10, :] /= 640  # y dim
        pairwise_patch_features[3:6, :] /= 255.  # normals
        pairwise_patch_features[12:15, :] /= 255.  # normals
        pairwise_patch_features[6:9, :] /= 255.  # rgb
        pairwise_patch_features[15:18, :] /= 255.  # rgb

        if self.rgb_normalize:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            pairwise_patch_features[6:9, :] = ((pairwise_patch_features[6:9, :].transpose(1, 0) - mean)/std).transpose(1, 0)
            pairwise_patch_features[15:18, :] = ((pairwise_patch_features[15:18, :].transpose(1, 0) - mean)/std).transpose(1, 0)

        if self.depth_normalize == 1:
            # Global normalization
            # Depth factor = depth_max calculated from the dataset
            DEPTH_FACTOR = 35.0
            pairwise_patch_features[2, :] /= DEPTH_FACTOR
            pairwise_patch_features[11, :] /= DEPTH_FACTOR
        elif self.depth_normalize == 2:
            # Local normalization within the image
            # calculate min / max values
            # Depth factor = max - min
            DEPTH_MIN = min(pairwise_patch_features[2, :].min(), pairwise_patch_features[11, :].min())
            DEPTH_MAX = max(pairwise_patch_features[2, :].max(), pairwise_patch_features[11, :].max())
            pairwise_patch_features[2, :] = ( pairwise_patch_features[2, :] - DEPTH_MIN ) / (DEPTH_MAX - DEPTH_MIN + 1e-5)
            pairwise_patch_features[11, :] = ( pairwise_patch_features[11, :] - DEPTH_MIN ) / (DEPTH_MAX - DEPTH_MIN + 1e-5)

        xyz_feats_p1, xyz_feats_p2 = pairwise_patch_features[:3, :], pairwise_patch_features[9:12, :]
        normal_feats_p1, normal_feats_p2 = pairwise_patch_features[3:6, :], pairwise_patch_features[12:15, :]
        rgb_feats_p1, rgb_feats_p2 = pairwise_patch_features[6:9, :], pairwise_patch_features[15:18, :]

        total_feats_p1, total_feats_p2 = None, None
        if self.xyz and not self.normals and not self.rgb:
            total_feats_p1, total_feats_p2 = xyz_feats_p1, xyz_feats_p2
        elif not self.xyz and self.normals and not self.rgb:
            total_feats_p1, total_feats_p2 = normal_feats_p1, normal_feats_p2
        elif not self.xyz and not self.normals and self.rgb:
            total_feats_p1, total_feats_p2 = rgb_feats_p1, rgb_feats_p2
        elif self.xyz and self.normals and not self.rgb:
            total_feats_p1, total_feats_p2 = torch.cat(
                (xyz_feats_p1, normal_feats_p1), dim=0), torch.cat((xyz_feats_p2, normal_feats_p2), dim=0)
        elif self.xyz and not self.normals and self.rgb:
            total_feats_p1, total_feats_p2 = torch.cat(
                (xyz_feats_p1, rgb_feats_p1), dim=0), torch.cat((xyz_feats_p2, rgb_feats_p2), dim=0)
        elif not self.xyz and self.normals and self.rgb:
            total_feats_p1, total_feats_p2 = torch.cat(
                (normal_feats_p1, rgb_feats_p1), dim=0), torch.cat((normal_feats_p2, rgb_feats_p2), dim=0)
        elif self.xyz and self.normals and self.rgb:
            total_feats_p1, total_feats_p2 = torch.cat(
                (xyz_feats_p1, normal_feats_p1, rgb_feats_p1), dim=0), torch.cat((xyz_feats_p2, normal_feats_p2, rgb_feats_p2), dim=0)

        # append dino feats as well, they are currently per-patch
        if self.dino:
            dino_feats_p1 = torch.from_numpy(dino_feats[patch_pairs.int()[:,0]].transpose(1,0)).float()
            dino_feats_p2 = torch.from_numpy(dino_feats[patch_pairs.int()[:,1]].transpose(1,0)).float()

            if self.dino_normalize:
                full_dino = torch.cat((dino_feats_p1, dino_feats_p2), dim=1)
                # normalize dino features
                dino_max = full_dino.max(dim=1)[0]
                dino_min = full_dino.min(dim=1)[0]
                ##dino_first = (dino_first-dino_min.unsqueeze(1))
                dino_feats_p1 = (dino_feats_p1 - dino_min.unsqueeze(1)) / (dino_max - dino_min).unsqueeze(1)
                dino_feats_p2 = (dino_feats_p2 - dino_min.unsqueeze(1)) / (dino_max - dino_min).unsqueeze(1)
            if total_feats_p1 is None:
                total_feats_p1, total_feats_p2 = dino_feats_p1, dino_feats_p2
            else:
                total_feats_p1, total_feats_p2 = torch.cat(
                    (total_feats_p1, dino_feats_p1), dim=0), torch.cat((total_feats_p2, dino_feats_p2), dim=0)

        if self.extreme:
            if self.dino and self.xyz:
                # take only z and dino6
                total_feats_p1, total_feats_p2 = total_feats_p1[(2,8),:], total_feats_p2[(2,8),:]
            elif self.xyz:
                # take only z
                total_feats_p1, total_feats_p2 = total_feats_p1[2, :].unsqueeze(0), total_feats_p2[2, :].unsqueeze(0)
            elif self.dino:
                # take only dino6
                total_feats_p1, total_feats_p2 = total_feats_p1[5, :].unsqueeze(0), total_feats_p2[5, :].unsqueeze(0)

        pairwise_patch_features = torch.cat((total_feats_p1, total_feats_p2), dim=0)

        return pairwise_patch_features, patch_binary_masks, patch_pairs, num_patches, num_pairs, gt_pairs

    def __getitem__(self, idx):

        if self.train_or_test == 'train' and self.dataloader_sample != None:
            return self.getitem_train(idx)
        else:
            return self.getitem_test(idx)

    def getitem_train(self, idx):
        cv2.setNumThreads(
            0)  # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355. Seems to work for me

        scene_path = self.selected_imgs[idx]

        # Get scene directory
        scene_dir = self.base_dir + scene_path[29:41]  # scene_path[:-22]
        # Get view number
        view_num = int(scene_path[-5])

        pairwise_patch_features, patch_binary_masks, patch_pairs, num_patches, num_pairs, gt_pairs = self.read_features(scene_dir, view_num)

        # randomly select 128/256 pairs
        rand_size = self.dataloader_sample

        feat_dim = pairwise_patch_features.shape[0]
        if self.fine:
            patch_binary_masks_padded = torch.zeros(2000, patch_binary_masks.shape[1],
                                                    patch_binary_masks.shape[2])  # 1000
        else:
            patch_binary_masks_padded = torch.zeros(500, patch_binary_masks.shape[1], patch_binary_masks.shape[2]) # 1000

        rng = default_rng()
        ## ensure half of the samples are positive, half are negative
        positives = np.where(gt_pairs == 1)[0]
        negatives = np.where(gt_pairs == 0)[0]

        negative_samples = (int)(rand_size * self.neg_sample) # rest is the negative
        positive_samples = rand_size - negative_samples

        if len(negatives) == 0:
            random_pairs_negatives = []
            positive_samples = int(rand_size)
            print("size: ", len(negatives), num_pairs, len(positives), gt_pairs.sum(), scene_path)
        elif negative_samples > len(negatives):
            random_pairs_negatives = rng.choice(negatives, size=negative_samples, replace=True)
        else:
            random_pairs_negatives = rng.choice(negatives, size=negative_samples, replace=False)

        if positive_samples > len(positives):
            random_pairs_positives = rng.choice(positives, size=positive_samples, replace=True)
        else:
            random_pairs_positives = rng.choice(positives, size=positive_samples, replace=False)

        if len(random_pairs_negatives) == 0:
            random_pairs = random_pairs_positives #np.concatenate((random_pairs_positives, random_pairs_positives), axis=0)
        else:
            random_pairs = np.concatenate((random_pairs_positives, random_pairs_negatives), axis=0)

        pairwise_patch_features_padded = pairwise_patch_features[:, random_pairs]

        rand_feat_size = rand_size // 2
        random_swaps = rng.choice(rand_size, size=rand_feat_size, replace=False)
        first_half, second_half = pairwise_patch_features_padded[:feat_dim // 2,
                                  random_swaps], pairwise_patch_features_padded[feat_dim // 2:, random_swaps]
        pairwise_patch_features_padded[:, random_swaps] = torch.cat((second_half, first_half), dim=0)

        gt_pairs_padded = gt_pairs[random_pairs]
        patch_pairs_padded = patch_pairs[random_pairs]
        patch_pairs_padded[random_swaps, 0], patch_pairs_padded[random_swaps, 1] = patch_pairs_padded[random_swaps, 1], \
                                                                                   patch_pairs_padded[random_swaps, 0]

        patch_binary_masks_padded[:num_patches, :, :] = patch_binary_masks


        return {'scene_dir': scene_dir,
                'view_num': view_num,
                'pairwise_patch_features': pairwise_patch_features_padded,
                'gt_pairs': gt_pairs_padded,
                'patch_pairs': patch_pairs_padded,
                'patch_binary_masks': patch_binary_masks_padded,
                'num_pairs': num_pairs,
                'num_patches': num_patches,
                }

    def getitem_test(self, idx):

        cv2.setNumThreads(
            0)  # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355. Seems to work for me

        scene_path = self.selected_imgs[idx]

        # Get scene directory
        scene_dir = self.base_dir + scene_path[29:41] #scene_path[:-22]
        # Get view number
        view_num = int(scene_path[-5])

        # RGB image
        rgb_img_filename = scene_dir + f"rgb_{view_num:05d}.jpeg"
        rgb_img = cv2.cvtColor(cv2.imread(rgb_img_filename), cv2.COLOR_BGR2RGB)

        # Depth image
        depth_img_filename = scene_dir + f"depth_{view_num:05d}.png"
        depth_img = cv2.imread(depth_img_filename,
                               cv2.IMREAD_ANYDEPTH)  # This reads a 16-bit single-channel image. Shape: [H x W]
        xyz_img = self.process_depth(depth_img)

        # Labels
        foreground_labels_filename = scene_dir + f"segmentation_{view_num:05d}.png"
        foreground_labels = util_.imread_indexed(foreground_labels_filename)
        scene_description_filename = scene_dir + "scene_description.txt"
        scene_description = json.load(open(scene_description_filename))
        scene_description['view_num'] = view_num

        rgb_normalized = self.process_rgb(rgb_img)
        rgb_normalized = data_augmentation.array_to_tensor(rgb_normalized)


        # Labels
        semseg_labels_filename = scene_dir + f"semseg_{view_num:05d}.png"
        semseg_labels = util_.imread_indexed(semseg_labels_filename)


        label_abs_path = '/'.join(foreground_labels_filename.split('/')[-2:])  # Used for evaluation

        # Turn these all into torch tensors
        rgb_img = data_augmentation.array_to_tensor(rgb_img)  # Shape: [3 x H x W]
        xyz_img = data_augmentation.array_to_tensor(xyz_img)  # Shape: [3 x H x W]
        foreground_labels = data_augmentation.array_to_tensor(foreground_labels)  # Shape: [H x W]



        pairwise_patch_features, patch_binary_masks, patch_pairs, num_patches, num_pairs, gt_pairs = self.read_features(scene_dir, view_num)

        feat_dim = pairwise_patch_features.shape[0]
        if self.fine:
            pairwise_patch_features_padded = torch.zeros((feat_dim, 10000))
            gt_pairs_padded = torch.zeros(10000)
            patch_pairs_padded = torch.zeros(10000, 2)
            patch_binary_masks_padded = torch.zeros(2000, patch_binary_masks.shape[1], patch_binary_masks.shape[2])
        else:
            pairwise_patch_features_padded = torch.zeros((feat_dim, 2000))
            gt_pairs_padded = torch.zeros(2000)
            patch_pairs_padded = torch.zeros(2000, 2)
            patch_binary_masks_padded = torch.zeros(500, patch_binary_masks.shape[1], patch_binary_masks.shape[2])

        pairwise_patch_features_padded[:, :num_pairs] = pairwise_patch_features
        gt_pairs_padded[:num_pairs] = gt_pairs
        patch_pairs_padded[:num_pairs, :] = patch_pairs
        patch_binary_masks_padded[:num_patches,:,:] = patch_binary_masks


        return {'rgb': rgb_normalized,
                'xyz': xyz_img,
                'semseg_labels': semseg_labels,
                'foreground_labels': foreground_labels,
                'scene_dir': scene_dir,
                'view_num': view_num,
                'label_abs_path': label_abs_path,
                'pairwise_patch_features': pairwise_patch_features_padded,
                'gt_pairs': gt_pairs_padded,
                'patch_pairs': patch_pairs_padded,
                'patch_binary_masks': patch_binary_masks_padded,
                'num_pairs': num_pairs,
                'num_patches': num_patches
                }

