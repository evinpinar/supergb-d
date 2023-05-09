
import time
import os
import argparse
import shutil
import numpy as np
import cv2

import src.util.utilities as util_

from data.process import get_patches, get_binary_patches, get_neighbor_patches, get_patch_boxes, get_pairing_gt


def calculate_normals_cv2(depth_img):
    zy, zx = np.gradient(depth_img)
    # or sobel
    # zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
    # zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(depth_img)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255

    return normal


def prepare_patches(rgb_img, depth_img, foreground_labels):


    patches = get_patches(rgb_img, depth_img, n_segments=256, input_modalities="RGBXYZ")

    patch_binary_masks = get_binary_patches(patches)

    patch_boxes = get_patch_boxes(patch_binary_masks)
    patch_pairs = get_neighbor_patches(patch_binary_masks, patch_boxes)
    patch_pair_gt = get_pairing_gt(patch_binary_masks, patch_pairs, foreground_labels)

    num_pairs = len(patch_pairs)
    num_patches, H, W = patch_binary_masks.shape[0],  patch_binary_masks.shape[1],  patch_binary_masks.shape[2]
    num_gt_pairs = len(patch_pair_gt)

    gt_pairs = np.zeros((num_pairs))

    for ind, (pi, pj) in enumerate(patch_pairs):

        if (pi, pj) in patch_pair_gt:
            gt_pairs[ind] = 1

    feat_dim = 3
    # assume fixed size for the batch processing!
    patch_centroids = np.zeros((3, num_patches))  # np.zeros((feat_dim, num_pairs))

    full_z = depth_img
    row = np.arange(0, H)
    col = np.arange(0, W)
    full_col, full_row = np.meshgrid(col, row)
    full_pc = np.stack((full_row, full_col, full_z), axis=2)

    depth_normals = calculate_normals_cv2(depth_img)
    patch_normals = np.zeros((3, num_patches))

    for i in range(num_patches):

        pi_mask = patch_binary_masks[i]

        pc = full_pc[pi_mask==True]  #np.stack((x, y, z), axis=1)
        pi_centroid = np.mean(pc, axis=0)

        patch_centroids[:, i] = pi_centroid

        i_x = pi_centroid[0].round().astype(int)
        i_y = pi_centroid[1].round().astype(int)
        patch_normals[:, i] = depth_normals[i_x, i_y]


    patch_features = {
        'patch_centroids': patch_centroids,
        'patch_normals': patch_normals
    }

    feat_dim = 2 * (3+3) # centroid, normal

    pairwise_patch_features = np.zeros((feat_dim, num_pairs))  # np.zeros((feat_dim, num_pairs))

    for ind, (pi, pj) in enumerate(patch_pairs):

        pi_feats = np.concatenate((patch_centroids[:, pi], patch_normals[:, pi]))
        pj_feats = np.concatenate((patch_centroids[:, pj], patch_normals[:, pj]))

        pairwise_patch_features[:, ind] = np.concatenate((pi_feats, pj_feats))

    return num_patches, num_pairs, num_gt_pairs, patch_pairs, patch_binary_masks, patch_features, pairwise_patch_features, gt_pairs


def preprocess_data(image_dirs, input_dir, output_dir ):

    base_dir = input_dir + 'training_set/'

    output_base_dir = output_dir + 'training_set/'
    output_feat_dir = output_dir + 'features'
    output_feat_dir_rgb = output_dir + 'features_rgb'

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    if not os.path.exists(output_feat_dir):
        os.makedirs(output_feat_dir)

    if not os.path.exists(output_feat_dir_rgb):
        os.makedirs(output_feat_dir_rgb)

    for image_path in image_dirs:
        start_scene = time.time()
        scene_num = image_path.split('/')[-2] # scene_path[29:41]
        scene_dir = base_dir + '/' + scene_num + '/'
        view_num = int(image_path[-5])
        #print(scene_num)

        # RGB image
        rgb_img_filename = scene_dir + f"rgb_{view_num:05d}.jpeg"
        rgb_img = cv2.cvtColor(cv2.imread(rgb_img_filename), cv2.COLOR_BGR2RGB)

        depth_img_filename = scene_dir + f"depth_{view_num:05d}.png"
        depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH)  # This reads a 16-bit single-channel image. Shape: [H x W]

        depth_img = (depth_img / 1000.).astype(np.float32)

        # Labels
        foreground_labels_filename = scene_dir + f"segmentation_{view_num:05d}.png"
        foreground_labels = util_.imread_indexed(foreground_labels_filename)

        try:
            num_patches, num_pairs, num_gt_pairs, patch_pairs, patch_binary_masks, patch_features, pairwise_patch_features, gt_pairs = prepare_patches(rgb_img, depth_img, foreground_labels)

            # save everything in a h5 file
            output_scene_dir = output_feat_dir + "/" + scene_num + "/"
            if not os.path.exists(output_scene_dir):
                os.makedirs(output_scene_dir)

            feat_file = output_scene_dir + f"feats_{view_num:05d}_"
            np.save(feat_file + "num_patches.npy", np.array([num_patches]))
            np.save(feat_file + "num_pairs.npy", np.array([num_pairs]))
            np.save(feat_file + "num_gt_pairs.npy", np.array([num_gt_pairs]))
            np.save(feat_file + "patch_pairs.npy", patch_pairs)
            np.savez_compressed(feat_file + "patch_binary_masks", patch_binary_masks.astype('bool'))
            np.save(feat_file + "pairwise_patch_features.npy", pairwise_patch_features)
            np.save(feat_file + "gt_pairs.npy", gt_pairs)

            ### also extract rgb features, save in a different directory
            patch_rgb_feats = np.zeros((num_patches, 3))
            for i in range(num_patches):
                masked_rgb = rgb_img[patch_binary_masks[i] == 1]
                rgb_mean = np.mean(masked_rgb, axis=0)
                patch_rgb_feats[i] = rgb_mean

            rgb_feats = np.zeros((6, num_pairs))  # [feat, num_pairs]
            for p in range(num_pairs):
                pi, pj = patch_pairs[p]
                rgb_feats[:, p] = np.concatenate((patch_rgb_feats[pi], patch_rgb_feats[pj]),
                                                 axis=0)  # should be a 6 dim vector

            # output_feat_dir_rgb
            output_scene_dir = output_feat_dir_rgb + "/" + scene_num + "/"
            rgb_feat_file = output_scene_dir + "/" + f"feats_{view_num:05d}_" + "rgb_feats.npy"
            if not os.path.exists(output_scene_dir):
                os.makedirs(output_scene_dir)
            np.save(rgb_feat_file, rgb_feats)

            # MOVE training_set real images to output directory for completeness!

            new_scene_dir = output_base_dir + "/" + scene_num + '/'
            if not os.path.exists(new_scene_dir):
                os.makedirs(new_scene_dir)

            # copy view files to new one
            shutil.copyfile(scene_dir + "scene_description.txt", new_scene_dir + "scene_description.txt")
            shutil.copyfile(scene_dir + f"rgb_{view_num:05d}.jpeg", new_scene_dir + f"rgb_{view_num:05d}.jpeg")
            shutil.copyfile(scene_dir + f"depth_{view_num:05d}.png", new_scene_dir + f"depth_{view_num:05d}.png")
            shutil.copyfile(scene_dir + f"segmentation_{view_num:05d}.png",
                            new_scene_dir + f"segmentation_{view_num:05d}.png")
            shutil.copyfile(scene_dir + f"semseg_{view_num:05d}.png", new_scene_dir + f"semseg_{view_num:05d}.png")


        except Exception as e:
            print("Could not extract feats for img: ", rgb_img_filename)
            print(e)

        end_scene = time.time()
        print("Time ======== full scene took ======== ", end_scene - start_scene)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--outputdir', default='logdir/', type=str, required=False)
    parser.add_argument('--numworkers', default=8, type=int, required=False)

    args = parser.parse_args()

    input_dataset = args.dataset
    output_dir = args.outputdir
    num_workers = args.numworkers

    print("preprocess data!! ")
    print(" input: ", input_dataset)
    print(" output: ", output_dir)
    print("num workers: ", num_workers)

    # prepare
    train_ids = np.load('train_ids.npy')
    val_ids = np.load('val_ids.npy')
    test_open_ids = np.load('test_open_ids.npy')

    image_paths = np.concatenate((train_ids, val_ids, test_open_ids))
    print("total images: ", len(image_paths))
    num_batch = (len(image_paths) + num_workers + 1)//num_workers
    stes=[]
    for i in range(num_workers):
        st = i*num_batch
        e = (i+1)*num_batch+1
        stes.append((image_paths[st:e], input_dataset, output_dir))

    import concurrent.futures as cf

    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(preprocess_data, stes)

    print("{} seconds".format(round(time.time() - start, 2)))
