
import numpy as np
import skimage.segmentation as seg

from collections import Counter

def get_patches(rgb_img, depth_img, n_segments=64, is_real=False, input_modalities="RGBXYZ"):


    if not is_real:
        dfseg = seg.slic(depth_img, n_segments=n_segments, compactness=0.08, sigma=0.01,
                            start_label=1)

        rgbseg = seg.slic(rgb_img, n_segments=n_segments, compactness=10, sigma=1,
                          start_label=1)
        if input_modalities == "XYZ":
            return dfseg
        elif input_modalities == "RGB":
            return rgbseg
        else:
            allseg = rgbseg * 1000 + dfseg
    else:
        dfseg = seg.slic(depth_img, n_segments=n_segments, compactness=0.01, sigma=0.1,
                         start_label=1)
        rgbseg = seg.slic(rgb_img, n_segments=n_segments, compactness=10, sigma=1,
                          start_label=1)
        allseg = rgbseg * 1000 + dfseg

    return allseg


def get_binary_patches(patches):
    #w = collections.Counter(patches.flatten())
    #lw = list(w.items())

    patch_ids = np.unique(patches)
    H, W = patches.shape[0], patches.shape[1]
    num_patch = len(patch_ids)
    bin_patches = np.zeros((num_patch, H, W))
    for i in range(num_patch):
        key = patch_ids[i]
        bin_patches[i] = patches == key

    return bin_patches

def get_boundingbox(bmask):
    #bmask = (mask.mean(axis=2) == 1)
    rows_with_white = np.max(bmask, axis=1)
    cols_with_white = np.max(bmask, axis=0)

    n_rows = len(rows_with_white)
    row_high = n_rows - np.argmax(rows_with_white[::-1])
    n_cols = len(cols_with_white)
    col_high = n_cols - np.argmax(cols_with_white[::-1])
    row_low = np.argmax(rows_with_white)
    col_low = np.argmax(cols_with_white)
    #im_cropped = mask[row_low:row_high, col_low:col_high]

    return row_low, row_high, col_low, col_high

def get_patch_boxes(patch_binary_masks):

    boxes = []
    num_patches = patch_binary_masks.shape[0]
    for patch in range(num_patches):
        x1, x2, y1, y2 = get_boundingbox(patch_binary_masks[patch])
        boxes.append((x1,x2,y1,y2))
    return boxes

def get_neighbor_patches(patches, patch_boxes):
    # Look up into neighboring patches on all directions
    # create a set of pairs

    def is_neighbour(pb1, pb2):
        row_low_1, row_high_1, col_low_1, col_high_1 = pb1
        row_low_2, row_high_2, col_low_2, col_high_2 = pb2

        if row_low_2 > row_high_1 or row_low_1 > row_high_2:
            return False

        if col_low_2> col_high_1 or col_low_1 > col_high_2:
            return False

        return True

    from itertools import combinations

    # All possible pairs in List
    # Using combinations()
    num_patches = patches.shape[0]
    pairs = list(set(combinations(range(num_patches), 2)))
    #
    # print("num pairs before filtering: ", len(pairs))

    neighboring_pairs = []
    for pi, pj in pairs:
        if is_neighbour(patch_boxes[pi], patch_boxes[pj]):
            neighboring_pairs.append((pi, pj))

    return neighboring_pairs

def get_pairing_gt(patches, patch_pairs, labels):

    pair_gt = []
    num_patch, H, W = patches.shape
    gt_labels = np.zeros((num_patch))
    for p in range(num_patch):
        vals = labels[patches[p]==1].astype(int)
        b = Counter(vals)
        lbl = b.most_common()[0][0]
        gt_labels[p] = lbl

    for pi, pj in patch_pairs:
        li = gt_labels[pi]
        lj = gt_labels[pj]
        if li == lj:
            pair_gt.append((pi, pj))

    return pair_gt

