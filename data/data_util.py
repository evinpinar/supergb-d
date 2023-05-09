import torch
import numpy as np
import cv2


def array_to_tensor(array):
    """ Converts a numpy.ndarray (N x H x W x C) to a torch.FloatTensor of shape (N x C x H x W)
        OR
        converts a nump.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    if array.ndim == 4: # NHWC
        tensor = torch.from_numpy(array).permute(0,3,1,2).float()
    elif array.ndim == 3: # HWC
        tensor = torch.from_numpy(array).permute(2,0,1).float()
    else: # everything else
        tensor = torch.from_numpy(array).float()

    return tensor

def translate(img, tx, ty, interpolation=cv2.INTER_LINEAR):
    """ Translate img by tx, ty

        @param img: a [H x W x C] image (could be an RGB image, flow image, or label image)
    """
    H, W = img.shape[:2]
    M = np.array([[1,0,tx],
                  [0,1,ty]], dtype=np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=interpolation)

def rotate(img, angle, center=None, interpolation=cv2.INTER_LINEAR):
    """ Rotate img <angle> degrees counter clockwise w.r.t. center of image

        @param img: a [H x W x C] image (could be an RGB image, flow image, or label image)
    """
    H, W = img.shape[:2]
    if center is None:
        center = (W//2, H//2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, M, (W, H), flags=interpolation)



def standardize_image(image):
    """ Convert a numpy.ndarray [H x W x 3] of images to [0,1] range, and then standardizes

        @return: a [H x W x 3] numpy array of np.float32
    """
    image_standardized = np.zeros_like(image).astype(np.float32)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i in range(3):
        image_standardized[...,i] = (image[...,i]/255. - mean[i]) / std[i]

    return image_standardized

def unstandardize_image(image):
    """ Convert a numpy.ndarray [H x W x 3] standardized image back to RGB (type np.uint8)
        Inverse of standardize_image()

        @return: a [H x W x 3] numpy array of type np.uint8
    """

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    orig_img = (image * std[None,None,:] + mean[None,None,:]) * 255.
    return orig_img.round().astype(np.uint8)


