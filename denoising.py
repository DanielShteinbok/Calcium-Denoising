import numpy as np
import cv2

def dff(image, background):
    return image - background - np.min(np.reshape(image - background, (background.shape[0]*background.shape[1], -1)))

def mse(a, b):
    return np.sum((a - b)**2)/a.shape[-1]/a.shape[-2]

def get_similar(full_stack, index, num_imgs):
    other_inds = [index-1, index+1]
    out_stack = np.empty((num_imgs, full_stack.shape[1], full_stack.shape[2]))
    def add_img(i, side):
        # side = 0 if adding previous, side=1 if adding next
        out_stack[i, :, :] = full_stack[other_inds[side],:,:]
        other_inds[side] += 2*side - 1
    for i in range(num_imgs):
        if other_inds[1] >= full_stack.shape[0]:
            add_img(i, 0)
        elif other_inds[0] < 0:
            add_img(i, 1)
        elif mse(full_stack[other_inds[0],:,:], full_stack[index,:,:]) > mse(full_stack[other_inds[1],:,:], full_stack[index,:,:]):
            add_img(i, 1)
        else:
            add_img(i, 0)
    return out_stack

def full_denoising(full_stack, index, background, num_images=5, h=4):
    """
    Parameters:
        full_stack should be of shape (C, height, width) where height is the height of the image (e.g. 800)
        and width is the width of the image (e.g. 1280)
    """
    out_image = np.empty_like(full_stack[index, :,:], dtype=np.uint8)
    #cv2.fastNlMeansDenoising((dff(
        #np.mean(get_similar(full_stack, index, num_images), axis=0),
    #background)//2).astype(np.uint8), out_image, h=h)
    #background)//256).astype(np.uint8), out_image, h=h)
    dff_img = dff(
        np.mean(get_similar(full_stack, index, num_images), axis=0),
    background)
    cv2.fastNlMeansDenoising((dff_img/dff_img.max()*255).astype(np.uint8), out_image, h=h)
    #cv2.fastNlMeansDenoising((dff_img/2).astype(np.uint8), out_image, h=h)
    #background)//256).astype(np.uint8), out_image, h=h)
    return out_image

