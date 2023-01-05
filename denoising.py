import numpy as np
import cv2

def dff(image, background):
    return image - background - np.min(np.reshape(image - background, (background.shape[0]*background.shape[1], -1)))

def mse(a, b):
    return np.sum((a - b)**2)/a.shape[-1]/a.shape[-2]

def get_similar(full_stack, index, num_imgs, mse_threshold=None, print_mses=False):
    # this just holds the two "edges"
    other_inds = [index-1, index+1]

    # rather than keeping a stack of the frames we want, we'll just keep a list of their indices
    # this is more computationally efficient, but also allows for dynamically changing
    # the number of images we care about, e.g. due to thresholding
    out_stack_inds = [index]
    #out_stack = np.empty((num_imgs, full_stack.shape[1], full_stack.shape[2]))
    def add_img(i, side):
        # side = 0 if adding previous, side=1 if adding next
        # verify here that the mse is below the threshold
        mse_val = mse(full_stack[index], full_stack[other_inds[side], :, :])
        if mse_threshold is not None:
            if  mse_val > mse_threshold:
                # do nothing if mse is too high
                return
        if print_mses:
            print("On iteration " + str(i) + ", MSE is: " + str(mse_val))
        #out_stack[i, :, :] = full_stack[other_inds[side],:,:]
        out_stack_inds.append(other_inds[side])
        other_inds[side] += 2*side - 1
    for i in range(num_imgs - 1):
        if other_inds[1] >= full_stack.shape[0]:
            add_img(i, 0)
        elif other_inds[0] < 0:
            add_img(i, 1)
        elif mse(full_stack[other_inds[0],:,:], full_stack[index,:,:]) > mse(full_stack[other_inds[1],:,:], full_stack[index,:,:]):
            add_img(i, 1)
        else:
            add_img(i, 0)
    #return out_stack
    # this should return exactly the same way that it did before,
    # only now we've changed the internal behaviour above
    return full_stack[out_stack_inds, :, :]

def full_denoising(full_stack, index, background, num_images=5, h=4, verbose=False):
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
        np.mean(get_similar(full_stack, index, num_images, print_mses=verbose), axis=0),
    background)
    cv2.fastNlMeansDenoising((dff_img/dff_img.max()*255).astype(np.uint8), out_image, h=h)
    #cv2.fastNlMeansDenoising((dff_img/2).astype(np.uint8), out_image, h=h)
    #background)//256).astype(np.uint8), out_image, h=h)
    return out_image

