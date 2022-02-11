import numpy as np
import random
import math
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F

# img1, img2 are 3D np arrays of (H, W, 3). flow is (H, W, 2).
def random_shift(img1, img2, flow, shift_sigmas=(16,10)):
    u_shift_sigma, v_shift_sigma = shift_sigmas
    # 90% of dx and dy are within [-2*u_shift_sigma, 2*u_shift_sigma] 
    # and [-2*v_shift_sigma, 2*v_shift_sigma].
    # Make sure at most one of dx, dy is large. Otherwise the shift is too difficult.
    if random.random() > 0.5:
        dx = np.random.laplace(0, u_shift_sigma / 4)
        dy = np.random.laplace(0, v_shift_sigma)
    else:
        dx = np.random.laplace(0, u_shift_sigma)
        dy = np.random.laplace(0, v_shift_sigma / 4)
    # Make sure dx and dy are even numbers.
    dx = (int(dx) // 2) * 2
    dy = (int(dy) // 2) * 2

    if dx >= 0 and dy >= 0:
        # img1 is cropped at the bottom-right corner.               img1[:-dy, :-dx]
        img1_bound = (0,  img1.shape[0] - dy,  0,  img1.shape[1] - dx)
        # img2 is shifted by (dx, dy) to the left and up. pixels at (dy, dx) ->(0, 0).
        #                                                           img2[dy:,  dx:]
        img2_bound = (dy, img1.shape[0],       dx, img1.shape[1])
    if dx >= 0 and dy < 0:
        # img1 is cropped at the right side, and shifted to the up. img1[-dy:, :-dx]
        img1_bound = (-dy, img1.shape[0],      0,  img1.shape[1] - dx)
        # img2 is shifted to the left and cropped at the bottom.    img2[:dy,  dx:]
        img2_bound = (0,   img1.shape[0] + dy, dx, img1.shape[1])
    if dx < 0 and dy >= 0:
        # img1 is shifted to the left, and cropped at the bottom.   img1[:-dy, -dx:]
        img1_bound = (0,   img1.shape[0] - dy, -dx, img1.shape[1])
        # img2 is cropped at the right side, and shifted to the up. img2[dy:,  :dx]
        img2_bound = (dy,  img1.shape[0],      0,   img1.shape[1] + dx)
    if dx < 0 and dy < 0:
        # img1 is shifted by (-dx, -dy) to the left and up. img1[-dy:, -dx:]
        img1_bound = (-dy, img1.shape[0],      -dx, img1.shape[1])
        # img2 is cropped at the bottom-right corner.       img2[:dy,  :dx]
        img2_bound = (0,   img1.shape[0] + dy, 0,   img1.shape[1] + dx)

    reversed_12 = random.random() > 0.5

    if reversed_12:
        img1_bound, img2_bound = img2_bound, img1_bound
        flow_delta = (-dx, -dy)
    else:
        flow_delta = (dx,  dy)

    T1, B1, L1, R1 = img1_bound
    T2, B2, L2, R2 = img2_bound
    img1a = img1[T1:B1, L1:R1]
    flowa = flow[T1:B1, L1:R1] - flow_delta
    img2a = img2[T2:B2, L2:R2]

    # Pad img1, img2 and flow by half of (dy, dx).
    dx2, dy2 = abs(dx) // 2, abs(dy) // 2
    # valid_mask: boolean array that indicates the remaining area after cropping/shifting.
    valid_mask = np.ones(img1a.shape[:2], dtype=bool)

    img1a       = np.pad(img1a,         ((dy2, dy2), (dx2, dx2), (0, 0)), 'constant')
    img2a       = np.pad(img2a,         ((dy2, dy2), (dx2, dx2), (0, 0)), 'constant')
    flowa       = np.pad(flowa,         ((dy2, dy2), (dx2, dx2), (0, 0)), 'constant')
    valid_mask  = np.pad(valid_mask,    ((dy2, dy2), (dx2, dx2)), 'constant', constant_values=False)

    return img1a, img2a, flowa, valid_mask
    
class FlowAugmentor:
    def __init__(self, ds_name, crop_size, min_scale=-0.2, max_scale=0.5, spatial_aug_prob=0.8, 
                 blur_kernel=5, blur_sigma=-1, do_flip=True, shift_prob=0, shift_sigmas=(16,10)):
        self.ds_name = ds_name
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = spatial_aug_prob
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # shift augmentation
        self.shift_prob = shift_prob
        # 90% of dx and dy are within [-2*u_shift_sigma, 2*u_shift_sigma] 
        # and [-2*v_shift_sigma, 2*v_shift_sigma].
        self.shift_sigmas = shift_sigmas        

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
            
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow
        
    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        valid = None
        if self.shift_prob > 0 and random.random() < self.shift_prob:
            img1, img2, flow, valid = random_shift(img1, img2, flow, self.shift_sigmas)

        if self.blur_sigma > 0:
            K = self.blur_kernel
            img1 = cv2.GaussianBlur(img1, (K, K), self.blur_sigma)
            img2 = cv2.GaussianBlur(img2, (K, K), self.blur_sigma)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow, valid


class SparseFlowAugmentor:
    def __init__(self, ds_name, crop_size, min_scale=-0.2, max_scale=0.5, 
                 spatial_aug_prob=0.8, do_flip=False, shift_prob=0, shift_sigmas=(16,10)):
        self.ds_name = ds_name
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = spatial_aug_prob
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

        # shift augmentation
        self.shift_prob = shift_prob
        # 90% of dx and dy are within [-2*u_shift_sigma, 2*u_shift_sigma] 
        # and [-2*v_shift_sigma, 2*v_shift_sigma].
        self.shift_sigmas = shift_sigmas

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    # crop_size: minimal image size after resizing. 
    # Images are cropped to crop_size at the end of spatial_transform().
    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        # min_scale: the scale is at least min_scale.
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)
        
        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
            
        if self.do_flip:
            if np.random.rand() < 0.5: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        # print(img1.shape)
        return img1, img2, flow, valid

    # img1, img2: (H, W, 3). valid: (H, W)
    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        valid2 = None
        if self.shift_prob > 0 and random.random() < self.shift_prob:
            img1, img2, flow, valid2 = random_shift(img1, img2, flow, self.shift_sigmas)

        if valid2 is not None:
            valid = valid * valid2

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid
