import os
import os.path as osp
import time
import math
import random

import cv2
import lmdb
import numpy as np
import torch
from torch.utils import data as data

import utils as util
from utils.registry import DATASET_REGISTRY
from .degradations import circular_lowpass_kernel, random_mixed_kernels_Info


@DATASET_REGISTRY.register()
class DASRDataset(data.Dataset):

    def __init__(self, opt):
        super(DASRDataset, self).__init__()
        self.opt = opt
        
        self.gt_paths, self.gt_sizes = util.get_image_paths(
            opt["data_type"], opt["dataroot"]
        )

        if opt["data_type"] == "lmdb":
            self.lmdb_envs = False

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.blur_kernel_size_minimum = opt['blur_kernel_size_minimum']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.blur_kernel_size2_minimum = opt['blur_kernel_size2_minimum']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(math.ceil(self.blur_kernel_size_minimum / 2), math.ceil(self.blur_kernel_size / 2))]
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.kernel_range2 = [2 * v + 1 for v in range(math.ceil(self.blur_kernel_size2_minimum / 2), math.ceil(self.blur_kernel_size2 / 2))]
        self.pulse_tensor2 = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor2[10, 10] = 1

        # standard_degrade_one_stage
        self.blur_kernel_size_standard1 = opt['blur_kernel_size_standard1']
        self.blur_kernel_size_minimum_standard1 = opt['blur_kernel_size_minimum_standard1']
        self.kernel_list_standard1 = opt['kernel_list_standard1']
        self.kernel_prob_standard1 = opt['kernel_prob_standard1']
        self.blur_sigma_standard1 = opt['blur_sigma_standard1']
        self.betag_range_standard1 = opt['betag_range_standard1']
        self.betap_range_standard1 = opt['betap_range_standard1']
        self.sinc_prob_standard1 = opt['sinc_prob_standard1']

        # weak_degrade_one_stage
        self.blur_kernel_size_weak1 = opt['blur_kernel_size_weak1']
        self.blur_kernel_size_minimum_weak1 = opt['blur_kernel_size_minimum_weak1']
        self.kernel_list_weak1 = opt['kernel_list_weak1']
        self.kernel_prob_weak1 = opt['kernel_prob_weak1']
        self.blur_sigma_weak1 = opt['blur_sigma_weak1']
        self.betag_range_weak1 = opt['betag_range_weak1']
        self.betap_range_weak1 = opt['betap_range_weak1']
        self.sinc_prob_weak1 = opt['sinc_prob_weak1']

    def _init_lmdb(self, dataroots):
        envs = []
        for dataroot in dataroots:
            envs.append(
                lmdb.open(
                    dataroot, readonly=True, lock=False, readahead=False, meminit=False
                )
            )
        self.lmdb_envs = True
        return envs[0] if len(envs) == 1 else envs

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb" and (not self.lmdb_envs):
            self.env = self._init_lmdb([self.opt["dataroot"]])

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.gt_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.gt_sizes[index].split("_")]
        else:
            resolution = None
        img_gt = util.read_img(self.env, gt_path, resolution)

        # -------------------- augmentation for training: flip, rotation -------------------- #
        H, W, C = img_gt.shape
        cropped_size = self.opt["img_size"]
        # randomly crop
        rnd_h = random.randint(0, max(0, H - cropped_size))
        rnd_w = random.randint(0, max(0, W - cropped_size))
        img_gt = img_gt[rnd_h : rnd_h + cropped_size, rnd_w : rnd_w + cropped_size, ...]
        # augment
        img_gt = util.augment([img_gt], self.opt["use_flip"], self.opt["use_rot"])

        # change color space if necessary
        if self.opt["color"]:
            img_gt = util.channel_convert(self.opt["color"], [img_gt])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_gt.shape[2] == 3:
            img_gt = img_gt[:, :, [2, 1, 0]]
        
        img_gt = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))
        ).float()

        return_d = {}

        # severe_degrade_two_stage
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2_info = random_mixed_kernels_Info(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)
            kernel2 = kernel2_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
            kernel_sinc_info = {'kernel':sinc_kernel, 'kernel_size':kernel_size, 'omega_c':omega_c}
        else:
            sinc_kernel = self.pulse_tensor2
            kernel_sinc_info = {'kernel': sinc_kernel, 'kernel_size': 0, 'omega_c': 0}

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        kernel_info['kernel'] = kernel
        kernel2_info['kernel'] = kernel2

        return_d['severe_degrade_two_stage'] = {'gt': img_gt, 'kernel1': kernel_info, 'kernel2': kernel2_info, 'sinc_kernel': kernel_sinc_info, 'gt_path': gt_path}
        # return_d = {'gt': img_gt, 'kernel1': kernel_info, 'kernel2': kernel2_info, 'sinc_kernel': kernel_sinc_info, 'gt_path': gt_path}

        kernel_info = {}

        # standard_degrade_one_stage

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob_standard1']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list_standard1,
                self.kernel_prob_standard1,
                kernel_size,
                self.blur_sigma_standard1,
                self.blur_sigma_standard1, [-math.pi, math.pi],
                self.betag_range_standard1,
                self.betap_range_standard1,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel = torch.FloatTensor(kernel)
        kernel_info['kernel'] = kernel

        return_d['standard_degrade_one_stage'] = {'gt': img_gt, 'kernel1': kernel_info, 'gt_path': gt_path}

        kernel_info = {}

        # weak_degrade_one_stage

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob_weak1']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel_info = random_mixed_kernels_Info(
                self.kernel_list_weak1,
                self.kernel_prob_weak1,
                kernel_size,
                self.blur_sigma_weak1,
                self.blur_sigma_weak1, [-math.pi, math.pi],
                self.betag_range_weak1,
                self.betap_range_weak1,
                noise_range=None)
            kernel = kernel_info['kernel']

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel = torch.FloatTensor(kernel)
        kernel_info['kernel'] = kernel

        return_d['weak_degrade_one_stage'] = {'gt': img_gt, 'kernel1': kernel_info, 'gt_path': gt_path}

        return return_d

    def __len__(self):
        return len(self.gt_paths)
