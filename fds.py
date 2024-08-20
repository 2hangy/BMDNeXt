# This file is modified from https://github.com/YyzHarry/imbalanced-regression under the MIT License.

import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import calibrate_mean_var

print = logging.info

class MS_AFDS(nn.Module):
    def __init__(self, feature_dim, bucket_num=100, bucket_start=3, start_update=0, start_smooth=1,
                 kernel_sizes=[3, 5, 7], sigmas=[1, 2, 3], momentum=0.9, epsilon=1e-5):
        super(MS_AFDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_windows = [self._get_kernel_window(ks, sigma) for ks, sigma in zip(kernel_sizes, sigmas)]
        self.half_ks = [(ks - 1) // 2 for ks in kernel_sizes]
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth
        self.epsilon = epsilon

        self.register_buffer('epoch', torch.zeros(1).fill_(start_update))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start))

    @staticmethod
    def _get_kernel_window(ks, sigma):
        assert ks % 2 == 1, "Kernel size must be odd!"
        half_ks = (ks - 1) // 2
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        base_kernel = np.array(base_kernel, dtype=np.float32)
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        return torch.tensor(kernel_window, dtype=torch.float32).to('cuda')

    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.zero_()
        for kernel_window, half_ks in zip(self.kernel_windows, self.half_ks):
            self.smoothed_mean_last_epoch += F.conv1d(
                input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                            pad=(half_ks, half_ks), mode='reflect'),
                weight=kernel_window.view(1, 1, -1), padding=0
            ).permute(2, 1, 0).squeeze(1)
            self.smoothed_var_last_epoch += F.conv1d(
                input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                            pad=(half_ks, half_ks), mode='reflect'),
                weight=kernel_window.view(1, 1, -1), padding=0
            ).permute(2, 1, 0).squeeze(1)
        self.smoothed_mean_last_epoch /= len(self.kernel_windows)
        self.smoothed_var_last_epoch /= len(self.kernel_windows)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            print(f"Updated smoothed statistics on Epoch [{epoch}]!")

    def update_running_stats(self, features, labels, epoch):
        if epoch < self.epoch:
            return

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        for label in torch.unique(labels):
            if self.bucket_start <= label < self.bucket_num:
                if label == self.bucket_start:
                    curr_feats = features[labels <= label]
                elif label == self.bucket_num - 1:
                    curr_feats = features[labels >= label]
                else:
                    curr_feats = features[labels == label]
                curr_num_sample = curr_feats.size(0)
                curr_mean = torch.mean(curr_feats, 0)
                curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

                self.num_samples_tracked[int(label - self.bucket_start)] += curr_num_sample
                factor = self.momentum if self.momentum is not None else \
                    (1 - curr_num_sample / float(self.num_samples_tracked[int(label - self.bucket_start)]))
                factor = 0 if epoch == self.start_update else factor
                self.running_mean[int(label - self.bucket_start)] = \
                    (1 - factor) * curr_mean + factor * self.running_mean[int(label - self.bucket_start)]
                self.running_var[int(label - self.bucket_start)] = \
                    (1 - factor) * curr_var + factor * self.running_var[int(label - self.bucket_start)]

        print(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels, epoch):
        if epoch < self.start_smooth:
            return features

        labels = labels.squeeze(1)
        for label in torch.unique(labels):
            if self.bucket_start <= label < self.bucket_num:
                bucket_idx = int(label - self.bucket_start)
                if label == self.bucket_start:
                    mask = labels <= label
                elif label == self.bucket_num - 1:
                    mask = labels >= label
                else:
                    mask = labels == label
                alpha = torch.exp(-self.num_samples_tracked[bucket_idx] / (torch.mean(self.num_samples_tracked) + self.epsilon))
                features[mask] = calibrate_mean_var(
                    features[mask],
                    self.running_mean_last_epoch[bucket_idx],
                    self.running_var_last_epoch[bucket_idx],
                    (1 - alpha) * self.running_mean_last_epoch[bucket_idx] + alpha * self.smoothed_mean_last_epoch[bucket_idx],
                    (1 - alpha) * self.running_var_last_epoch[bucket_idx] + alpha * self.smoothed_var_last_epoch[bucket_idx]
                )
        return features