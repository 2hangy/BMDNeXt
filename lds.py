# This file is modified from https://github.com/YyzHarry/imbalanced-regression under the MIT License.

from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang
from collections import Counter
import numpy as np
import torch

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

def get_bin_idx(label, min_val=0.4, max_val=1.9, num_bins=25):
    bin_width = (max_val - min_val) / num_bins
    bin_idx = int((label - min_val) // bin_width)
    
    return min(bin_idx, num_bins - 1)

def ms_alds(target, min_val=0.4, max_val=1.9, num_bins=25, K=[3, 5, 7], beta=0.5, gamma=1.0, lambda_min=0.5, lambda_max=1.0, epsilon=1e-8):
    bin_index_per_label = [get_bin_idx(label, min_val=min_val, max_val=max_val, num_bins=num_bins) for label in target]
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = np.array([num_samples_of_bins.get(i, 0) for i in range(num_bins)], dtype=float)
    # Compute adaptive kernel bandwidth for each bin
    sigma_i = beta * (np.log(1 / (emp_label_dist / np.sum(emp_label_dist) + epsilon)) + gamma)
    # Compute adaptive regularization coefficient for each bin
    total_emp_dist = np.sum(emp_label_dist)
    lambda_i = lambda_min + (lambda_max - lambda_min) * (emp_label_dist / (total_emp_dist + epsilon))
    
    eff_label_dist = np.zeros(num_bins)
    S = len(K)

    for k_s in K:
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=k_s, sigma=np.mean(sigma_i))
        smoothed_dist = convolve1d(emp_label_dist, lds_kernel_window, mode='constant')
        eff_label_dist += smoothed_dist
    
    eff_label_dist = eff_label_dist / S + lambda_i * emp_label_dist  
    
    adaptive_weights = torch.tensor([1 / (eff_label_dist[get_bin_idx(label, min_val=min_val, max_val=max_val, num_bins=num_bins)] + epsilon) for label in target], device=target.device)

    return emp_label_dist, eff_label_dist, adaptive_weights