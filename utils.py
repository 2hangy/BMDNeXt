import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2

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

def visualize_input_and_features(data, features, stage_idx, max_channels=None):
    data_slice = data.cpu().numpy()[0, :, :, :, data.shape[-1] // 2]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    im = ax.imshow(np.transpose(data_slice, (1, 2, 0)), cmap=cm.gray)
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    plt.show()

    feat_maps = features[stage_idx].data.cpu().numpy()
    depth = feat_maps.shape[-1]
    n_channels = feat_maps.shape[1]
    
    if max_channels is None:
        max_channels = n_channels
    
    nx = int(np.ceil(np.sqrt(max_channels)))
    ny = int(np.ceil(max_channels / nx))
    
    fig, axs = plt.subplots(ny, nx, figsize=(nx*4, ny*4), dpi=200)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for i in range(max_channels):
        row = i // nx
        col = i % nx
        feat = feat_maps[0, i, :, :, depth//2] 
        ax = axs[row, col]
        im = ax.imshow(feat, cmap=cm.viridis)
        ax.axis('off')
        
    plt.tight_layout(pad=0.2)
    plt.show()