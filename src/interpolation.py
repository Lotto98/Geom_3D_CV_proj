from typing import Tuple
import numpy as np
import itertools
import cv2 as cv
from compute_light import load_light_results
from tqdm import tqdm
import multiprocessing
import torchrbf
import torch
import os
from scipy.interpolate import Rbf

def compute_ptm_coefficients_2d(MLIC:np.ndarray, light_poses:np.ndarray) -> np.ndarray:
    """
    Compute PTM coefficients for each pixel using 2D light poses.
    
    Args:
        MLIC (numpy array): Tensor of shape (n_images, H, W). The images under different light conditions.
        light_poses (numpy array): Light poses (n_images, 2). The light directions in UV space.
        
    Returns:
        numpy array: PTM coefficients (H, W, 6).
    """
    n_images, H, W = MLIC.shape
    
    # Flatten images for easier processing
    intensities = MLIC.reshape(n_images, -1)  # Shape: (n_images, H*W)
    
    # Build the 2D design matrix
    u, v = light_poses[:, 0], light_poses[:, 1]
    L = np.stack([ u**2, v**2, u*v, u, v, np.ones_like(u) ], axis=1)  # Shape: (n_images, 6)
    
    # Compute coefficients for each pixel
    L_pseudo_inv = np.linalg.pinv(L)  # Shape: (6, n_images)
    coeffs = L_pseudo_inv @ intensities  # Shape: (6, H*W)
    
    return coeffs.T.reshape(H, W, 6)

def render_ptm_2d(coeffs:np.ndarray, light_dir:np.ndarray) -> np.ndarray:
    """
    Render a PTM image under a given 2D light direction.
    
    Args:
        coeffs (numpy array): PTM coefficients (H, W, 6).
        light_dir (numpy array): Light direction (2,).
        
    Returns:
        numpy array: Rendered image.
    """
    u, v = light_dir
    L = np.array([ u**2, v**2, u*v, u, v, np.ones_like(u) ])  # Shape: (6,)
    
    rendered = np.sum(coeffs * L, axis=2)  # Weighted sum using coefficients
    rendered = np.clip(rendered, 0, 255).astype(np.uint8)  # Ensure valid pixel range
    return rendered

def process_pixel_RBF(args)->Tuple[int, int, np.ndarray]:
    """
    Process a single pixel using RBF interpolation, for parallel processing.

    Args:
        args: list of arguments (x, y, MLIC_resized, L_poses, directions_uv, directions_grid, regular_grid_dim):
            x (int): x-coordinate of the pixel
            y (int): y-coordinate of the pixel
            MLIC_resized (np.ndarray): resized MLIC tensor
            L_poses (np.ndarray): light poses
            directions_uv (np.ndarray): light directions in UV space
            directions_grid (np.ndarray): light directions in grid space
            regular_grid_dim (np.ndarray): dimensions of the regular grid

    Returns:
        Tuple[int, int, np.ndarray]: x, y, regular_grid for the pixel (x, y)
    """
    # Unpack arguments
    x, y, MLIC_resized, L_poses, directions_uv, directions_grid, regular_grid_dim = args
    
    # Perform RBF interpolation
    model_xy = Rbf(L_poses[:, 0], L_poses[:, 1], MLIC_resized[:, y, x], function='linear', smooth=1, )
    
    # Compute values for the regular grid
    values = model_xy(directions_uv[:, 0], directions_uv[:, 1])
    values = np.clip(values, 0, 255)
    
    # Populate the regular grid
    regular_grid = np.zeros(regular_grid_dim, dtype=np.uint8)
    regular_grid[directions_grid[:, 1], directions_grid[:, 0]] = values
    
    return x, y, regular_grid

def process_pixel_RBF_cuda(x:int, y:int, val_train:torch.Tensor, directions_train:torch.Tensor, 
                            direction_inference:torch.Tensor, directions_grid:np.ndarray, 
                            regular_grid_dim:np.ndarray)->Tuple[int, int, np.ndarray]:
    """
    Process a single pixel using RBF interpolation, for cuda processing.

    Args:
        x (int): x-coordinate of the pixel
        y (int): y-coordinate of the pixel
        val_train (torch.Tensor): values of intensity for each light direction for the pixel (x, y)
        directions_train (torch.Tensor): light directions computed by analysis.py
        direction_inference (torch.Tensor): light directions for the regular grid in UV space
        directions_grid (np.ndarray): light directions for the regular grid in grid space
        regular_grid_dim (np.ndarray): dimensions of the regular grid

    Returns:
        Tuple[int, int, np.ndarray]: x, y, regular_grid for the pixel (x, y)
    """
    
    # Fit the RBF model
    model_xy = torchrbf.RBFInterpolator(directions_train, val_train[:, y, x], kernel='linear', smoothing=1, device="cuda")
    
    # Compute values for the regular grid (actual interpolation)
    values = model_xy(direction_inference)
    values = np.clip(values.cpu().numpy(), 0, 255)
    
    # Populate the regular grid
    regular_grid = np.zeros(regular_grid_dim, dtype=np.uint8)
    regular_grid[directions_grid[:, 1], directions_grid[:, 0]] = values
    
    return x, y, regular_grid

def interpolation(coin_number:int, coin_dim:Tuple[int, int], regular_grid_dim:Tuple[int, int], method:str, nprocesses:int=-1):
    """
    Interpolate the light for a given coin using the specified method and save the results to disk.

    Args:
        coin_number (int): coin number to interpolate.
        coin_dim (Tuple[int, int]): coin image dimensions to resize to.
        regular_grid_dim (Tuple[int, int]): regular grid dimensions to interpolate.
        method (str): interpolation method.
        nprocesses (int, optional): number of processors to use. Defaults to -1. RBF only.

    Raises:
        ValueError: if the method is not recognized. 
    """

    if nprocesses == -1:
        nprocesses = multiprocessing.cpu_count()
    
    # Load data
    filename = f"coin{coin_number}"
    MLIC, L_poses, U_hat, V_hat = load_light_results(filename)

    # Resize images to the desired dimensions
    MLIC_resized = []
    for coin in MLIC:
        coin = cv.resize(coin, coin_dim)
        MLIC_resized.append(coin)
    MLIC_resized = np.array(MLIC_resized)

    # Prepare regular grid directions in UV space and grid space
    directions_u = np.linspace(-1, 1, regular_grid_dim[0])
    directions_v = np.linspace(-1, 1, regular_grid_dim[1])
    directions_uv = np.array([np.array(uv) for uv in itertools.product(directions_u, directions_v)])
    directions_grid = np.array([np.array([x, y]) for x, y in itertools.product(range(0, regular_grid_dim[0]), range(0, regular_grid_dim[1]))])
    
    if method == "RBF":
        
        # Prepare arguments for parallel processing
        args_list = [
            (x, y, MLIC_resized, L_poses, directions_uv, directions_grid, regular_grid_dim)
            for x in range(coin_dim[0])
            for y in range(coin_dim[1])
        ]
        
        # Create a pool of worker processes and execute the function in parallel
        with multiprocessing.Pool(nprocesses) as pool:
            results = list(tqdm(pool.imap_unordered(process_pixel_RBF, args_list), total=len(args_list), desc="Rendering RBF"))

        # Collect the results into a tensor of regular grids
        regular_grids = np.zeros((coin_dim[1], coin_dim[0], regular_grid_dim[1], regular_grid_dim[0]), dtype=np.uint8)
        for x, y, regular_grid in tqdm(results, desc="Collecting results"):
            regular_grids[y, x] = regular_grid
            
    elif method == "PTM":
        # Compute PTM coefficients: alphas
        coeff = compute_ptm_coefficients_2d(MLIC_resized, L_poses)
        
        # Render PTM for each pixel, using the regular grid directions
        regular_grids = np.zeros((coin_dim[1], coin_dim[0], regular_grid_dim[1], regular_grid_dim[0]), dtype=np.uint8)
        for (u,v), (x,y) in tqdm(zip(directions_uv, directions_grid), desc="Rendering PTM", total=len(directions_uv)):
            regular_grids[:, :, y, x] = render_ptm_2d(coeff, np.array([u,v]))
        
    elif method == "RBF_cuda":
        
        # Prepare data for CUDA processing: GPU tensors
        directions_train = torch.tensor(L_poses[:, 0:2], dtype=torch.float32).cuda()
        val_train = torch.tensor(MLIC_resized, dtype=torch.float32).cuda()
        direction_inference = torch.tensor(directions_uv[:, 0:2], dtype=torch.float32).cuda()
        
        #Execute the RBF interpolation using CUDA
        results = []
        for x,y in tqdm(itertools.product( range(coin_dim[0]), range(coin_dim[1]) ), total=coin_dim[0]*coin_dim[1]):
            x, y, regular_grid = process_pixel_RBF_cuda(x,y, val_train, directions_train, direction_inference, directions_grid, regular_grid_dim)
            results.append((x,y,regular_grid))
        
        # Collect the results into a tensor of regular grids
        regular_grids = np.zeros((coin_dim[1], coin_dim[0], regular_grid_dim[1], regular_grid_dim[0]), dtype=np.uint8)
        for x, y, regular_grid in tqdm(results, desc="Collecting results"):
            regular_grids[y, x] = regular_grid
    else:
        raise ValueError(f"Method {method} not recognized")

    # Save regular grids tensor to disk using numpy compressed format (.npz) to save space
    results_path = f"./results/{method}"
    os.makedirs(results_path, exist_ok=True)
    
    np.savez_compressed(os.path.join(results_path, f"{filename}_{coin_dim}_{regular_grid_dim}.npz"), 
                        regular_grids=regular_grids,
                        regular_grid_dim=regular_grid_dim,
                        coin_dim=coin_dim)