import numpy as np
import itertools
import cv2 as cv
from analysis import load_results
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

import os

from scipy.interpolate import Rbf

import argparse

def compute_ptm_coefficients_2d(MLIC, light_poses):
    """
    Compute PTM coefficients for each pixel using 2D light poses.
    
    Args:
        MLIC (numpy array): Tensor of shape (n_images, H, W).
        light_poses (numpy array): Light poses (n_images, 2).
        
    Returns:
        numpy array: PTM coefficients (H, W, 3).
    """
    n_images, H, W = MLIC.shape
    
    # Flatten images for easier processing
    intensities = MLIC.reshape(n_images, -1)  # Shape: (n_images, H*W)
    
    # Build the 2D design matrix
    x, y = light_poses[:, 0], light_poses[:, 1]
    L = np.stack([x**2, y**2], axis=1)  # Shape: (n_images, 2)
    
    # Normalize the design matrix
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    
    # Compute coefficients for each pixel
    L_pseudo_inv = np.linalg.pinv(L)  # Shape: (3, n_images)
    coeffs = L_pseudo_inv @ intensities  # Shape: (3, H*W)
    
    return coeffs.T.reshape(H, W, 2)

def render_ptm_2d(coeffs, light_dir):
    """
    Render a PTM image under a given 2D light direction.
    
    Args:
        coeffs (numpy array): PTM coefficients (H, W, 3).
        light_dir (numpy array): Light direction (2,).
        
    Returns:
        numpy array: Rendered image.
    """
    x, y = light_dir
    L = np.array([x**2, y**2])  # Shape: (3,)
    
    rendered = np.sum(coeffs * L, axis=2)  # Weighted sum using coefficients
    rendered = np.clip(rendered, 0, 255).astype(np.uint8)  # Ensure valid pixel range
    return rendered

def process_pixel_RBF(args):
    x, y, MLIC_resized, L_poses, directions_uv, directions_grid, regular_grid_dim = args
    model_xy = Rbf(L_poses[:, 0], L_poses[:, 1], MLIC_resized[:, y, x], function='linear', smooth=1, )
    values = model_xy(directions_uv[:, 0], directions_uv[:, 1])
    regular_grid = np.zeros(regular_grid_dim, dtype=np.uint8)
    regular_grid[directions_grid[:, 1], directions_grid[:, 0]] = values
    
    return x, y, regular_grid

def interpolation(filename, coin_dim, regular_grid_dim, method, nprocesses=-1):

    if nprocesses == -1:
        nprocesses = multiprocessing.cpu_count()
    
    # Load data
    MLIC, L_poses, U_hat, V_hat = load_results(filename)

    MLIC_resized = []
    for coin in MLIC:
        coin = cv.resize(coin, coin_dim)
        MLIC_resized.append(coin)
    MLIC_resized = np.array(MLIC_resized)

    directions_u = np.linspace(-1, 1, regular_grid_dim[0])
    directions_v = np.linspace(-1, 1, regular_grid_dim[1])
    directions_uv = np.array([np.array(uv) for uv in itertools.product(directions_u, directions_v)])

    directions_grid = np.array([np.array([x, y]) for x, y in itertools.product(range(0, regular_grid_dim[0]), range(0, regular_grid_dim[1]))])
    
    print(method)
    
    if method == "RBF":
        
        # Prepare arguments for parallel processing
        args_list = [
            (x, y, MLIC_resized, L_poses, directions_uv, directions_grid, regular_grid_dim)
            for x in range(coin_dim[0])
            for y in range(coin_dim[1])
        ]
        
        # Create a pool of worker processes
        with multiprocessing.Pool(nprocesses) as pool:
            # Use tqdm to display progress
            results = list(tqdm(pool.imap_unordered(process_pixel_RBF, args_list), total=len(args_list), desc="Rendering RBF"))

        # Collect the results
        regular_grids = np.zeros((coin_dim[1], coin_dim[0], regular_grid_dim[1], regular_grid_dim[0]), dtype=np.uint8)
        for x, y, regular_grid in results:
            regular_grids[y, x] = regular_grid
            
    elif method == "PTM":
        
        coeff = compute_ptm_coefficients_2d(MLIC_resized, L_poses)
        
        regular_grids = np.zeros((coin_dim[1], coin_dim[0], regular_grid_dim[1], regular_grid_dim[0]), dtype=np.uint8)
        
        for (u,v), (x,y) in tqdm(zip(directions_uv, directions_grid), desc="Rendering PTM", total=len(directions_uv)):
            regular_grids[:, :, y, x] = render_ptm_2d(coeff, np.array([u,v]))
    else:
        raise ValueError(f"Method {method} not recognized")

    # Save regular grids
    results_path = f"./results/{method}"
    os.makedirs(results_path, exist_ok=True)
    
    np.savez_compressed(os.path.join(results_path, f"{filename}.npz"), 
                        regular_grids=regular_grids,
                        regular_grid_dim=regular_grid_dim,
                        coin_dim=coin_dim)

if __name__ == "__main__":
    
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--filename", type=str, required=True)
    argparse.add_argument("--coin_dim", type=int, required=True, nargs='+')
    argparse.add_argument("--regular_grid_dim", type=int, required=True, nargs='+')
    argparse.add_argument("--method", type=str, required=True)
    argparse.add_argument("--nprocesses", type=int, required=True)
    args = argparse.parse_args()
    
    filename = args.filename
    coin_dim = tuple(args.coin_dim)
    regular_grid_dim = tuple(args.regular_grid_dim)
    
    #print(f"Interpolating {filename} with coin_dim={coin_dim} and regular_grid_dim={regular_grid_dim}")
    
    interpolation(filename=filename, 
                    coin_dim=coin_dim, 
                    regular_grid_dim=regular_grid_dim, 
                    method=args.method,
                    nprocesses=args.nprocesses)
