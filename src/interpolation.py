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

def process_pixel_RBF(args):
    x, y, MLIC_resized, L_poses, directions_uv, directions_grid, regular_grid_dim = args
    model_xy = Rbf(L_poses[:, 0], L_poses[:, 1], MLIC_resized[:, y, x], function='linear', smooth=1, )
    values = model_xy(directions_uv[:, 0], directions_uv[:, 1])
    regular_grid = np.zeros(regular_grid_dim)
    regular_grid[directions_grid[:, 1], directions_grid[:, 0]] = values
    
    return x, y, regular_grid

def interpolation(filename, coin_dim, regular_grid_dim, method, nprocesses=-1):
    
    regular_grids = {}

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
    
    # Prepare arguments for parallel processing
    args_list = [
        (x, y, MLIC_resized, L_poses, directions_uv, directions_grid, regular_grid_dim)
        for x in range(coin_dim[0])
        for y in range(coin_dim[1])
    ]
    
    if method == "RBF":
        process_pixel = process_pixel_RBF
    else:
        raise ValueError(f"Method {method} not recognized")
    
    # Create a pool of worker processes
    with multiprocessing.Pool(nprocesses) as pool:
        # Use tqdm to display progress
        results = list(tqdm(pool.imap_unordered(process_pixel, args_list), total=len(args_list)))

    # Collect the results
    for x, y, regular_grid in results:
        regular_grids[(x, y)] = regular_grid

    # Save regular grids
    results_path = f"./results/{method}"
    os.makedirs(results_path, exist_ok=True)
    
    np.savez_compressed(os.path.join(results_path, f"{filename}.npz"), regular_grids)

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
                    method="RBF",
                    nprocesses=args.nprocesses)
