from matplotlib import pyplot as plt
import argparse
import numpy as np

from compute_light import load_light_results

import cv2 as cv

def plot_pixel(x, y, MLIC, L_poses, regular_grids=[], methods=[]):
    
    assert len(regular_grids) == len(methods), "Number of regular grids and methods must be the same"
    
    print(f"Pixel coordinates: {args.x}, {args.y} for dimensions {MLIC.shape[1:]}")
    
    fig, ax = plt.subplots(1, 1+len(regular_grids), figsize=(5+5*len(regular_grids), 5))
    
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    
    ax[0].set_ylim(-1, 1)
    ax[0].set_xlim(-1, 1)
    scatter = ax[0].scatter(L_poses[:, 0], L_poses[:, 1], c=MLIC[:, y, x], cmap='viridis', s=2, vmax=255, vmin=0,)
    ax[0].set_title(f"f({x}, {y}, ...)")
    ax[0].set_xlabel("U")
    ax[0].set_ylabel("V")

    for i, regular_grid in enumerate(regular_grids):
        image = ax[i+1].matshow(regular_grid[y, x], cmap='viridis', vmax=255, vmin=0,)
        ax[i+1].set_title(f"Interpolation {methods[i]}")
        ax[i+1].axis('off')
        ax[i+1].invert_yaxis()

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    plt.colorbar(mappable=scatter, cax=cbar_ax)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolation visualizer")
    parser.add_argument("--coin", type=int, required=True, help="Coin number: 1, 2, 3, 4")
    parser.add_argument("--x", type=int, required=True, help="X coordinate of the pixel to visualize")
    parser.add_argument("--y", type=int, required=True, help="Y coordinate of the pixel to visualize")
    
    parser.add_argument("--methods", help="Method of interpolation to visualize", nargs='+', 
                        type=str, default=["RBF", "PTM"])
    args=parser.parse_args()
    
    filename = f"coin{args.coin}"
    
    MLIC, L_poses, U_hat, V_hat = load_light_results(filename)
    
    to_plot = {}
    coin_dims = []
    
    for m in args.methods:
        loaded = np.load(f"./results/{m}/{filename}.npz", allow_pickle=True)
        regular_grids = loaded["regular_grids"]
        regular_grid_dim = loaded["regular_grid_dim"]
        coin_dim = loaded["coin_dim"]
        
        to_plot[m] = (regular_grids)
        coin_dims.append(coin_dim)
    
    assert np.all([coin_dim == coin_dims[0] for coin_dim in coin_dims]), \
        "All interpolated coins must have the same dimensions"
    
    MLIC_resized = np.array([cv.resize(coin, coin_dim) for coin in MLIC])
        
    if args.x >= coin_dim[1] or args.y >= coin_dim[0]:
        raise ValueError(f"Pixel coordinates out of bounds: {args.x}, {args.y} for dimensions {coin_dim}")
    
    plot_pixel(args.x, args.y, MLIC_resized, L_poses, [r for r in to_plot.values()], [m for m in to_plot.keys()])