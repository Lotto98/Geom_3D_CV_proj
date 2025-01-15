import numpy as np
import itertools
import cv2 as cv
from analysis import load_results
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

def fit_model(x, y, MLIC_resized, L_poses):
    from scipy.interpolate import Rbf
    model = Rbf(L_poses[:, 0], L_poses[:, 1], MLIC_resized[:, y, x], function='linear', smooth=1)
    return model

def plot_pixel(x, y, MLIC, L_poses, regular_grids={}):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_ylim(-1, 1)
    ax[0].set_xlim(-1, 1)
    scatter = ax[0].scatter(L_poses[:, 0], L_poses[:, 1], c=MLIC[:, y, x], cmap='viridis', s=2)
    ax[0].set_title(f"f({x}, {y}, ...)")
    ax[0].set_xlabel("U")
    ax[0].set_ylabel("V")
    fig.colorbar(scatter, ax=ax[0])

    if regular_grids != {}:
        ax[1].matshow(regular_grids[(x, y)])
        ax[1].set_title("Interpolation")
    else:
        ax[1].set_title("No interpolation")
    ax[1].axis('off')
    plt.gca().invert_yaxis()

    plt.show()

def process_pixel(args):
    x, y, MLIC_resized, L_poses, directions_uv, directions_grid, regular_grid_dim = args
    model_xy = fit_model(x, y, MLIC_resized, L_poses)
    values = model_xy(directions_uv[:, 0], directions_uv[:, 1])

    regular_grid = np.zeros((regular_grid_dim[0], regular_grid_dim[1]))
    for (x_grid, y_grid), val in zip(directions_grid, values):
        regular_grid[y_grid, x_grid] = val

    return (x, y, regular_grid)

def interpolation():
    filename = "coin1"
    resize_dim = (32, 32)
    regular_grid_dim = (100, 100)

    regular_grids = {}

    # Load data
    MLIC, L_poses, U_hat, V_hat = load_results(filename)

    MLIC_resized = []
    for coin in MLIC:
        coin = cv.resize(coin, resize_dim)
        MLIC_resized.append(coin)
    MLIC_resized = np.array(MLIC_resized)

    directions_u = np.linspace(-1, 1, regular_grid_dim[0])
    directions_v = np.linspace(-1, 1, regular_grid_dim[1])
    directions_uv = np.array([np.array(uv) for uv in itertools.product(directions_u, directions_v)])

    directions_grid = np.array([np.array([x, y]) for x, y in itertools.product(range(0, regular_grid_dim[0]), range(0, regular_grid_dim[1]))])

    # Prepare arguments for parallel processing
    args_list = [
        (x, y, MLIC_resized, L_poses, directions_uv, directions_grid, regular_grid_dim)
        for x in range(resize_dim[0])
        for y in range(resize_dim[1])
    ]

    # Create a pool of worker processes
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        # Use tqdm to display progress
        results = list(tqdm(pool.imap(process_pixel, args_list), total=len(args_list)))

    # Collect the results
    for x, y, regular_grid in results:
        regular_grids[(x, y)] = regular_grid

    # Save regular grids
    np.save("regular_grids.npy", regular_grids)

if __name__ == "__main__":
    interpolation()

    regular_grids = np.load("regular_grids.npy", allow_pickle=True).item()

    # Load data
    MLIC, L_poses, U_hat, V_hat = load_results("coin1")

    MLIC_resized = []
    for coin in MLIC:
        coin = cv.resize(coin, (16, 16))
        MLIC_resized.append(coin)
    MLIC_resized = np.array(MLIC_resized)

    # Plot pixel
    plot_pixel(10, 10, MLIC_resized, L_poses, regular_grids)
    plt.show()
