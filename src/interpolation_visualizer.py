from functools import partial
import os
from typing import List, Tuple
from matplotlib import pyplot as plt
import argparse
import numpy as np
from tqdm import tqdm

from compute_light import load_light_results

import cv2 as cv
import io
import gc

# Global variables for the light source selection
x_inp:int = 50
y_inp:int = 50
drawing:bool = False

image_size:int = 512
def plot(x:int, y:int, pixel_selector:np.ndarray):
    """
    Plot the pixel position on the screen.

    Args:
        x (int): x coordinate in input in [0, image_size].
        y (int): y coordinate in input in [0, image_size].
        pixel_selector (np.ndarray): image to plot the pixel on.
    """
    
    global image_size
    
    x = int(x)
    y = int(y)
    
    img = cv.circle(pixel_selector.copy(), (x, y), 5, (0, 0, 255), -1)
    
    cv.imshow("Pixel selector", img)
    
def set_pixel_position(x:int, y:int, pixel_selector:np.ndarray):
    """
    Set the pixel position and plot it on the screen.

    It also updates the global variables x_inp and y_inp.
    Args:
        x (int): input x coordinate in [0, image_size].
        y (int): input y coordinate in [0, image_size].
        pixel_selector (np.ndarray): image to plot the pixel on.
    """
    
    global x_inp, y_inp, image_size
    
    plot(x, y, pixel_selector)
    
    x_inp = x
    y_inp = y

# mouse callback function
def mouse_callback(event:int,x:int,y:int,flags,param, pixel_selector:np.ndarray):
    """
    Callback function to handle mouse events. 
    It sets the pixel position when the left mouse button is clicked and gragged

    Args:
        event (int): event type.
        x (int): x coordinate.
        y (int): y coordinate.
    """
    global drawing
        
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        set_pixel_position(x, y, pixel_selector)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            set_pixel_position(x, y, pixel_selector)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        set_pixel_position(x, y, pixel_selector)

def get_img_from_fig(fig:plt.figure, dpi=70)->np.ndarray:
    """
    Get a numpy array from a matplotlib figure.

    Args:
        fig (plt.figure): matplotlib figure.
        dpi (int, optional): Number of dots per inch. Defaults to 100.

    Returns:
        np.ndarray: image as a numpy array.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    del buf
    gc.collect()
    img = cv.imdecode(img_arr, 1)

    return img

def plot_pixel(x:int, y:int, MLIC:np.ndarray, L_poses:np.ndarray, regular_grids:List[np.ndarray] = [], methods:List[str] = []):
    """
    Plot the observed light intensities and the interpolated light intensities.

    Args:
        x (int): x-coordinate of the pixel
        y (int): y-coordinate of the pixel
        MLIC (np.ndarray): observed light intensities
        L_poses (np.ndarray): observed light positions
        regular_grids (List[np.ndarray], optional): interpolated light intensities. Defaults to [].
        methods (List[str], optional): interpolation methods. Defaults to [].
    """
    
    # Check if the number of regular grids and methods is the same
    assert len(regular_grids) == len(methods), "Number of regular grids and methods must be the same"
    
    # Create the plot
    fig, ax = plt.subplots(1, 1+len(regular_grids), figsize=(5+5*len(regular_grids), 5))
    
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    
    # Plot the observed light intensities
    ax[0].set_ylim(-1, 1)
    ax[0].set_xlim(-1, 1)
    scatter = ax[0].scatter(L_poses[:, 0], L_poses[:, 1], c=MLIC[:, y, x], cmap='viridis', s=2, vmax=255, vmin=0,)
    ax[0].set_title(f"f(x={x}, y={y}, ...)")
    ax[0].set_xlabel("U")
    ax[0].set_ylabel("V")

    # Plot the regular grids (interpolated light intensities)
    for i, regular_grid in enumerate(regular_grids):
        image = ax[i+1].matshow(regular_grid[y, x], cmap='viridis', vmax=255, vmin=0,)
        ax[i+1].set_title(f"Interpolation {methods[i]}")
        ax[i+1].axis('off')
        ax[i+1].invert_yaxis()

    # Add a colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    plt.colorbar(mappable=scatter, cax=cbar_ax)
    
    # Convert the plot to an image
    img = get_img_from_fig(fig)
    plt.close(fig)
    
    # Resize the image and show it
    img=cv.resize(img, (400+400*len(regular_grids), 400))
    cv.imshow("plot", img)

def interpolation_visualizer(coin:int, methods:List[str], coin_dim_input:Tuple[int, int], regular_grid_dim_input:Tuple[int, int]):
    """
    Visualize the interpolation of the light intensities for a given coin.

    Args:
        coin (int): coin number.
        methods (List[str]): list of interpolation methods.
        coin_dim_input (Tuple[int, int]): coin dimensions.
        regular_grid_dim_input (Tuple[int, int]): regular grid dimensions.

    Raises:
        FileNotFoundError: if the file with the interpolated data is not found.
    """
    filename = f"coin{coin}"
    MLIC, L_poses, U_hat, V_hat = load_light_results(filename)
    
    # Load the regular grids
    filename = filename+f"_{coin_dim_input}_{regular_grid_dim_input}"
    regular_grids_list = []
    coin_dims = []
    
    for m in tqdm(methods, desc="Loading regular grids"):
        
        if not os.path.exists(f"./results/{m}/{filename}.npz"):
            raise FileNotFoundError(f"File './results/{m}/{filename}.npz' not found: did you run the interpolation using {m} with coin{coin}, coin dimensions {coin_dim_input} and regular grid dimensions {regular_grid_dim_input}?")
        
        loaded = np.load(f"./results/{m}/{filename}.npz", allow_pickle=True)
        regular_grids = loaded["regular_grids"]
        regular_grid_dim = loaded["regular_grid_dim"]
        coin_dim = loaded["coin_dim"]
        
        regular_grids_list.append(regular_grids)
        coin_dims.append(coin_dim)
    
    assert np.all([coin_dim == coin_dims[0] for coin_dim in coin_dims]), \
        "All interpolated coins must have the same dimensions"
    
    # Resize the images to the desired dimensions
    MLIC_resized = np.array([cv.resize(coin, coin_dim) for coin in MLIC])
    
    # Compute the nearest light position to the center of the image
    distances = np.linalg.norm(L_poses[:,:2] - np.zeros( (1,1) ) , axis=1)
    
    # Prepare the pixel selector: the image to display the pixel position on
    pixel_selector = np.concatenate([  np.expand_dims(MLIC_resized[np.argmin(distances),:,:], axis=2),
                                np.expand_dims(U_hat, axis=2),
                                np.expand_dims(V_hat, axis=2) ], axis=2)
    pixel_selector = cv.cvtColor(pixel_selector.astype(np.uint8), cv.COLOR_YUV2BGR)
    
    # Create a window to select the pixel position
    cv.namedWindow("Pixel selector")
    callback_with_extra = partial(mouse_callback, pixel_selector=pixel_selector)
    cv.setMouseCallback("Pixel selector", callback_with_extra)
    
    plot(x_inp, y_inp, pixel_selector.copy())
    
    # Display the pixel interpolations
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        plot_pixel(x_inp, y_inp, MLIC_resized, L_poses, regular_grids_list, methods)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolation visualizer")
    parser.add_argument("--coin", type=int, required=True, help="Coin number: 1, 2, 3, 4")
    parser.add_argument("--methods", help="Method of interpolation to visualize", nargs='+', 
                        type=str, required=True, choices=["RBF", "PTM", "RBF_cuda"])
    
    parser.add_argument("--coin-dim", type=int, nargs=2, required=True, help="Coin dimensions")
    parser.add_argument("--regular-grid-dim", type=int, nargs=2, required=True, help="Regular grid dimensions")
    args=parser.parse_args()
    
    coin = args.coin
    methods = args.methods
    coin_dim_input = tuple(args.coin_dim)
    regular_grid_dim_input = tuple(args.regular_grid_dim)
    
    interpolation_visualizer(coin, methods, coin_dim_input, regular_grid_dim_input)