import os
import numpy as np
from compute_light import load_light_results, find_nearest_point, plot_light_source
import itertools
import cv2 as cv
import argparse


# Global variables for the light source selection
drawing:bool = False
image_size:int = 200

# Note: x_inp and y_inp are in grid space
x_inp:int = 50
y_inp:int = 50
regular_grid_dim_x:int = 100
regular_grid_dim_y:int = 100

def visualize_light_source(x:int, y:int):
    """
    Plot the light source on the screen. The light source is represented by a white circle on a black background.

    Args:
        x (int): x coordinate in input in [0, image_size].
        y (int): y coordinate in input in [0, image_size].
    """
    
    global image_size
    
    img = plot_light_source(x, y, convert=False, image_size=image_size)
    cv.imshow("Light selector", img)
    
def set_light(x:int, y:int):
    """
    Set the given light source position and plot it on the screen.

    It also updates the global variables x_inp and y_inp in the range [0, grid_dim] (grid space).

    Args:
        x (int): input x coordinate in [0, image_size].
        y (int): input y coordinate in [0, image_size].
    """
    
    global x_inp, y_inp, image_size, regular_grid_dim_x, regular_grid_dim_y
    
    # Plot the light source before conversion
    visualize_light_source(x, y)
    
    # Move origin to bottom left corner
    y = image_size - y - 1
    
    # Convert in grid space
    x_inp = ( x / image_size)*regular_grid_dim_x
    y_inp = ( y / image_size)*regular_grid_dim_y

# mouse callback function
def mouse_callback(event:int,x:int,y:int,flags,param):
    """
    Callback function to handle mouse events. 
    It sets the light source position when the left mouse button is clicked and dragged.

    Args:
        event (int): event type.
        x (int): x coordinate.
        y (int): y coordinate.
    """
    global drawing
    
    # Implement dragging
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        set_light(x, y)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            set_light(x, y)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        set_light(x, y)

def relighting(coin_number:int, method:str, coin_dim_input:tuple, regular_grid_dim_input:tuple):
    """
    Relight a coin using the interpolated light data from the specified method.

    Args:
        coin_number (int): coin number.
        method (str): method of interpolation to load data for.
        coin_dim_input (tuple): coin dimensions.
        regular_grid_dim_input (tuple): regular grid dimensions.

    Raises:
        FileNotFoundError: if the file with the interpolated data is not found.
    """
    
    # Check existence of interpolated data
    coin_path = f"./results/{method}/coin{coin_number}_{coin_dim_input}_{regular_grid_dim_input}.npz"
    if not os.path.exists(coin_path):
        raise FileNotFoundError(f"File '{coin_path}' not found: did you run the interpolation using {method} with coin{coin_number}, coin dimensions {coin_dim_input} and regular grid dimensions {regular_grid_dim_input}?")
    
    # Load the interpolated data: regular grids and dimensions
    loaded = np.load(coin_path, allow_pickle=True)
    regular_grids = loaded["regular_grids"]
    regular_grid_dim = loaded["regular_grid_dim"]
    coin_dim = loaded["coin_dim"]
    print("Regular grid loaded")
    
    # Check if the dimensions match the loaded data: coin_dim and regular_grid_dim should match the input arguments.
    assert coin_dim[0] == coin_dim_input[0] and coin_dim[1] == coin_dim_input[1], f"Coin dimensions do not match: {coin_dim} != {coin_dim_input}"
    assert regular_grid_dim[0] == regular_grid_dim_input[0] and regular_grid_dim[1] == regular_grid_dim_input[1], f"Regular grid dimensions do not match: {regular_grid_dim} != {regular_grid_dim_input}"
    
    #set global variables, so that they can be used in the mouse callback:
    # x_inp, y_inp : light source position in grid space
    # regular_grid_dim_x, regular_grid_dim_y : regular grid dimensions
    global x_inp, y_inp, regular_grid_dim_x, regular_grid_dim_y
    
    # Set the initial light source position to the center of the unit circle
    # The light source position is in grid space
    regular_grid_dim_x = regular_grid_dim[1]
    regular_grid_dim_y = regular_grid_dim[0]
    x_inp = (regular_grid_dim_x//2) 
    y_inp = (regular_grid_dim_y//2)
    
    # Load U_hat and V_hat and resize them to the coin dimensions
    _, _, U_hat, V_hat = load_light_results(f"coin{coin_number}")
    U_hat = cv.resize(U_hat, coin_dim).astype(np.uint8)
    V_hat = cv.resize(V_hat, coin_dim).astype(np.uint8)
    
    # Prepare the regular grid directions in grid space
    directions_grid = np.array([np.array([y,x]) for x,y in itertools.product(range(0, regular_grid_dim[0]), range(0, regular_grid_dim[0]))])
    
    # Create a window to select the light source position
    img = plot_light_source(100, 100, convert=False)
    cv.imshow("Light selector", img)
    cv.setMouseCallback("Light selector", mouse_callback)
    
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Find the nearest point in the regular grid: y_inp and x_inp may be fractional and 
        # need to be rounded to the nearest (interpolated) grid point
        nearest_point = find_nearest_point(directions_grid, np.array([y_inp, x_inp]))
        y_grid, x_grid, = nearest_point
        
        # Display the coin with the selected light source:
        
        #1) Get the regular grid at the selected point (Y channel)
        coin = regular_grids[:,:,y_grid,x_grid]
        
        #2) Concatenate the coin with the U_hat and V_hat channels
        coin = np.concatenate((np.expand_dims(coin, axis=2), 
                                np.expand_dims(U_hat, axis=2), 
                                np.expand_dims(V_hat, axis=2)), axis=2)
        
        #3) Convert to BGR for visualization and resize for better display
        coin = cv.cvtColor(coin, cv.COLOR_YUV2BGR)
        cv.imshow("Coin", cv.resize(coin, (512,512)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relighting of a coin using the data from a specified interpolation method")
    parser.add_argument("--coin", type=int, required=True, help="Coin number: 1, 2, 3, 4", choices=[1, 2, 3, 4])
    parser.add_argument("--method", type=str, required=True, 
                        help="Method of interpolation to visualize",
                        choices=["RBF", "PTM", "RBF_cuda"])
    parser.add_argument("--coin-dim", type=int, nargs=2, required=True, help="Coin dimensions")
    parser.add_argument("--regular-grid-dim", type=int, nargs=2, required=True, help="Regular grid dimensions")
    
    args = parser.parse_args()
    
    coin_dim = tuple(args.coin_dim)
    regular_grid_dim = tuple(args.regular_grid_dim)
    
    relighting(args.coin, args.method, coin_dim, regular_grid_dim)