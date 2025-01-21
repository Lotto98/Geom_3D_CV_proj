import os
import numpy as np
from compute_light import load_light_results, find_nearest_point, plot_light_source
import itertools
import cv2 as cv
import argparse

x_inp = 50
y_inp = 50
drawing = False

def plot(x_inp, y_inp):
    img = plot_light_source(x_inp, y_inp, convert=False)
    cv.imshow("Light selector", img)
    
def set_light(x, y):
    
    global x_inp, y_inp
    
    plot(x, y)
    
    y = 200 - y - 1
    
    x_inp = ( x / 200)*100
    y_inp = ( y / 200)*100

# mouse callback function
def mouse_callback(event,x,y,flags,param):
    
    global drawing
        
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
    
    coin_path = f"./results/{method}/coin{coin_number}_{coin_dim_input}_{regular_grid_dim_input}.npz"
    
    if not os.path.exists(coin_path):
        raise FileNotFoundError(f"File '{coin_path}' not found: did you run the interpolation using {method} with coin{coin_number}, coin dimensions {coin_dim_input} and regular grid dimensions {regular_grid_dim_input}?")
    
    loaded = np.load(coin_path, allow_pickle=True)
    regular_grids = loaded["regular_grids"]
    regular_grid_dim = loaded["regular_grid_dim"]
    coin_dim = loaded["coin_dim"]
    
    print("Regular grid loaded")
    
    assert coin_dim[0] == coin_dim_input[0] and coin_dim[1] == coin_dim_input[1], f"Coin dimensions do not match: {coin_dim} != {coin_dim_input}"
    assert regular_grid_dim[0] == regular_grid_dim_input[0] and regular_grid_dim[1] == regular_grid_dim_input[1], f"Regular grid dimensions do not match: {regular_grid_dim} != {regular_grid_dim_input}"
    
    # Load data
    MLIC, L_poses, U_hat, V_hat = load_light_results(f"coin{coin_number}")
    U_hat = cv.resize(U_hat, coin_dim).astype(np.uint8)
    V_hat = cv.resize(V_hat, coin_dim).astype(np.uint8)
    
    directions_grid = np.array([np.array([y,x]) for x,y in itertools.product(range(0, regular_grid_dim[0]), range(0, regular_grid_dim[0]))])
    
    img = plot_light_source(100, 100, convert=False)
    cv.imshow("Light selector", img)
    cv.setMouseCallback("Light selector", mouse_callback)
    
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        nearest_point = find_nearest_point(directions_grid, np.array([y_inp, x_inp]))
        y_grid, x_grid, = nearest_point
        
        coin = regular_grids[:,:,y_grid,x_grid]
        coin = np.concatenate((np.expand_dims(coin, axis=2), 
                                np.expand_dims(U_hat, axis=2), 
                                np.expand_dims(V_hat, axis=2)), axis=2)
        
        coin = cv.cvtColor(coin, cv.COLOR_YUV2BGR)
        
        cv.imshow("Coin", cv.resize(coin, (512,512)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relighting")
    parser.add_argument("--coin", type=int, required=True, help="Coin number: 1, 2, 3, 4")
    parser.add_argument("--method", type=str, required=True, 
                        help="Method of interpolation to visualize",
                        choices=["RBF", "PTM", "RBF_cuda"])
    parser.add_argument("--coin-dim", type=int, nargs=2, required=True, help="Coin dimensions")
    parser.add_argument("--regular-grid-dim", type=int, nargs=2, required=True, help="Regular grid dimensions")
    
    args = parser.parse_args()
    
    coin_dim = tuple(args.coin_dim)
    regular_grid_dim = tuple(args.regular_grid_dim)
    
    relighting(args.coin, args.method, coin_dim, regular_grid_dim)