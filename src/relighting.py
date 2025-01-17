
import numpy as np

from compute_light import load_light_results, find_nearest_point, plot_light_source

import itertools

import cv2 as cv

import argparse

x_inp = 0
y_inp = 0
drawing = False

def plot(x_inp, y_inp):
    img = plot_light_source(x_inp, y_inp, convert=False)
    cv.imshow("Light selector", img)

# mouse callback function
def mouse_callback(event,x,y,flags,param):
    
    global drawing, x_inp, y_inp
    
    #if (x - 100)**2 + (y - 100)**2 <= 100**2:
    
    y = 200 - y -1
        
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        plot(x, y)
        x_inp = x
        y_inp = y
        #print(x_inp, y_inp)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            plot(x, y)
            x_inp = x
            y_inp = y
            #print(x_inp, y_inp)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        plot(x, y)
        x_inp = x
        y_inp = y
        #print(x_inp, y_inp)

def relighting(coin_number:int, method:str):
    
    loaded = np.load(f"./results/{method}/coin{coin_number}.npz", allow_pickle=True)
    regular_grids = loaded["regular_grids"]
    regular_grid_dim = loaded["regular_grid_dim"]
    coin_dim = loaded["coin_dim"]
    
    print("Regular grid loaded")
    
    # Load data
    MLIC, L_poses, U_hat, V_hat = load_light_results("coin1")
    U_hat = cv.resize(U_hat, coin_dim).astype(np.uint8)
    V_hat = cv.resize(V_hat, coin_dim).astype(np.uint8)
    
    directions_grid = np.array([np.array([x,y]) for x,y in itertools.product(range(0, regular_grid_dim[0]), range(0, regular_grid_dim[0]))])
    
    img = plot_light_source(x_inp, y_inp)
    cv.imshow("Light selector", img)
    cv.setMouseCallback("Light selector", mouse_callback)
    
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        coin = np.zeros((coin_dim[0], coin_dim[1], 1), dtype=np.uint8)
        
        nearest_point = find_nearest_point(directions_grid, np.array([x_inp, y_inp]))
                
        x_grid, y_grid = nearest_point
        
        coin = regular_grids[:,:,y_grid,x_grid]#.astype(np.uint8)
        
        #print(coin.max())
        
        coin = np.concatenate((np.expand_dims(coin, axis=2), 
                                np.expand_dims(U_hat, axis=2), 
                                np.expand_dims(V_hat, axis=2)), axis=2)
        
        coin = cv.cvtColor(coin, cv.COLOR_YUV2RGB)
        
        cv.imshow("Coin", cv.resize(coin, (512,512)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relighting")
    parser.add_argument("--coin", type=int, required=True, help="Coin number: 1, 2, 3, 4")
    parser.add_argument("--method", type=str, required=True, 
                        help="Method of interpolation to visualize",
                        choices=["RBF", "PTM"])
    args = parser.parse_args()
    relighting(args.coin, args.method)