
import numpy as np

from analysis import load_results, find_nearest_point, plot_light_source

import itertools

import cv2 as cv

x_inp = 0
y_inp = 0
drawing = False

def plot(x_inp, y_inp):
    img = plot_light_source(x_inp, y_inp, convert=False)
    cv.imshow("Light selector", img)

# mouse callback function
def mouse_callback(event,x,y,flags,param):
    
    global drawing, x_inp, y_inp
    
    y = 200 - y -1
    
    #if (x - 100)**2 + (y - 100)**2 <= 100**2:
        
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        plot(x, y)
        x_inp = x
        y_inp = y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            plot(x, y)
            x_inp = x
            y_inp = y
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        plot(x, y)
        x_inp = x
        y_inp = y


def relighting():
    regular_grids = np.load("./results/RBF/coin1.npz", allow_pickle=True)
    regular_grids = regular_grids["arr_0"].item()
    regular_grid_dim = (100, 100)
    
    # Load data
    MLIC, L_poses, U_hat, V_hat = load_results("coin1")
    
    directions_grid = np.array([np.array([x,y]) for x,y in itertools.product(range(0, regular_grid_dim[0]), range(0, regular_grid_dim[0]))])
    
    img = plot_light_source(x_inp, y_inp)
    cv.imshow("Light selector", img)
    cv.setMouseCallback("Light selector", mouse_callback)
    
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
        coin = np.zeros((256, 256, 1), dtype=np.uint8)
        
        nearest_point = find_nearest_point(directions_grid, np.array([x_inp, y_inp]))
                
        x_grid, y_grid = nearest_point
        
        for x in range(256):
            for y in range(256):
                
                regular_grid = regular_grids[(x, y)]
                
                coin[y,x] = regular_grid[x_grid, y_grid]
        
        cv.imshow("Coin", cv.flip(cv.resize(coin, (512,512)),1))

if __name__ == "__main__":
    relighting()