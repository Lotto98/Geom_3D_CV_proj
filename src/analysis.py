from compute_light import compute_light
import subprocess

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", type=int, required=True, help="Coin number: 1, 2, 3, 4")
    parser.add_argument("--compute-light", action="store_true", help="Compute light")
    parser.add_argument("--interpolate", action="store_true", help="Interpolate computed light")
    
    #compute light arguments
    parser.add_argument("--debug", action="store_true", help="Debug mode", default=False)
    parser.add_argument("--debug_moving", action="store_true", help="Debug mode moving light", default=False)
    parser.add_argument("--debug_static", action="store_true", help="Debug mode static light", default=False)
    
    #interpolation arguments
    parser.add_argument("--method", type=str)
    parser.add_argument("--nprocesses", type=int, default=-1)
    
    parser.add_argument("--coin_dim", type=int, nargs='+', default=[256, 256])
    parser.add_argument("--regular_grid_dim", type=int, nargs='+', default=[100, 100])
    
    args = parser.parse_args()
    
    if not args.interpolate and not args.compute_light:
        print("Choose at least one action: --compute-light or --interpolate")
        exit()
        
    if args.interpolate and args.method is None:
        print("Choose a method for interpolation: --method RBF or PTM")
        exit()
    
    print(args)
    
    if args.compute_light:
        compute_light(coin_number=args.coin, 
                        debug=args.debug, 
                        debug_moving=args.debug_moving, debug_static=args.debug_static)
    
    if args.interpolate:
        filename = f"coin{args.coin}"
        regular_grid_dim = tuple(args.regular_grid_dim)
        resize_dim = tuple(args.coin_dim)
        nprocesses = args.nprocesses
        method = args.method
        
        execution_string = f"python3 src/interpolation.py"
        execution_string += f" --filename {filename}"
        execution_string += f" --coin_dim {resize_dim[0]} {resize_dim[1]}"
        execution_string += f" --regular_grid_dim {regular_grid_dim[0]} {regular_grid_dim[1]}"
        execution_string += f" --method {method}"
        execution_string += f" --nprocesses {nprocesses}"
        
        print(execution_string)
        if nprocesses == -1 or nprocesses > 1:
            execution_string = f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; {execution_string}"
        subprocess.call(execution_string, shell=True)