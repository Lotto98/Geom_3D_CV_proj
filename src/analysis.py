from compute_light import compute_light
import os
import argparse
import gc

if __name__ == "__main__":
    
    
    description = """
    This script is used to compute the light for a given coin and interpolate it using different methods.
    """
    
    
    parser = argparse.ArgumentParser(description="Analysis script")
    
    parser.add_argument(
        "--coin", 
        type=int, 
        choices=[1, 2, 3, 4], 
        required=True, 
        help="Specify the coin type to process (1, 2, 3, or 4)."
    )
    
    parser.add_argument(
        "--compute-light", 
        action="store_true", 
        help="Enable light computation."
    )
    
    parser.add_argument(
        "--interpolate", 
        action="store_true", 
        help="Enable interpolation for computed light."
    )

    # Compute light arguments
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode for general processing. For `--compute-light` only."
    )
    parser.add_argument(
        "--debug-moving", 
        action="store_true", 
        help="Enable debug mode for moving light. For `--compute-light` only."
    )
    parser.add_argument(
        "--debug-static", 
        action="store_true", 
        help="Enable debug mode for static light. For `--compute-light` only."
    )

    # Interpolation arguments
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["RBF", "PTM", "RBF_cuda"],  
        help="Interpolation method. For `--interpolate` only."
    )
    parser.add_argument(
        "--nprocesses", 
        type=int, 
        default=-1, 
        help="Number of processes for interpolation (-1 to use all available). For `--method RBF` only."
    )

    parser.add_argument(
        "--coin-dim", 
        type=int, 
        nargs=2, 
        default=[512, 512], 
        metavar=("WIDTH", "HEIGHT"), 
        help="Dimensions of the coin (default: [512, 512]). For `--interpolate` only."
    )
    parser.add_argument(
        "--regular-grid-dim", 
        type=int, 
        nargs=2, 
        default=[100, 100], 
        metavar=("ROWS", "COLS"), 
        help="Dimensions of the regular grid (default: [100, 100]). For `--interpolate` only."
    )
    
    args = parser.parse_args()
    
    if not args.interpolate and not args.compute_light:
        print("Choose at least one action: --compute-light or --interpolate")
        exit()
        
    if args.interpolate and args.method is None:
        print("Choose a method for interpolation: --method RBF or PTM or RBF_cuda")
        exit()
        
    if args.interpolate and args.nprocesses < -1:
        print("Number of processes must be greater than -1")
        exit()
    
    if args.compute_light and not args.interpolate and (args.method is not None or args.nprocesses != -1):
        print("WARNING: Interpolation method and number of processes will be ignored for light computation!")
    
    if args.interpolate and not args.compute_light and (args.debug or args.debug_moving or args.debug_static):
        print("WARNING: Debug flags are ignored for interpolation without light computation!")
    
    print(args)
    
    if args.compute_light:
        print(f"\nComputing light for coin {args.coin} with debug={args.debug}, debug_moving={args.debug_moving}, debug_static={args.debug_static}")
        compute_light(args.coin, args.debug, args.debug_moving, args.debug_static)
        gc.collect()
    
    if args.interpolate:
        string_to_print = f"\nInterpolating light for coin {args.coin} with method={args.method}"
        if args.method=="RBF":
            string_to_print += f", nprocesses={args.nprocesses}"
        print(string_to_print)
        
        # Set environment variables for scipy RBF method with multiple processes 
        if args.method == "RBF" and (args.nprocesses == -1 or args.nprocesses > 1):
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
        
        from interpolation import interpolation
        interpolation(args.coin, tuple(args.coin_dim), tuple(args.regular_grid_dim), args.method, args.nprocesses)