import numpy as np
import cv2 as cv
import os
from typing import Tuple, List

from tqdm import tqdm

def get_frames_from_video(video_path:str, num_frames_to_select:int=20) -> List[np.ndarray]:
    """
    Get num_frames_to_select evenly spaced frames from the video.

    Args:
        video_path (str): Path to the video file.
        num_frames_to_select (int, optional): number of frames to select. Defaults to 20.

    Raises:
        FileNotFoundError: Raised if the video file does not exist.

    Returns:
        List[np.ndarray]: List of selected frames.
    """
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' does not exist.")
    
    cap = cv.VideoCapture(video_path)

    # Get total frame count
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {frame_count}")

    # Select n evenly spaced frames
    frame_indices = np.linspace(0, frame_count-1, num_frames_to_select, dtype=int)

    selected_frames = []

    for idx in frame_indices:
        # Set the current frame position
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            selected_frames.append(frame)
        
    # Release resources
    cap.release()
    
    return selected_frames, selected_frames[0].shape[1:]

def get_points(selected_frames:List[np.ndarray], chessboard_dim:Tuple[int,int], to_show:bool) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    OpenCV script to get the object points and image points from the selected frames.
    
    https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

    Args:
        selected_frames (List[np.ndarray]): List of selected frames.
        chessboard_dim (Tuple[int,int]): The dimensions of the chessboard.
        to_show (bool): show the chessboard corners.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Tuple of object points and image points.
    """
    
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_dim[0]*chessboard_dim[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_dim[0],0:chessboard_dim[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    if to_show:
        # Naming a window 
        cv.namedWindow("original corners", cv.WINDOW_NORMAL) 
        cv.resizeWindow("original corners", 700, 500) 

    for i, frame in tqdm(enumerate(selected_frames), desc="Processing frames", total=len(selected_frames)):
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_dim, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            if to_show:
                # Draw and display the corners
                cv.drawChessboardCorners(frame, chessboard_dim, corners, ret)
                cv.imshow('original corners', frame)
                
                while True:
                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q'):
                        exit()
                    elif key != 255:
                        break
                    if cv.getWindowProperty("original corners", cv.WND_PROP_VISIBLE) < 1:
                        exit()
    cv.destroyAllWindows()
    
    return objpoints, imgpoints

def calibrate_camera(objpoints: List[np.ndarray],
                    imgpoints: List[np.ndarray], 
                    imagesize: Tuple[int,int], name:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imagesize, None, None)
    
    if not ret:
        print("Calibration failed.")
        return
    
    #save the camera parameters to a file
    os.makedirs('calibration', exist_ok=True)
    np.savez(f'calibration/param_{name}.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    
    return mtx, dist, rvecs, tvecs

def calculate_error(frames:List[np.ndarray], 
                    objpoints:List[np.ndarray], 
                    imgpoints: List[np.ndarray], 
                    mtx:np.ndarray, 
                    dist: np.ndarray, 
                    rvecs: np.ndarray, 
                    tvecs: np.ndarray, 
                    to_show:bool,
                    chessboard_dim:tuple) -> None:
    """
    Calculate the error of the calibration.
    
    Based on the OpenCV tutorial:
    https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

    Args:
        frames (List[np.ndarray]): list of selected frames.
        objpoints (List[np.ndarray]): list of object points, one for each frame.
        imgpoints (List[np.ndarray]): list of image points, one for each frame.
        mtx (np.ndarray): the intrinsic camera matrix.
        dist (np.ndarray): the distortion coefficients.
        rvecs (np.ndarray): the rotation vectors.
        tvecs (np.ndarray): the translation vectors.
        to_show (bool): show the projected corners.
        chessboard_dim (tuple): the dimensions of the chessboard.
    """
    
    if to_show:
        # Naming a window 
        cv.namedWindow("projected corners test", cv.WINDOW_NORMAL) 
        cv.resizeWindow("projected corners test", 700, 500) 
    
    mean_error = 0
    for i, img in zip(range(len(objpoints)), frames):
        
        # project the corners for each frame to check the error
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
        
        print(f"error for frame {i}: ",error)
        
        if to_show:
            #show the projected corners
            cv.drawChessboardCorners(img, chessboard_dim, imgpoints2, True)
            cv.imshow('projected corners test', img)
            
            while True:
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    exit()
                elif key != 255:
                    break
                if cv.getWindowProperty("projected corners test", cv.WND_PROP_VISIBLE) < 1:
                    exit()
        
        mean_error += error
    
    print("Mean error: ",mean_error/len(objpoints))
    cv.destroyAllWindows()
    
def load_camera_parameters(name:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the camera parameters from the file.

    Args:
        name (str): Name of the camera to load the parameters for.

    Raises:
        FileNotFoundError: Raised if the file does not exist.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        Tuple containing the intrinsic camera matrix, distortion coefficients, rotation vectors and translation
    """
    
    if not os.path.exists(f'calibration/param_{name}.npz'):
        raise FileNotFoundError(f"File calibration/param_{name}.npz not found: did you run the calibration for {name} ?")
    
    data = np.load(f'calibration/param_{name}.npz')
    
    mtx = data['mtx']
    dist = data['dist']
    rvecs = data['rvecs']
    tvecs = data['tvecs']
    
    print("\nCamera parameters loaded.\n")
    print("Intrinsic camera matrix: ")
    print(mtx,"\n")
    print("Distortion coefficients: ")
    print(dist,"\n\n")
    
    return mtx, dist, rvecs, tvecs

def main_calibration(name:str, chessboard_dim:Tuple[int,int], num_frames:int, to_show:bool, to_calculate_error:bool):
    """
    Main function to calibrate the camera.

    Args:
        name (str): Name of the camera setup to calibrate.
        chessboard_dim (Tuple[int,int]): The dimensions of the chessboard.
        num_frames (int): Number of frames to select from the video.
        to_show (bool): Show the chessboard corners and projected corners.
        to_calculate_error (bool): Calculate the error of the calibration.
    """
    
    if name == "moving_light":
        video_path = './data/cam2 - moving light/calibration.mp4'
    elif name == "static":
        video_path = './data/cam1 - static/calibration.mov'
    else:
        raise "Invalid camera setup name."

    frames, image_size = get_frames_from_video(video_path, num_frames_to_select=num_frames)

    objpoints, imgpoints = get_points(selected_frames=frames, chessboard_dim=chessboard_dim, to_show=to_show)

    mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, image_size, name=name)

    if to_calculate_error:
        calculate_error(frames, objpoints, imgpoints, mtx, dist, rvecs, tvecs, to_show=to_show, chessboard_dim=chessboard_dim)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Calibrate camera using chessboard images.')
    parser.add_argument('--name', type=str, default='moving_light', help='Name of the camera setup (moving_light/static)')
    parser.add_argument('--chessboard_dim', 
                        type=int, 
                        nargs=2, 
                        default=[9, 6], 
                        metavar=("COLS", "ROWS"),  
                        help='Chessboard dimensions. Default: (9, 6)')
    parser.add_argument('--num_frames', type=int, default=20, help='Number of frames to select from the video. Default: 20')
    
    parser.add_argument('--debug', type=bool, default=False, action=argparse.BooleanOptionalAction, 
                        help='Show the chessboard corners and projected corners')
    parser.add_argument('--error', type=bool, default=False, action=argparse.BooleanOptionalAction, 
                        help='Calculate the error of the calibration')
    args = parser.parse_args()
    
    name = args.name
    chessboard_dim = args.chessboard_dim
    num_frames = args.num_frames
    
    to_show = args.debug
    to_calculate_error = args.error
    
    # Print a summary of the parameters
    print(args)
    
    main_calibration(name, chessboard_dim, num_frames, to_show, to_calculate_error)
