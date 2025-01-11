import numpy as np
import cv2 as cv
import os
from typing import Tuple, List

from tqdm import tqdm

def get_frames_from_video(video_path:str, num_frames_to_select:int=20) -> List[np.ndarray]:
    
    if not os.path.exists(video_path):
        
        raise "Video file does not exist."
    
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
        cv.resizeWindow("original corners", 600, 400) 

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
                cv.waitKey(0)

    cv.destroyAllWindows()
    
    return objpoints, imgpoints

def calibrate_camera(objpoints: List[np.ndarray],
                    imgpoints: List[np.ndarray], 
                    imagesize: Tuple[int,int], name:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, imagesize, None, None)
    
    if not ret:
        print("Calibration failed.")
        return
    
    #save the camera parameters to a file
    os.makedirs('calibration', exist_ok=True)
    np.savez(f'calibration/param_{name}.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    
    return mtx, dist, rvecs, tvecs

def calculate_error(frames:List[np.ndarray], objpoints:List[np.ndarray], imgpoints: List[np.ndarray], 
                    mtx:np.ndarray, dist: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray, to_show:bool):
    
    if to_show:
        # Naming a window 
        cv.namedWindow("projected corners test", cv.WINDOW_NORMAL) 
        cv.resizeWindow("projected corners test", 600, 400) 
    
    mean_error = 0
    for i, img in zip(range(len(objpoints)), frames):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
        print(f"error for frame {i}: ",error)
        
        if to_show:
            cv.drawChessboardCorners(img, (9,6), imgpoints2, True)
            cv.imshow('projected corners test', img)
            cv.waitKey(0)
        
        mean_error += error
    
    print("Mean error: ",mean_error/len(objpoints))
    cv.destroyAllWindows()
    
def load_camera_parameters(name:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    if not os.path.exists(f'calibration/param_{name}.npz'):
        print("Camera parameters file does not exist.")
        return
    
    data = np.load(f'calibration/param_{name}.npz')
    
    mtx = data['mtx']
    dist = data['dist']
    rvecs = data['rvecs']
    tvecs = data['tvecs']
    
    return mtx, dist, rvecs, tvecs

def main(name:str, chessboard_dim:Tuple[int,int], num_frames:int, to_show:bool, to_calculate_error:bool):
    
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
        calculate_error(frames, objpoints, imgpoints, mtx, dist, rvecs, tvecs, to_show=to_show)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Calibrate camera using chessboard images.')
    parser.add_argument('--name', type=str, default='moving_light', help='Name of the camera setup (moving_light/static)')
    parser.add_argument('--chessboard_dim', type=tuple, default=(9,6), help='Chessboard dimensions')
    parser.add_argument('--num_frames', type=int, default=20, help='Number of frames to select from the video')
    
    parser.add_argument('--debug', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Show the chessboard corners and projected corners')
    parser.add_argument('--error', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Calculate the error')
    args = parser.parse_args()
    
    name = args.name
    chessboard_dim = args.chessboard_dim
    num_frames = args.num_frames
    
    to_show = args.debug
    to_calculate_error = args.error
    
    # Print a summary of the parameters
    print(f"Name of the calibration video: {name}")
    print(f"Chessboard dimensions: {chessboard_dim}")
    print(f"Number of frames: {num_frames}")
    print(f"Debug: {to_show}")
    print(f"Calculate error: {to_calculate_error}")
    
    # ask the user if they want to continue
    response = input("Do you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Exiting the program.")
        exit()
    
    main(name, chessboard_dim, num_frames, to_show, to_calculate_error)
