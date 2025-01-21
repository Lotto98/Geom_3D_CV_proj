import os
from typing import Tuple
import numpy as np
import cv2 as cv
import moviepy as mp
import multiprocessing
from calibration import load_camera_parameters
import librosa
from tqdm import tqdm

# Paths to the input videos
static_path = "./data/cam1 - static/"
moving_path = "./data/cam2 - moving light/"

# Paths to the output videos
aligned_static_path = "./data_aligned/cam1 - static/"
aligned_moving_path = "./data_aligned/cam2 - moving light/"

def align_videos(video1_path:str, video2_path:str, audio_sr:int=10000)->None:
    """
    Align two videos based on their audio tracks:
    1) Extract audio tracks.
    2) Find time offset and crop videos accordingly.
    3) Set fps to 30 for both videos.
    4) Set duration to the shorter video.
    5) Write the aligned videos to disk.

    Args:
        video1_path (str): video path for the static camera.
        video2_path (str): video path for the moving light camera.
        audio_sr (int, optional): sample rate for the audio tracks to use. Defaults to 10000.
    """

    def extract_audio_from_video(video_path:str)->Tuple[np.ndarray, int]:
        """
        Extract and load audio from a video file.

        Args:
            video_path: (str): path to the video file.

        Returns:
            Tuple[np.ndarray, int]: audio signal mono channel and sample rate.
        """
        # Load the video file
        video = mp.VideoFileClip(video_path)
        audio = video.audio.to_soundarray()

        # Convert to mono if the audio is stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Average the channels to get mono

        return audio, video.audio.fps


    def find_offset(audio1:np.ndarray, audio2:np.ndarray, sr:int, sec:int=15)->float:
        """
        Find the time offset between two audio tracks using cross-correlation:
        1) Limit the audio signals to the first sec seconds.
        2) Compute cross-correlation.
        3) Find the index of the maximum correlation.
        4) Convert delay to time offset in seconds.

        Args:
            audio1: (ndarray): first audio signal.
            audio2: (ndarray): second audio signal.
            sr: (int): sample rate of the audio signals.

        Returns:
            float: time offset in seconds. 
            If positive, audio2 should be shifted to align with audio1.
            If negative, audio1 should be shifted to align with audio2.
        """
        # Ensure audio1 and audio2 have the same length for comparison
        max_len = min(len(audio1), len(audio2), sec * sr)  # Limit to the first sec seconds
        audio1 = audio1[:max_len]
        audio2 = audio2[:max_len]

        # Compute cross-correlation
        correlation = np.correlate(audio1, audio2, mode="full")

        # Find the index of the maximum correlation
        delay = np.argmax(correlation) - (len(audio2) - 1)  # Adjust to center alignment

        # Convert delay to time offset in seconds
        time_offset = delay / sr

        return time_offset

    def crop_video(video_path:str, offset:float)->mp.VideoFileClip:
        """
        Crop video based on the offset.

        Args:
            video_path (str): path to the video file.
            offset (float): time offset in seconds.
        
        Returns:
            VideoFileClip: cropped video.
        """
        
        video = mp.VideoFileClip(video_path)
        video = video.subclipped(offset, video.duration)
        
        return video
    
    if not os.path.exists(video1_path):
        raise FileNotFoundError(f"File not found: {video1_path}")
    if not os.path.exists(video2_path):
        raise FileNotFoundError(f"File not found: {video2_path}")

    if os.path.exists(f"./data_aligned/cam1 - static/{video1_path.split('/')[-1]}") and os.path.exists(f"./data_aligned/cam2 - moving light/{video2_path.split('/')[-1]}"):
        print("The videos are already aligned")
        return
    
    #1) Extract audio tracks
    audio1, sr1 = extract_audio_from_video(video1_path)
    audio2, sr2 = extract_audio_from_video(video2_path)

    audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=audio_sr)
    audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=audio_sr)
    
    #2) Find time offset and crop videos accordingly
    offset = find_offset(audio1, audio2, audio_sr)
    print(f"Time offset: {offset:.2f} seconds")
    
    if offset > 0:
        video1 = crop_video(video1_path, offset)
        video2 = mp.VideoFileClip(video2_path)
    else:
        video1 = mp.VideoFileClip(video1_path)
        video2 = crop_video(video2_path, offset) 

    #3) Set fps to 30 for both videos
    video1 = video1.with_fps(30)
    video2 = video2.with_fps(30)
    
    #4) Set duration to the shorter video
    duration = min(video1.duration, video2.duration)
    video1 = video1.subclipped(0, duration)
    video2 = video2.subclipped(0, duration)
    
    #5) Write the aligned videos to disk
    os.makedirs(aligned_static_path, exist_ok=True)
    os.makedirs(aligned_moving_path, exist_ok=True)
    
    output1 = os.path.join(aligned_static_path, video1_path.split('/')[-1])
    output2 = os.path.join(aligned_moving_path, video2_path.split('/')[-1])
    
    video1.write_videofile(output1, codec="libx264", preset="ultrafast", threads=multiprocessing.cpu_count(), audio=True)
    video2.write_videofile(output2, codec="libx264", preset="ultrafast", threads=multiprocessing.cpu_count(), audio=True)

def detect_fiducial_marker(image:np.ndarray, threshold:int=100, debug:bool=False)->Tuple[np.ndarray, np.ndarray]:
    """
    Detect the fiducial marker in an image:
    1) Threshold the image.
    2) Find contours.
    3) Simplify the contours: Ramer-Douglas-Peucker algorithm.
    4) Validate the contours with a white dot (midpoint).
    5) Order the points clockwise.

    Args:
        image (np.ndarray): image to detect the marker.
        threshold (int, optional): initial threshold to use for thresholding. Defaults to 100.
        debug (bool, optional): show the binary image. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: detected marker and midpoint.
    """
    
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # Step 1: Thresholding
    
    #if auto_threshold:
    #    _, binary = cv.threshold(image_grey, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    #else:
    _, binary = cv.threshold(image_grey, threshold, 255, cv.THRESH_BINARY)

    # Step 2: Morphological Operations (to remove noise)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    
    if debug:
        cv.namedWindow("Binary", cv.WINDOW_NORMAL)
        cv.resizeWindow("Binary", 640, 480)
        cv.imshow("Binary", binary)

    # Step 3: Find contours (hierarchical)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        # should not happen
        return None, None 

    # Prepare list for detected markers
    detected_markers = {}

    for i, contour in enumerate(contours):

        # Step 4: Simplify contour using Ramer-Douglas-Peucker algorithm
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        detected_markers[i]=approx.reshape(-1, 2)
    
    # Step 6: Validate with white dot (midpoint check)
    valid_markers = []
    valid_midpoints = []
    for i, marker in detected_markers.items():
        
        if len(marker) != 4:
            continue
        
        if hierarchy[0][i][3] == -1: # No parent
            continue
        
        if cv.contourArea(marker) < 1000:
            continue
        
        valid = False
        for j in range(4):
            # Midpoints of corresponding vertices
            point_a = marker[j]
            point_b = find_nearest_point(detected_markers[hierarchy[0][i][3]], point_a)
            
            midpoint = (point_a + point_b) // 2
            
            if binary[midpoint[1], midpoint[0]] == 255:
                
                if valid: # More than one white dot
                    valid = False
                    valid_markers.pop()
                    valid_midpoints.pop()
                    break
                
                valid = True
                valid_markers.append(marker)
                valid_midpoints.append(midpoint)
    
    if len(valid_markers) == 0:
        if threshold > 0:
            return detect_fiducial_marker(image, threshold=threshold-10, debug=debug)
    
    
    marker = None
    midpoint = None
    
    if len(valid_markers) == 1:
        
        marker = valid_markers[0]
        midpoint = valid_midpoints[0]
        
    elif len(valid_markers) > 1:
        areas = [cv.contourArea(marker) for marker in valid_markers]
        idx = np.argmax(areas)
        
        if isinstance(idx, np.ndarray):
            idx = idx[0]
        
        marker = valid_markers[idx]
        midpoint = valid_midpoints[idx]
    
    return order_points_clockwise(marker, midpoint), midpoint

def find_nearest_point(points:np.ndarray, point:np.ndarray)->np.ndarray:
    """
    Find the nearest point to a given point.

    Args:
        points (np.ndarray): points to search from.
        point (np.ndarray): point to find the nearest point to.

    Returns:
        np.ndarray: Nearest point.
    """
    distances = np.linalg.norm(points - point, axis=1)
    return points[np.argmin(distances)]
    
def plot_fiducial_marker(image:np.ndarray, marker:np.ndarray, midpoint:np.ndarray)->np.ndarray:
    """
    Plot the fiducial marker on the image.

    Args:
        image (np.ndarray): image to plot the marker.
        marker (np.ndarray): marker to plot.
        midpoint (np.ndarray): midpoint of the marker.

    Returns:
        np.ndarray: Image with the plotted marker.
    """
    
    colors = [
        (0, 0, 225),   # Red
        (255, 0, 255), # Magenta
        (255, 0, 0),   # Blue
        (0, 102, 255) #Orange
    ]
    
    #plot the midpoint
    cv.circle(image, tuple(midpoint), 10, (0, 0, 255), -1)
    
    #plot the marker
    cv.polylines(image, [marker], isClosed=True, color=(0, 255, 0), thickness=6)
    for idx, (corner, color) in enumerate(zip(marker, colors)):
        
        cv.circle(image, tuple(corner), 7, color, -1)
        
        coordinate_text = tuple(corner)
        
        if idx==0:
            coordinate_text = (coordinate_text[0]+5, coordinate_text[1]-5)
        elif idx == 1:
            coordinate_text = (coordinate_text[0]+5, coordinate_text[1]+15)
        elif idx == 2:
            coordinate_text = (coordinate_text[0]-25, coordinate_text[1]+20)
        elif idx == 3:
            coordinate_text = (coordinate_text[0]-25, coordinate_text[1]-10)
        
        cv.putText(image, f"c{idx}", coordinate_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    return image

def order_points_clockwise(points:np.ndarray, midpoint:np.ndarray)->np.ndarray:
    """
    Sort the points in clockwise order starting from the nearest point to the midpoint.

    Args:
        points (np.ndarray): points to sort.
        midpoint (np.ndarray): point to start the sorting from.

    Returns:
        np.ndarray: sorted points.
    """
    
    points = np.array(points)  # Ensure points are a NumPy array

    # Calculate the centroid
    centroid = np.mean(points, axis=0)

    # Compute angles of points relative to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort points by angle for clockwise order
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    start_point = find_nearest_point(points, midpoint)
    
    start_index = np.where((sorted_points == start_point).all(axis=1))[0][0]
    
    # Reorder the points to start from the specified start point
    sorted_points = np.concatenate((sorted_points[start_index:], sorted_points[:start_index]))

    return sorted_points

def plot_light_source(u:float, v:float, convert:bool=True, image_size:int=200)->np.ndarray:
    """
    Plot a light source on a black background.

    Args:
        u (float): x-coordinate of the light source in normalized coordinates (-1 to 1) or pixel coordinates.
        v (float): y-coordinate of the light source in normalized coordinates (-1 to 1) or pixel coordinates.
        convert (bool, optional): Whether to convert normalized coordinates to pixel coordinates. Defaults to True.
        image_size (int, optional): Size of the output image. Defaults to 200.

    Returns:
        np.ndarray: Image with the plotted light source.
    """
    
    if convert:
        u = int(((u + 1) / 2) * (image_size-1))
        v = int(((v + 1) / 2) * (image_size-1))
        
        v = image_size - v - 1 
    
    image = np.zeros((image_size, image_size, 1), dtype=np.uint8)
    
    cv.circle(image, (image_size//2, image_size//2), image_size//2, 255, 2)
    
    cv.circle(image, (u, v), 2, 255, -1)
    cv.line(image, (image_size//2, image_size//2), (u, v), 255, 1)
    
    return image

def calculateLightPosition(objectPoints:np.ndarray, imagePoints:np.ndarray, 
                            cameraMatrix:np.ndarray, dist:np.ndarray, 
                            method="Homography")->np.ndarray:
    """
    Compute the light source position from the object and image points.

    Args:
        objectPoints (np.ndarray): "3D" object points: 4x3 array.
        imagePoints (np.ndarray): 2D image points: 4x2 array.
        cameraMatrix (np.ndarray): intrinsic camera matrix.
        dist (np.ndarray): distortion coefficients.
        method (str, optional): Method to compute the light source: "PnP" or "Homography". Defaults to "Homography".

    Raises:
        ValueError: if the solvePnP method fails.

    Returns:
        np.ndarray: Light source position.
    """
    
    if method == "PnP":
        
        retval, rvec, t = cv.solvePnP(objectPoints, imagePoints, cameraMatrix, dist, flags=cv.SOLVEPNP_P3P )
        
        if not retval:
            raise ValueError("Error in solvePnP")
        
        R, _ = cv.Rodrigues(rvec)
        
    elif method == "Homography":
        moving_homography, _ = cv.findHomography(objectPoints, imagePoints)
            
        inv_K__H = np.linalg.inv(cameraMatrix) @ moving_homography
        r1 = inv_K__H[:, 0]
        r2 = inv_K__H[:, 1]
        t = inv_K__H[:, 2]
        
        alpha = 2 / (np.linalg.norm(r1) + np.linalg.norm(r2))
        
        RT = (inv_K__H / alpha)
        
        r1 = RT[:, 0]
        r2 = RT[:, 1]
        t = RT[:, 2]
        r3 = np.cross(r1, r2)
        Q = np.column_stack([r1, r2, r3])
        
        U, _, Vt = np.linalg.svd(Q)
        R = U @ Vt
        
    res = -R.T @ t
    res = res / np.linalg.norm(res)
    
    assert np.sqrt(res[0]**2 + res[1]**2) <= 1, "Error: Light source outside the unit circle"
    assert res[2] >= 0, "Error: Light source behind the camera"
    
    return res

def compute_light(coin_number:int, debug=True, debug_moving=False, debug_static=False)->None: 
    """
    Compute the light directions for a given coin number.

    Args:
        coin_number (int): Number of the coin to compute the light source: 1, 2, 3, or 4.
        debug (bool, optional): Debug flag: show static and moving frames and the corresponding light direction. Defaults to True.
        debug_moving (bool, optional): show the thresholded moving frame. Defaults to False.
        debug_static (bool, optional): show the thresholded static frame. Defaults to False.
    """

    filename = f"coin{coin_number}"
    
    video1_path = os.path.join(static_path, filename+".mov")
    video2_path = os.path.join(moving_path, filename+".mp4")

    align_videos(video1_path, video2_path)

    a_video1_path = os.path.join(aligned_static_path, filename+".mov")
    a_video2_path = os.path.join(aligned_moving_path, filename+".mp4")

    cap_static = cv.VideoCapture(a_video1_path)
    cap_moving = cv.VideoCapture(a_video2_path)

    if debug:
        cv.namedWindow("Static", cv.WINDOW_NORMAL)
        cv.namedWindow("Moving", cv.WINDOW_NORMAL)
        cv.resizeWindow("Static", 640, 480)
        cv.resizeWindow("Moving", 640, 480)

    mtx, dist, rvecs, tvecs = load_camera_parameters("moving_light")

    is_first_frame = True

    skipped_static = 0
    skipped_moving = 0

    MLIC = []
    L_poses = []

    U_coin = []
    V_coin = []

    print("Frames static:", cap_static.get(cv.CAP_PROP_FRAME_COUNT), "Frames moving:", cap_moving.get(cv.CAP_PROP_FRAME_COUNT))
    print("FPS static", cap_static.get(cv.CAP_PROP_FPS), "FPS moving", cap_moving.get(cv.CAP_PROP_FPS))
    
    bar = tqdm(total=cap_static.get(cv.CAP_PROP_FRAME_COUNT), desc="Processing frames... skipped static: 0 skipped moving: 0")
    
    while cap_static.isOpened() or cap_moving.isOpened():
        ret_static, frame_static = cap_static.read()
        ret_moving, frame_moving = cap_moving.read()
        
        if not ret_static or not ret_moving:
            break
        
        if is_first_frame:
            marker_static, midpoint_static = detect_fiducial_marker(frame_static, debug=debug_static)
        
        frame_moving = cv.undistort(frame_moving, mtx, dist)
        marker_moving, midpoint_moving = detect_fiducial_marker(frame_moving, debug=debug_moving)
        
        if marker_static is None:
            is_first_frame = True
            skipped_static += 1
            bar.update(1)
            bar.desc = f"Processing frames... skipped static: {skipped_static} skipped moving: {skipped_moving}"
            continue
        
        if marker_moving is None:
            skipped_moving += 1
            bar.update(1)
            bar.desc = f"Processing frames... skipped static: {skipped_static} skipped moving: {skipped_moving}"
            continue
        
        if debug:
            cv.imshow("Static", plot_fiducial_marker(frame_static, marker_static, midpoint_static))
            cv.imshow("Moving", plot_fiducial_marker(frame_moving, marker_moving, midpoint_moving))
        
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        if is_first_frame:
            #"crop" static image using the static marker
            _, _, w_static, h_static = cv.boundingRect(marker_static)
            marker_reference_points_static = np.array([ [0, 0, 1], [0, h_static, 1], 
                                                        [w_static, h_static, 1], [w_static, 0, 1]  ],dtype=np.float32)
            static_homography, _ = cv.findHomography(marker_static.astype(np.float32), marker_reference_points_static)
        
        coin = cv.warpPerspective(frame_static, static_homography, (w_static,h_static))
        coin = cv.flip(coin, 0) 
        
        #convert to YUV and get the Y channel
        coin_yuv = cv.cvtColor(coin, cv.COLOR_BGR2YUV)
        coin_y = coin_yuv[:,:,0]
        MLIC.append(coin_y)
        
        #take the U and V components
        U_coin.append(coin_yuv[:,:,1])
        V_coin.append(coin_yuv[:,:,2])
        
        #compute the light source
        x, y, w, h = cv.boundingRect(marker_moving)
        marker_reference_points = np.array([[0, 0, 1], [0, h, 1], [w, h, 1], [w, 0, 1]  ], dtype=np.float32)
        
        res = calculateLightPosition(objectPoints=marker_reference_points,
                            imagePoints=marker_moving.astype(np.float32),
                            cameraMatrix=mtx, dist=dist)
        
        L_poses.append(res)
        
        if debug:
            cv.imshow("Coin", coin)
            cv.imshow("Light source", plot_light_source(res[0], res[1]))
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        bar.update(1)
        is_first_frame = False
        
    MLIC = np.array(MLIC)
    L_poses = np.array(L_poses)
    U_hat = np.mean(np.array(U_coin), axis=0)
    V_hat = np.mean(np.array(V_coin), axis=0)
    
    cap_static.release()
    cap_moving.release()
    cv.destroyAllWindows()
    
    result_path = f"./results_intermediate"
    os.makedirs(result_path, exist_ok=True)
    
    result_path = os.path.join(result_path, f"{filename}.npz")
    np.savez(result_path, MLIC=MLIC, L_poses=L_poses, V_hat=V_hat, U_hat=U_hat)

def load_light_results(filename:str)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the results of the light computation.

    Args:
        filename (str): Name of the file to load: "coin1", "coin2", "coin3", or "coin4".

    Raises:
        FileNotFoundError: if the file is not found.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: MLIC, L_poses, U_hat, V_hat. 
        
        MLIC is the light intensity coin, L_poses are the light poses, 
        U_hat and V_hat are the U and V mean components of the coin.
    """
    
    path = f"results_intermediate/{filename}.npz"
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: '{path}': did you run compute_light?")
    
    results = np.load(path)
    MLIC = results['MLIC']
    L_poses = results['L_poses']
    U_hat = results['U_hat']
    V_hat = results['V_hat']
    
    return MLIC, L_poses, U_hat, V_hat