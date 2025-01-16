import os
import numpy as np
import cv2 as cv
import moviepy as mp

import multiprocessing

from calibration import load_camera_parameters

import matplotlib.pyplot as plt
import librosa

from tqdm import tqdm
import subprocess

# Paths to the input videos
static_path = "./data/cam1 - static/"
moving_path = "./data/cam2 - moving light/"

# Paths to the output videos
aligned_static_path = "./data_aligned/cam1 - static/"
aligned_moving_path = "./data_aligned/cam2 - moving light/"

def align_videos(video1_path, video2_path, audio_sr=10000):

    def extract_audio_from_video(video_path):
        """
        Extract and load audio from a video file.

        Parameters:
        - video_path: str, path to the video file.
        - sr: int, target sample rate for the audio.

        Returns:
        - audio: ndarray, extracted mono audio signal at the specified sample rate.
        """
        # Load the video file
        video = mp.VideoFileClip(video_path)
        audio = video.audio.to_soundarray()

        # Convert to mono if the audio is stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Average the channels to get mono

        return audio, video.audio.fps


    def find_offset(audio1, audio2, sr):
        """
        Find the time offset between two audio tracks using cross-correlation.

        Parameters:
        - audio1: ndarray, first audio signal.
        - audio2: ndarray, second audio signal.
        - sr: int, sample rate of the audio signals.

        Returns:
        - time_offset: float, time offset in seconds where audio2 should be shifted
                    to align with audio1.
        """
        # Ensure audio1 and audio2 have the same length for comparison
        max_len = min(len(audio1), len(audio2), 15 * sr)  # Limit to the first 15 seconds
        audio1 = audio1[:max_len]
        audio2 = audio2[:max_len]

        # Compute cross-correlation
        correlation = np.correlate(audio1, audio2, mode="full")

        # Find the index of the maximum correlation
        delay = np.argmax(correlation) - (len(audio2) - 1)  # Adjust to center alignment

        # Convert delay to time offset in seconds
        time_offset = delay / sr

        return time_offset

    def crop_video(video_path, offset):
        """Crop video based on the offset."""
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
    
    # Extract audio tracks
    audio1, sr1 = extract_audio_from_video(video1_path)
    audio2, sr2 = extract_audio_from_video(video2_path)

    audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=audio_sr)
    audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=audio_sr)
    
    # Find time offset
    offset = find_offset(audio1, audio2, audio_sr)
    print(f"Time offset: {offset:.2f} seconds")
    
    if offset > 0:
        video1 = crop_video(video1_path, offset)
        video2 = mp.VideoFileClip(video2_path)
    else:
        video1 = mp.VideoFileClip(video1_path)
        video2 = crop_video(video2_path, offset) 

    #set fps to 30
    video1 = video1.with_fps(30)
    video2 = video2.with_fps(30)
    
    #set duration to the shorter video
    duration = min(video1.duration, video2.duration)
    video1 = video1.subclipped(0, duration)
    video2 = video2.subclipped(0, duration)
    
    # Write the aligned videos to disk
    
    os.makedirs(aligned_static_path, exist_ok=True)
    os.makedirs(aligned_moving_path, exist_ok=True)
    
    output1 = os.path.join(aligned_static_path, video1_path.split('/')[-1])
    output2 = os.path.join(aligned_moving_path, video2_path.split('/')[-1])
    
    video1.write_videofile(output1, codec="libx264", preset="ultrafast", threads=multiprocessing.cpu_count(), audio=True)
    video2.write_videofile(output2, codec="libx264", preset="ultrafast", threads=multiprocessing.cpu_count(), audio=True)

def detect_fiducial_marker(image, threshold=100, debug=False):
    
    image_grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # Step 1: Otsu's Thresholding
    
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

def find_nearest_point(points, point):
    distances = np.linalg.norm(points - point, axis=1)
    return points[np.argmin(distances)]
    
def plot_fiducial_marker(image, marker, midpoint):
    
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

def order_points_clockwise(points, midpoint):
    
    if points is None or midpoint is None:
        return None
    
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

def plot_light_source(u,v, convert=True, image_size=200):
    
    if convert:
        u = int(((u + 1) / 2) * (image_size-1))
        v = int(((v + 1) / 2) * (image_size-1))
    
        # Adjust for Cartesian coordinates
        v = image_size - v - 1
    
    image = np.zeros((image_size, image_size, 1), dtype=np.uint8)
    
    cv.circle(image, (image_size//2, image_size//2), image_size//2, 255, 2)
    
    cv.circle(image, (u, v), 2, 255, -1)
    cv.line(image, (image_size//2, image_size//2), (u, v), 255, 1)
    
    #image = cv.flip(image, 0)
    
    return image

def getLightPose(objectPoints, imagePoints, cameraMatrix, dist, method="PnP"):
    
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
    
    #print(res)
    
    if np.sqrt(res[0]**2 + res[1]**2) > 1:
        print("Error: Light source outside the unit circle")
    
    if res[2] < 0:
        print("Error: Light source behind the camera")
    
    return res

def analysis(filename="coin1", debug=True, debug_moving=False, debug_static=False): 

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

    first = True

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
        
        if first:
            marker_static, midpoint_static = detect_fiducial_marker(frame_static, debug=debug_static)
            first = False
        
        frame_moving = cv.undistort(frame_moving, mtx, dist)
        marker_moving, midpoint_moving = detect_fiducial_marker(frame_moving, debug=debug_moving)
        
        if marker_static is None:
            first = True
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
        
        #crop static image using the marker
        x, y, w, h = cv.boundingRect(marker_static)
        marker_reference_points = np.array([[0, 0, 1], [0, h, 1], [w, h, 1], [w, 0, 1]  ], dtype=np.float32)
        static_homography, _ = cv.findHomography(marker_static.astype(np.float32), marker_reference_points)
        
        coin = cv.warpPerspective(frame_static, static_homography, (w,h))
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
        
        res = getLightPose(objectPoints=marker_reference_points,
                            imagePoints=marker_moving.astype(np.float32),
                            cameraMatrix=mtx, dist=dist)
        
        L_poses.append(res)
        
        if debug:
            cv.imshow("Coin", coin)
            cv.imshow("Light source", plot_light_source(res[0], res[1]))
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        bar.update(1)

    MLIC = np.array(MLIC)
    L_poses = np.array(L_poses)
    U_hat = np.mean(U_coin, axis=0)
    V_hat = np.mean(V_coin, axis=0)
    
    cap_static.release()
    cap_moving.release()
    cv.destroyAllWindows()
    
    result_path = f"./results_intermediate"
    os.makedirs(result_path, exist_ok=True)
    
    result_path = os.path.join(result_path, f"{filename}.npz")
    np.savez(result_path, MLIC=MLIC, L_poses=L_poses, V_hat=V_hat, U_hat=U_hat)
    
    return MLIC, L_poses, U_hat, V_hat

def load_results(filename):
    results = np.load(f"./results_intermediate/{filename}.npz")
    MLIC = results['MLIC']
    L_poses = results['L_poses']
    U_hat = results['U_hat']
    V_hat = results['V_hat']
    
    return MLIC, L_poses, U_hat, V_hat

def plot_pixel(x, y, MLIC, L_poses, regular_grids={}):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_ylim(-1, 1)
    ax[0].set_xlim(-1, 1)
    scatter = ax[0].scatter(L_poses[:, 0], L_poses[:, 1], c=MLIC[:, y, x], cmap='viridis', s=2)
    ax[0].set_title(f"f({x}, {y}, ...)")
    ax[0].set_xlabel("U")
    ax[0].set_ylabel("V")
    fig.colorbar(scatter, ax=ax[0])

    if regular_grids != {}:
        ax[1].matshow(regular_grids[(x, y)])
        ax[1].set_title("Interpolation")
    else:
        ax[1].set_title("No interpolation")
    ax[1].axis('off')
    plt.gca().invert_yaxis()

    plt.show()

if __name__ == "__main__":
    
    filename = "coin1"
    regular_grid_dim = (100, 100)
    resize_dim = (256, 256)
    nprocesses = -1
    method = "RBF"
    
    #analysis(filename=filename, debug=True, debug_moving=False, debug_static=False)
    
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
    
    MLIC, L_poses, U_hat, V_hat = load_results(filename)
    
    MLIC_resized = [cv.resize(coin, (200, 200)) for coin in MLIC]
    MLIC_resized = np.array(MLIC_resized)
    
    regular_grids = np.load(f"./results/RBF/{filename}.npy", allow_pickle=True).item()
    
    plot_pixel(10, 20, MLIC_resized, L_poses, regular_grids=regular_grids)

