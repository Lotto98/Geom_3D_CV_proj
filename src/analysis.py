import os
import numpy as np
import cv2 as cv
import moviepy as mp

import multiprocessing

from calibration import load_camera_parameters

import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.interpolate import Rbf

# Paths to the input videos
static_path = "./data/cam1 - static/"
moving_path = "./data/cam2 - moving light/"

# Paths to the output videos
aligned_static_path = "./data_aligned/cam1 - static/"
aligned_moving_path = "./data_aligned/cam2 - moving light/"

def align_videos(video1_path, video2_path, audio_sample_rate=10000):
    
    def extract_audio_from_video(video_path, sr):
        """Extract and load audio from a video file."""
        video = mp.VideoFileClip(video_path)
        audio = video.audio.to_soundarray(fps=sr)
        # Ensure audio is at least 2D, convert to mono
        if audio.ndim == 1:
            return audio  # Already mono
        return np.mean(audio, axis=1)  # Convert stereo to mono

    def find_offset(audio1, audio2, sr):
        """Find the time offset between two audio tracks using cross-correlation."""
        #use the first 15 seconds of the audio
        correlation = np.correlate(audio1[:15*sr], audio2[:15*sr], mode="full")
        delay = np.argmax(correlation) - 15*sr
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
    audio1 = extract_audio_from_video(video1_path, audio_sample_rate)
    audio2 = extract_audio_from_video(video2_path, audio_sample_rate)

    # Find time offset
    offset = find_offset(audio1, audio2, audio_sample_rate)
    print(f"Time offset: {offset:.2f} seconds")
    
    if offset > 0:
        video1 = mp.VideoFileClip(video1_path)
        video2 = crop_video(video2_path, offset)
    else:
        video1 = crop_video(video1_path, -offset)
        video2 = mp.VideoFileClip(video2_path)

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
    
    video1.write_videofile(output1, codec="libx264", preset="ultrafast", threads=multiprocessing.cpu_count(), audio=False)
    video2.write_videofile(output2, codec="libx264", preset="ultrafast", threads=multiprocessing.cpu_count(), audio=False)

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
        
    #plot the midpoint
    cv.circle(image, tuple(midpoint), 10, (0, 0, 255), -1)
    
    #plot the marker
    cv.polylines(image, [marker], isClosed=True, color=(0, 255, 0), thickness=6)
    for idx, corner in enumerate(marker):
        cv.circle(image, tuple(corner), 7, (0, 0, 255), -1)
        cv.putText(image, f"c{idx}", tuple(corner), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
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

def plot_light_source(u,v, image_size=200):
    
    u = int(((u + 1) / 2) * image_size)
    v = int(((v + 1) / 2) * image_size)
    
    image = np.zeros((image_size, image_size, 1), dtype=np.uint8)
    
    cv.circle(image, (image_size//2, image_size//2), image_size//2, 255, 2)
    
    cv.circle(image, (u, v), 2, 255, -1)
    cv.line(image, (image_size//2, image_size//2), (u, v), 255, 1)
    
    return image

def plot_pixel(x, y, MLIC, L_poses):
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.scatter(L_poses[:,0], L_poses[:,1], c=MLIC[:,y, x], cmap='viridis', s=1,)
    plt.title(f"f({x}, {y}, ...)")
    plt.xlabel("U")
    plt.ylabel("V")
    plt.colorbar()
    plt.show()

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

    U_hat = None
    V_hat = None

    print(cap_static.get(cv.CAP_PROP_FRAME_COUNT), cap_moving.get(cv.CAP_PROP_FRAME_COUNT))
    
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
        
        #crop static image using the marker
        x, y, w, h = cv.boundingRect(marker_static)

        #crop the rectified static image
        dest_points = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32)
        static_homography, _ = cv.findHomography(marker_static, dest_points)
        
        coin = cv.warpPerspective(frame_static, static_homography, (w, h))
        
        #convert to YUV and get the Y channel
        coin_yuv = cv.cvtColor(coin, cv.COLOR_BGR2YUV)
        coin_y = coin_yuv[:,:,0]
        MLIC.append(coin_y)
        
        #compute mean U and V
        if U_hat is None and V_hat is None:
            U_hat = coin_yuv[:,:,1]
            V_hat = coin_yuv[:,:,2]
        else:
            U_hat += coin_yuv[:,:,1]
            V_hat += coin_yuv[:,:,2]
        
        #compute the light source
        moving_homography, _ = cv.findHomography(marker_moving, dest_points)
        
        H = np.linalg.inv(mtx) @ moving_homography
        r1 = H[:, 0] / np.linalg.norm(H[:, 0])
        r2 = H[:, 1] / np.linalg.norm(H[:, 0])
        t = H[:, 2] / np.linalg.norm(H[:, 0])
        res = t / np.linalg.norm(t)
        
        if np.sqrt(res[0]**2 + res[1]**2) > 1 or res[2]<0:
            print("Error")
        
        L_poses.append(res)
        
        if debug:
            cv.imshow("Static", plot_fiducial_marker(frame_static.copy(), marker_static, midpoint_static))
            cv.imshow("Moving", plot_fiducial_marker(frame_moving.copy(), marker_moving, midpoint_moving))
            cv.imshow("Coin", coin)
            cv.imshow("Light source", plot_light_source(res[0], res[1]))
        
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        bar.update(1)

    U_hat = U_hat.astype(float) * 1/len(MLIC)
    V_hat = V_hat.astype(float) * 1/len(MLIC)

    MLIC = np.array(MLIC)
    L_poses = np.array(L_poses)
    
    cap_static.release()
    cap_moving.release()
    cv.destroyAllWindows()
    
    result_path = f"./results_intermediate"
    os.makedirs(result_path, exist_ok=True)
    
    result_path = os.path.join(result_path, f"{filename}.npz")
    np.savez(result_path, MLIC=MLIC, L_poses=L_poses)
    
    return MLIC, L_poses, U_hat, V_hat


if __name__ == "__main__":
    
    filename = "coin1"
    #analysis(filename=filename)

    results = np.load(f"./results_intermediate/{filename}.npz")
    MLIC = results['MLIC']
    L_poses = results['L_poses']

    plot_pixel(109, 200, MLIC, L_poses)
    
    #Rbf = Rbf(y=L_poses[:,:2], d=MLIC, kernel='linear')

