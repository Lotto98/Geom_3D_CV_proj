import os
import numpy as np
import cv2 as cv
import moviepy as mp

import multiprocessing

from calibration import load_camera_parameters

import matplotlib.pyplot as plt

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

filename = "coin4"

video1_path = os.path.join(static_path, filename+".mov")
video2_path = os.path.join(moving_path, filename+".mp4")

align_videos(video1_path, video2_path)

a_video1_path = os.path.join(aligned_static_path, filename+".mov")
a_video2_path = os.path.join(aligned_moving_path, filename+".mp4")

cap_static = cv.VideoCapture(a_video1_path)
cap_moving = cv.VideoCapture(a_video2_path)


cv.namedWindow("Static", cv.WINDOW_NORMAL)
cv.namedWindow("Moving", cv.WINDOW_NORMAL)
cv.resizeWindow("Static", 640, 480)
cv.resizeWindow("Moving", 640, 480)

mtx, dist, rvecs, tvecs = load_camera_parameters("moving_light")

first = True

skipped_static = 0
skipped_moving = 0

while cap_static.isOpened() and cap_moving.isOpened():
    ret_static, frame_static = cap_static.read()
    ret_moving, frame_moving = cap_moving.read()
    
    if not ret_static or not ret_moving:
        break
    
    if first:
        marker_static, midpoint_static = detect_fiducial_marker(frame_static)
        
        first = False
    
    frame_moving = cv.undistort(frame_moving, mtx, dist)
    marker_moving, midpoint_moving = detect_fiducial_marker(frame_moving, debug=False)
    
    if marker_static is None:
        first = True
        skipped_static += 1
        print("skipped static", skipped_static)
        continue
    
    if marker_moving is None:
        skipped_moving += 1
        print("skipped moving", skipped_moving)
        continue
    
    cv.imshow("Static", plot_fiducial_marker(frame_static.copy(), marker_static, midpoint_static))
    cv.imshow("Moving", plot_fiducial_marker(frame_moving.copy(), marker_moving, midpoint_moving))
    
    #crop static image using the marker
    roi = cv.boundingRect(marker_static)
    x, y, w, h = roi
    coin = frame_static[y:y+h, x:x+w]
    cv.imshow("Coin", coin)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    dest_points = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32)
    
    moving_homography, bho = cv.findHomography(marker_moving, dest_points)
    
    retval, rotations, translations, normals = cv.decomposeHomographyMat(moving_homography, mtx)
    
    i = 0
    while True:
        rotation = rotations[i]
        translation = translations[i]
        
        res = (-rotation.T @ translation)
        res = res / np.linalg.norm(res)
    
        if res[2] < 0:
            i += 1
        else:
            break
    
    if res[2] < 0:
        print(res)
    
    #warped = cv.warpPerspective( frame_moving, moving_homography[0], (500,500) )
    #cv.imshow("warped", warped )
    
cap_static.release()
cap_moving.release()
cv.destroyAllWindows()

