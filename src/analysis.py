import os
import numpy as np
import cv2 as cv
import moviepy as mp
from PIL import Image
from io import BytesIO
from cairosvg import svg2png

import multiprocessing

from calibration import load_camera_parameters

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


filename = "coin1"

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

print(os.path.exists(os.path.join("./data/", "marker.svg")))

with open(os.path.join("./data/", "marker.svg"), "r") as f:
    svg_data = f.read()

png = svg2png(bytestring=svg_data)
pil_img = Image.open(BytesIO(png)).convert('RGBA')
marker = cv.cvtColor(np.array(pil_img), cv.COLOR_RGBA2BGRA)

#cv.imshow("Marker", marker)

while cap_static.isOpened() and cap_moving.isOpened():
    ret_static, frame_static = cap_static.read()
    ret_moving, frame_moving = cap_moving.read()
    
    if not ret_static or not ret_moving:
        break
    
    frame_moving = cv.undistort(frame_moving, mtx, dist)
    
    img_g = frame_static.astype(np.float32)
    img_g = img_g - np.mean(img_g)
    conv = cv.filter2D( img_g, -1,  marker)
    maxval = np.amax( conv )
    
    if maxval>5E6: # Correlation threshold
        maxpos = np.unravel_index( np.argmax( conv ), conv.shape )
    else:
        maxpos=[-999,-999]
    
    
    cv.rectangle( frame_static, (maxpos[1]-15, maxpos[0]-15),
                         (maxpos[1]+15, maxpos[0]+15), 
                         (255,0,0), 2 )
    
    cv.imshow("Static", frame_static)
    cv.imshow("Moving", frame_moving)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap_static.release()
cap_moving.release()
cv.destroyAllWindows()

