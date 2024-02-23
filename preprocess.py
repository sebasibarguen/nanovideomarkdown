import os

import cv2
from pydub import AudioSegment


VIDEO_FILE_PATH = "data/video.mp4"
AUDIO_FILE_PATH = "data/audio.mp3"

FRAMES_DIR = "data/frames"


def extract_audio(video_file_path, audio_file_path):

    audio = AudioSegment.from_file(video_file_path)
    audio.export(audio_file_path, format="mp3")


def split_audio_into_chunks(audio_file_path, chunk_size=5 * 60 * 1000):

    audio_chunk_dir = "data/audio_chunk"

    if not os.path.exists(audio_chunk_dir):
        os.makedirs(audio_chunk_dir)

    audio = AudioSegment.from_mp3(audio_file_path)

    chunks = list(audio[::chunk_size])

    for i, chunk in enumerate(chunks):
        chunk.export(f"{audio_chunk_dir}/{i}.mp3", format="mp3")


def extract_frames(video_path, frames_dir):
    """
    Extracts frames from a video file and saves them in a directory.
    Use cv2 and set fps to 1 to extract one frame per second.
    Frame name is the frame number and the time in seconds.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Capture the video
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS)) # Get frames per second of the video

    success, frame = video.read()
    count = 0
    second = 0

    while success:
        # Only save one frame per second
        if count % fps == 0:
            # Save frame as JPEG file
            cv2.imwrite(f"{frames_dir}/frame_{second:04d}.jpg", frame)
            second += 1
        
        success, frame = video.read()
        count += 1

    video.release()


extract_audio(VIDEO_FILE_PATH, AUDIO_FILE_PATH)

split_audio_into_chunks(AUDIO_FILE_PATH)

extract_frames(VIDEO_FILE_PATH, FRAMES_DIR)
