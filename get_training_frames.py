#Get training images

import cv2  # import the opencv library
import os  # import the os library for interacting with the operating system
import numpy as np  # import numpy library
from tqdm import tqdm  # import tqdm (progress bars)
from collections import deque  # import deque

base_dir = os.path.expanduser("~/Desktop/NYU/For William Wei") # base directory where files are stored

video_filename = '1minlaser.mp4'  # name of the video file
video_path = os.path.join(base_dir, video_filename)  # full path to the video file

output_dir = os.path.join(base_dir, 'VideoinFrames')  # directory to save extracted frames
os.makedirs(output_dir, exist_ok=True)  # create the directory if it doesn't exist

output_key_frame_dir = os.path.join(base_dir, 'KeyFrames')  # directory to save frames with contours
os.makedirs(output_key_frame_dir, exist_ok=True)  # create the directory if it doesn't exist

def get_frames():
    cap = cv2.VideoCapture(video_path)  # open the video file
    if not cap.isOpened():  # check if the video file was opened successfully
        print(f"[Error] Could not open video file at {video_path}")  # print an error message
        exit()  # exit the program

    frame_count = 0  # initialize the frame counter
    fps = cap.get(cv2.CAP_PROP_FPS)  # get the frames per second (fps) of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get the total number of frames in the video

    # loop over all frames in the video and save them as images
    for _ in tqdm(range(total_frames), desc="Extracting frames"):  # show a progress bar for frame extraction
        ret, frame = cap.read()  # read a frame from the video
        if not ret:  # check if the frame was read successfully
            break  # exit the loop if there are no more frames

        # construct the output path for saving the current frame
        frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')

        # save the frame as an image file
        cv2.imwrite(frame_path, frame)
        frame_count += 1  # increment the frame counter

    cap.release()  # release the video capture object
    print(f"Finished saving {frame_count} frames to {output_dir}")  # print a message indicating completion
    return frame_count


def move_key_frames(total_frames):
    for frame_count in range(int(total_frames/100)):
        frame_path = os.path.join(output_dir, f'frame_{frame_count*100:04d}.jpg')
        dst_path = os.path.join(output_key_frame_dir, f'frame_{frame_count*100:04d}.jpg')
        if (os.path.exists(frame_path)):
            os.rename(frame_path,dst_path)
        else:
            print(f"Frame at {frame_path} does not exist.")

def main():
    move_key_frames(3602)

if __name__ == "__main__":
    main()

