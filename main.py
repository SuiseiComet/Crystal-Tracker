from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class Custom_DeepSORT_Tracker:
    def __init__(self):
        # Move initialize variables into local storage.
       self.tracker = DeepSort(max_cosine_distance=0,n_init=15, max_age=120)

    def __del__(self):
        pass

    def update_tracker(self, detections_from_frame):
        bbs = []
        frame_results = detections_from_frame[0]

        if frame_results.obb.xywhr.cpu().numpy().tolist():

            num_of_obj = len(frame_results.obb.xywhr.cpu().numpy().tolist())

            for object_index in range(num_of_obj):
                box = frame_results.obb.xywhr.cpu().numpy().tolist()[object_index][:4]
                conf = frame_results.obb.conf.cpu().numpy().tolist()[object_index]
                classes = frame_results.obb.cls.cpu().numpy().tolist()[object_index]
                bb = (box, conf, classes)
                bbs.append(bb)

        # bbs expected to be a list of detections, each in tuples of 
        # ( [left,top,width,height], confidence, detection_class )
        tracks = self.tracker.update_tracks(bbs, frame=frame_results.orig_img)
        return tracks


# function to convert images to a video
def images_to_video(input_folder, output_video_path, fps=30):
    files = os.listdir(input_folder)  # list all files in the input folder
    files.sort()  # sort files by name

    frame_size = None
    out = None

    for i, file_name in enumerate(tqdm(files, desc="Creating video")):  # show a progress bar for video creation
        file_path = os.path.join(input_folder, file_name)  # get the full path of the current file
        frame = cv2.imread(file_path)  # read the image file

        if out is None:
            frame_size = frame.shape[:2][::-1]  # get the size of the frame
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # define the codec for video writing
            out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)  # create a video writer object

        out.write(frame)  # write the frame to the video

    if out is not None:
        out.release()  # release the video writer object

    print(f"Video saved successfully at: {output_video_path}")  # print a message indicating completion




import cv2  # import the opencv library
import os  # import the os library for interacting with the operating system
import numpy as np  # import numpy library
from tqdm import tqdm  # import tqdm (progress bars)
from collections import deque  # import deque    

base_dir = os.path.expanduser(r'F:\Data for Thesis') # base directory where files are stored
video_path = r'C:\Users\billy\Desktop\NYU\For William Wei\30sec.mp4'

output_dir = os.path.join(base_dir, 'SavedFramesYolo')  # directory to save extracted frames
os.makedirs(output_dir, exist_ok=True)  # create the directory if it doesn't exist

def main():
    # Make an object using our class
    tracker = Custom_DeepSORT_Tracker()

    # Load the YOLOv8 model
    model = YOLO(r'C:\Users\billy\Desktop\NYU\Crystal Tracking\runs\obb\train5\weights\best.pt')


    # Perform inference on an image
    for i in range(200):
        image_path = os.path.join(r'F:\Data for Thesis\VideoinFrames', f'frame_{i:04d}.jpg')
        results = model(image_path)

        # Run DeepSORT on each frame
        tracks = tracker.update_tracker(results)

        img = cv2.imread(image_path)
        for track in tracks:
            cv2.putText(img, f'Crystal {track.track_id}', (int(track.to_ltwh()[0] - 10), int(track.to_ltwh()[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # label the crystal

            output_image_path = os.path.join(output_dir, os.path.basename(image_path))
            cv2.imwrite(output_image_path, img)
        #cv2.imshow("image", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    output_video_path = os.path.join(output_dir, "video.mp4")
    images_to_video(output_dir, output_video_path)
    
    #print(len(tracks))
    # Print tracks
    for track in tracks:
        print(track.to_ltwh())
        print(track.det_conf)
        print(track.det_class)
        print(track.track_id)




main()