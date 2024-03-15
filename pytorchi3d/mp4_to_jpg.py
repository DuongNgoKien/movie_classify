import cv2
from pathlib import Path
from pytorchi3d.utils import createDirectory
import os
import glob
import random
import ntpath

def change_fps(file_path, fps):
    output_name = ntpath.basename(file_path)
    output_path = file_path.replace(output_name, 'standard_'+output_name)

    if os.path.exists(output_path):
        return output_path

    cmd = 'ffmpeg -i "{}" -filter:v fps={} "{}" -y'.format(file_path, fps, output_path)

    os.system(cmd)

    return output_path

def convert_mp4_to_jpg(file_path, save_path):
    
    video_name = Path(file_path).stem
    print('convert mp4 to jpg :', video_name)

    cap = cv2.VideoCapture(file_path)
    FPS = cap.get(5)
    count = 1

    createDirectory(save_path + f"/{video_name}")

    new_file_path = change_fps(file_path, fps=24)
    cap = cv2.VideoCapture(new_file_path)
    FPS = cap.get(5)
        
    frame = cap.get(7)
    
    for i in range(1, int(frame)):
        success, img = cap.read()
            
        if not success:
            print('Can not open this video. Please try again.')
            break

        img = cv2.resize(img, (640, 360))

        cv2.imwrite(save_path + f"/{video_name}" + '/image' + '-' + str(count).zfill(6) + '.jpg', img)
        count += 1
    
    return FPS, save_path + f"/{video_name}"

if __name__ == "__main__":
    pass