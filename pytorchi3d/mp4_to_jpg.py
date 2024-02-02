import cv2
from pathlib import Path
from pytorchi3d.utils import createDirectory
import os
import glob

def change_fps(file_name, fps):
    input_name = file_name
    file_name = file_name.replace("webm", "mp4")
    # cmd = 'ffmpeg -i "{input}" -c:v libx264 -c:a libmp3lame -b:a 384K "{output}"'.format(
    #                                                 input = input_name, 
    #                                                 output = output)

    cmd = 'ffmpeg -i "{}" -filter:v fps={} "{}" -y'.format(input_name, fps, file_name) #os.path.join(this_path, save_file_id))

    os.system(cmd)

    return file_name

def convert_mp4_to_jpg(file_path, save_path):
    
    video_name = Path(file_path).stem
    print('convert mp4 to jpg :', video_name)

    cap = cv2.VideoCapture(file_path)
    FPS = cap.get(5)
    count = 1
    #print(save_path + f"/{video_name}")
    createDirectory(save_path + f"/{video_name}")
    
    if FPS != 24:
        new_file_path = change_fps(file_path, fps=24)
        cap = cv2.VideoCapture(new_file_path)
        FPS = cap.get(5)
        
    frame = cap.get(7)
    
    for i in range(1, int(frame)):
        success, img = cap.read()
            
        if not success:
            break

        img = cv2.resize(img, (640, 360))
        #print(save_path + f"/{video_name}" + '/image' + '-'+str(count).zfill(6)+'.jpg')
        cv2.imwrite(save_path + f"/{video_name}" + '/image' + '-'+str(count).zfill(6)+'.jpg', img)
        count+=1
    
    return FPS, save_path + f"/{video_name}"

if __name__ == "__main__":
    list_file_paths = glob.glob('/home/www/data/data/saigonmusic/Dev_AI/kiendn/dataset/horror_film/MKoffical/*.*')
    for p in list_file_paths:
        convert_mp4_to_jpg(p, '/home/www/data/data/saigonmusic/Dev_AI/kiendn/dataset/horror_film/images/MKoffical')