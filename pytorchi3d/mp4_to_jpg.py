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
    #extension = file_name.split('.')[-1]
    #file_name = file_name.replace(extension, "mp4")
    # cmd = 'ffmpeg -i "{input}" -c:v libx264 -c:a libmp3lame -b:a 384K "{output}"'.format(
    #                                                 input = input_name, 
    #                                                 output = output)

    cmd = 'ffmpeg -i "{}" -filter:v fps={} "{}" -y'.format(file_path, fps, output_path) #os.path.join(this_path, save_file_id))

    os.system(cmd)

    return output_path

def convert_mp4_to_jpg(file_path, save_path):
    
    video_name = Path(file_path).stem
    print('convert mp4 to jpg :', video_name)

    cap = cv2.VideoCapture(file_path)
    FPS = cap.get(5)
    count = 1
    #print(save_path + f"/{video_name}")
    if createDirectory(save_path + f"/{video_name}"):
        return 24, save_path + f"/{video_name}"
    
    # if FPS != 24:
    new_file_path = change_fps(file_path, fps=24)
    new_file_path = file_path
    cap = cv2.VideoCapture(new_file_path)
    FPS = cap.get(5)
        
    frame = cap.get(7)
    
    for i in range(1, int(frame)):
        success, img = cap.read()
            
        if not success:
            print('not')
            break

        img = cv2.resize(img, (640, 360))
        #print(save_path + f"/{video_name}" + '/image' + '-'+str(count).zfill(6)+'.jpg')
        cv2.imwrite(save_path + f"/{video_name}" + '/image' + '-' + str(count).zfill(6)+'.jpg', img)
        count+=1
    
    return FPS, save_path + f"/{video_name}"

if __name__ == "__main__":
    #list_file_paths = glob.glob('/home/www/data/data/saigonmusic/Dev_AI/kiendn/dataset/horror_film/MKoffical/*.*')
    list_file_paths = glob.glob('/home/www/data/data/saigonmusic/Dev_AI/kiendn/dataset/horror_film/Bloody/*.*')
    #list_file_paths = ["/home/www/data/data/saigonmusic/Dev_AI/kiendn/dataset/horror_film/kill/Child's Play (1988) - Dr. Death's Voodoo Scene (7⧸12) ｜ Movieclips [CSuwuvVcRHU].webm"]
    list_file_paths = random.sample(list_file_paths, len(list_file_paths))
    l = []
    save_path = '/home/www/data/data/saigonmusic/Dev_AI/kiendn/dataset/horror_film/images/Bloody'
    for i in list_file_paths:
        video_name = Path(i).stem
        if not os.path.exists(save_path + f"/{video_name}"):
            l.append(i)
    for p in l:
        convert_mp4_to_jpg(p, '/home/www/data/data/saigonmusic/Dev_AI/kiendn/dataset/horror_film/images/test2')