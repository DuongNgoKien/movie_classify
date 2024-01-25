import cv2
from pathlib import Path
from pytorchi3d.utils import createDirectory

def convert_mp4_to_jpg(file_path, save_path):
    
    video_name = Path(file_path).stem
    print('convert mp4 to jpg :', video_name)

    cap = cv2.VideoCapture(file_path)
    FPS = cap.get(5) 
    frame = cap.get(7)
    count = 1
    #print(save_path + f"/{video_name}")
    createDirectory(save_path + f"/{video_name}")
    
    if FPS == 30:
        for i in range(1, int(frame)+1):
            success, img = cap.read()
            
            if not success:
                break

            img = cv2.resize(img, (640, 360))
            if i % (1.25) < 1:
                cv2.imwrite(save_path + f"/{video_name}" + '/image' + '-'+str(count).zfill(6)+'.jpg', img)
                count+=1
    elif FPS == 24:
        for i in range(1, int(frame)+1):
            success, img = cap.read()
            
            if not success:
                break

            img = cv2.resize(img, (640, 360))
            #print(save_path + f"/{video_name}" + '/image' + '-'+str(count).zfill(6)+'.jpg')
            cv2.imwrite(save_path + f"/{video_name}" + '/image' + '-'+str(count).zfill(6)+'.jpg', img)
            count+=1
    else:
        print("*"*30)
        print(f"FPS : {FPS} - {file_path}")
        print("*"*30)
    
    return FPS, save_path + f"/{video_name}"
