import numpy as np
import cv2
from skimage.transform import resize
import torch

def capture(filename,timesep,rgb,h,w,frame_interval):
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    times = []

    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
    
        if not ret:
            break
        frm = resize(frame,(h, w,rgb))
        frm = np.expand_dims(frm,axis=0)
        frm = np.moveaxis(frm, -1, 1)
        if(np.max(frm)>1):
            frm = frm/255.0
        frames.append(frm)
        timestamp = i / fps
        times.append(timestamp)
        
    cap.release()
    
    num_frames = (np.shape(frames)[0] // timesep) * timesep
    
    frames = np.vstack(frames[:num_frames])
    torch_frames = torch.from_numpy(np.float32(frames)).view(-1,timesep,rgb,h,w)
    times = times[:num_frames]
    l_intervals = []
    start = times[0]
    for i in range(1,num_frames):
        if i%timesep == 0:
            end = times[i-1]
            l_intervals.append((start,end))
            start = times[i]
    l_intervals.append((start, times[num_frames-1]))
    return torch_frames, l_intervals

def resume_checkpoint(resume_path, model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model'])