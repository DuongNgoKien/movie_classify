import os
import numpy as np

from utils import *
from pipeline import smoke_drink_detect
from pipeline.detect_scene import infer
from pytorchi3d.mp4_to_jpg import convert_mp4_to_jpg
from torchvggish.torchvggish.mp4_to_wav import convert_mp4_to_avi
from image_detect import detect_violence, detect_pornography
from config import *


def post_predictions(pred, elapsed_seconds, threshold=0.7):
    sum_prob, count, start, end = 0, 0, 0, 0
    if elapsed_seconds.ndim == 2:
        start_seconds = elapsed_seconds[:,0]
        end_seconds = elapsed_seconds[:,1]
    else:
        start_seconds = elapsed_seconds
        end_seconds = elapsed_seconds
    for i in range(pred.shape[0]):
        if pred[i] >= 0.5:
            if count == 0:
                start = start_seconds[i]
            count += 1
            sum_prob += pred[i]
        else:
            if count !=0:
                end = end_seconds[i-1]
                avg_prob = sum_prob/count
                if avg_prob >= threshold:
                    print(str(start) + " -> " + str(end))
                        
                count = 0
                sum_prob = 0
    if count != 0:
        end = end_seconds[i]
        avg_prob = sum_prob/count
        if avg_prob >= threshold:
            print(str(start) + " -> " + str(end))


def violence_detect_test():
    print("[TEST_CASE]: Violence")
    video_path = "videos/thor_cut.mp4"
    fps, img_dir = convert_mp4_to_jpg(video_path, IMAGE_PATH)
    list_img_dir = [img_dir]
    audio_path = convert_mp4_to_avi(video_path, AUDIO_PATH)
    list_audio_path = [audio_path]
    pred, elapsed_seconds = detect_violence(list_img_dir, list_audio_path, fps)
    post_predictions(pred, elapsed_seconds)


def pornography_detect_test():
    print("[TEST_CASE]: Pornography")
    video_path = "videos/pornography.mp4"
    pred, elapsed_seconds = detect_pornography(video_path)
    post_predictions(pred, elapsed_seconds)
    
def smoke_drink_detect_test():
    print("[TEST_CASE]: smoke drink")
    video_path = "videos/36.mp4"
    pred, elapsed_seconds = smoke_drink_detect.infer(video_path)
    post_predictions(pred, elapsed_seconds, threshold=0.2)
    

def summary_test():
    print("[TEST_CASE]: Text summary")
    ROOT_API = "http://183.81.35.24:32774"
    CONTENT_SCRIPT_UPDATE_STATUS_API = f"{ROOT_API}/Content_Script/Update_Status"
    CONTENT_SCRIPT_UPDATE_API = f"{ROOT_API}/Content_Script/Update"
    
    print(requests.put(f"{CONTENT_SCRIPT_UPDATE_STATUS_API}", params={
        "ids": 3,
        "status": "Done"
    }))


def extract_audio_test():
    print("[TEST_CASE]: Extract audio")
    video_path = "videos/thor_cut.mp4"
    output_path =  audio_extract(video_file=video_path)
    print(output_path)


def extract_frame_test():
    print("[TEST_CASE]: Extract images")
    video_path = "videos/thor_cut.mp4"
    path_save = frames_extract(video_file=video_path)
    print(len(os.listdir(path_save)))


def whisper_infer_test():
    print("[TEST_CASE]: Whisper inference")
    audio_file = "audios/thor_cut.mp3"
    result = whisper_infer(audio_path=audio_file, language="en")
    print(result)
    
    
if __name__ == "__main__":
    smoke_drink_detect_test()