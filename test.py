import os
import numpy as np
import opennsfw2 as n2

import numpy as np
import torch

from utils import *
from decord import VideoReader, cpu
from transformers import (
    VideoMAEFeatureExtractor,
    VideoMAEForVideoClassification
)
from pipeline import smoke_drink_detect
from pipeline.audio_feature_extract import AudioFeatureExtractor
from pipeline.image_feature_extract import ImageFeatureExtractor
from pipeline.detect_scene import infer
from pytorchi3d.mp4_to_jpg import convert_mp4_to_jpg
from torchvggish.torchvggish.mp4_to_wav import convert_mp4_to_avi


AUDIO_PATH = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/audios"
IMAGE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/images"
IMG_FEATURE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/features/img_features"
AUDIO_FEATURE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/features/audio_features"
VIOLENCE_CHECKPOINT= "/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/ckpt/violence.pkl"
HORROR_CHECKPOINT = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/ckpt/horror.pkl"


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


def smoke_drink_test():        
    pred, elapsed_time = smoke_drink_detect.infer(video_path = '/home/www/data/data/saigonmusic/Dev_AI/kiendn/Smoking for the aesthetics.mp4')
    post_predictions(pred, elapsed_time, threshold=0.4)


def violence_detect_test():
    pass


def summary_test():
    ROOT_API = "http://183.81.35.24:32774"
    CONTENT_SCRIPT_UPDATE_STATUS_API = f"{ROOT_API}/Content_Script/Update_Status"
    CONTENT_SCRIPT_UPDATE_API = f"{ROOT_API}/Content_Script/Update"
    
    print(requests.put(f"{CONTENT_SCRIPT_UPDATE_STATUS_API}", params={
        "ids": 3,
        "status": "Done"
    }))


def extract_audio_test():
    video_path = "videos/vietnamese.mp4"
    output_path =  audio_extract(video_file=video_path)
    print(output_path)


def extract_frame_test():
    video_path = "videos/chinese.mp4"
    path_save = frames_extract(video_file=video_path)
    print(len(os.listdir(path_save)))


def whisper_infer_test():
    audio_file = "audio/english.mp3"
    result = whisper_infer(audio_path=audio_file, language="en")
    print(result)
    
    
if __name__ == "__main__":
    import requests
    data = {
        'command_id': 571, 
        'content_id': 23, 
        'category_id': 11, 
        'timespan': '0:01:31,00 --> 0:02:59,00', 
        'content': "Ahihihi", 
        'detect_from': 'text', 
        'threshold': 80
    }
    print(requests.post(
        url="http://183.81.35.24:32774/content_category/create",
        json=data
    ))