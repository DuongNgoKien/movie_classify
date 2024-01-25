import os
import nltk
import requests
import time
import torch
import menovideo.menovideo as menoformer
import detect_violence
import opennsfw2 as n2
import numpy as np

from transformers import pipeline,AutoTokenizer,AutoModelForSeq2SeqLM
from waitress import serve
from flask import Flask, request, Response

from utils import audio_extract, whisper_infer, translation, timestamp_format
from inference_detect import sentiment_analysis_inference
from summary import summary_infer
from pipeline.audio_feature_extract import AudioFeatureExtractor
from pipeline.image_feature_extract import ImageFeatureExtractor
from pipeline.violence_detect import infer
from pytorchi3d.mp4_to_jpg import convert_mp4_to_jpg
from torchvggish.torchvggish.mp4_to_wav import convert_mp4_to_avi

app = Flask(__name__)

LANGUAGES = {
    "vietnamese": "vi",
    "chinese": "zh",
    "english": "en"
}


VIDEO_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/videos"
AUDIO_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/audios"
IMAGE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/images"
IMG_FEATURE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/features/img_features"
AUDIO_FEATURE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/features/audio_features"
SUB_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/subs"
ROOT_API = "http://183.81.35.24:32774"
COMMAND_UPDATE_STATUS_API = f"{ROOT_API}/content_command/update_status"
CONTENT_UPDATE_STATUS_API = f"{ROOT_API}/content/update_status"


def analysis_process():
    """Process all content in wait list."""
    # get all content wait list
    content_list = requests.get(f"{ROOT_API}/content_command/get_wait")
    content_list = content_list.json()
    content_list = [
        content for content in content_list if content["status"] == "wait"
    ]
    # processing
    for content in content_list:  
        # get content information
        content_id = content["content_id"]
        category_id = content["category_id"]
        video_id = content["id"]
        # language = LANGUAGES[content["language"]]
        language = "en"
        
        content_info = requests.get(
            f"{ROOT_API}/content/get_by_id/{content_id}"
        ).json()
        # check path and download if not exists.
        video_path = content_info["path"]
        print(f"Video_path: {video_path}")
        if not os.path.exists(video_path):
            try:
                video_url = content_info["url"]
                title = content_info["title"].replace(" ", "_")
                os.system(f"wget -O {title}.mp4 {video_url} -P {VIDEO_PATH}")
                video_path = f"{VIDEO_PATH}/{title}.mp4"
            except:
                update_status(
                    type="command_status",
                    video_id=video_id,
                    status="Dowload error"
                )
        
        # check content_type
        content_type = content["command"]
        
        if content_type == "text":
            # process audio analysis
            update_status(
                type="command_status", video_id=video_id, status="Processing"
            )
            
            audio_path = audio_extract(video_path)
            sub_file_path = os.path.join(SUB_PATH, language, f"{video_id}.srt")
            speech2text_result = whisper_infer(
                audio_path,
                language,
                sub_file_path
            )
            speech2text_update_api = f"{ROOT_API}/content_script/create"
            requests.post(
                speech2text_update_api,
                json = {
                    "content_id": content_id,
                    "script": speech2text_result,
                    "language": "en",
                    "user_id": 1
                }
            )
            
            # translate if language is not vietnamese
            if language != "en":
                sub_file_path = os.path.join(
                    SUB_PATH,
                    "en",
                    f"{video_id}_translate.srt"
                )
                speech2text_result = speech2text_result.split("\n")
                speech2text_result = translation(
                    speech2text_result,
                    language=language,
                    sub_file_path=sub_file_path
                )
            
            # Do text analysis with speech2text_result           
            text_analysis_results = sentiment_analysis_inference(
                category_id,
                sub_file_path
            )
  
            analysis_update_api = f"{ROOT_API}/category_content/create"
            for text_analysis_result in text_analysis_results:
                text_analysis_data = {
                    "content_id": content_id,
                    "category_id": category_id,
                    "timespan": text_analysis_result["time"],
                    "content": text_analysis_result["text"],
                    "detect_from": "text",
                    "analysis_threshold": text_analysis_result["probability"]
                }
                requests.post(url=analysis_update_api, json=text_analysis_data)
            
            update_status(
                type="command_status", video_id=video_id, status="Done"
            )
            update_status(
                type="content_status", video_id=video_id, status="Done"
            )
            
        else:
            update_status(
                type="command_status", video_id=video_id, status="Processing"
            )
            # process image analysis
            category_api = f"{ROOT_API}/Category_Content/create"
            fps, img_dir = convert_mp4_to_jpg(video_path, IMAGE_PATH)
            audio_path = convert_mp4_to_avi(video_path, AUDIO_PATH)
            
            pred, elapsed_seconds = detect_violence(img_dir, audio_path, fps)
            post_predictions(pred, elapsed_seconds, category_api, video_id, content_id, category_id='2', content='Bao luc')
            
            pred, elapsed_seconds = detect_pornography(video_path)
            post_predictions(pred, elapsed_seconds, category_api, video_id, content_id, category_id='4', content='Khieu dam')
            
            update_status(
                type="command_status", video_id=video_id, status="Done"
            )
            update_status(
                type="content_status", video_id=video_id, status="Done"
            )



def update_status(type, video_id, status):
    api = (
        COMMAND_UPDATE_STATUS_API if type == "command_status"
        else CONTENT_UPDATE_STATUS_API
    )
    api = f"{api}?id={video_id}&status={status}"
    print(api)
    requests.put(api)


def detect_violence(img_dir, audio_path, fps):
    #Image Feature Extraction
    img_feature_extractor = ImageFeatureExtractor(root=img_dir) 
    rgb_feature_files, elapsed_frames = img_feature_extractor.extract_image_features()
    #Audio Feature Extraction
    audio_feature_extractor = AudioFeatureExtractor(
        audio_path,
        feature_save_path=AUDIO_FEATURE_PATH
    )
    audio_feature_files = audio_feature_extractor.extract_audio_features()
    pred = infer(rgb_feature_files, audio_feature_files)
    elapsed_seconds = np.array(elapsed_frames)/fps
    
    return pred, elapsed_seconds

 
def detect_pornography(video_path):
    elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(video_path)
    elapsed_seconds = np.array(elapsed_seconds)
    nsfw_probabilities = np.array(nsfw_probabilities)
    
    return nsfw_probabilities, elapsed_seconds


def post_predictions(
    pred,
    elapsed_seconds,
    api,
    video_id,
    content_id,
    category_id,
    content,
    threshold=0.7
):
    sum_prob, count, start, end = 0, 0, 0, 0
    
    if elapsed_seconds.ndim == 2:
        start_seconds = elapsed_seconds[:, 0]
        end_seconds = elapsed_seconds[:, 1]
    else:
        start_seconds = elapsed_seconds
        end_seconds = elapsed_seconds
        
    for i in range(pred.shape[0]):
        if pred[i] >= 0.5:
            if count == 0:
                start = start_seconds[i]
                start = timestamp_format(start * 1000)
            count += 1
            sum_prob += pred[i]
        else:
            if count !=0:
                end = end_seconds[i-1]
                end = timestamp_format(end * 1000)
                avg_prob = sum_prob/count
                if avg_prob >= threshold:
                    json_data = {
                        "category_id": category_id, 
                        'content_id': content_id, 
                        'timespan': start + " --> " + end,
                        'content': content, 
                        'detect_from': 'image',
                        'analysis_threshold': avg_prob
                    }
                    print(json_data)
                    requests.post(api, json = json_data)
                count = 0
                sum_prob = 0
    if count != 0:
        end = end_seconds[i]
        end = timestamp_format(end * 1000)
        avg_prob = sum_prob/count
        if avg_prob >= threshold:
            json_data = {
                'category_id': category_id,
                'content_id': content_id, 
                'timespan': start + " --> " + end,
                'content': content, 
                'detect_from': 'image',
                'analysis_threshold': avg_prob
            }
            print(json_data)
            requests.post(api, json = json_data)
            
            
if __name__ == "__main__":
    analysis_process()