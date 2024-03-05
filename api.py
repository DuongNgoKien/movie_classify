import os
import nltk
import requests
import time
import torch
import menovideo.menovideo as menoformer
import opennsfw2 as n2
import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from waitress import serve
from flask import Flask, request, Response

from utils import (
    audio_extract, 
    whisper_infer,
    translation,
    write_sub_file,
    post_predictions,
    update_progress_status,
    update_status,
)
from image_detect import (
    detect_horror,
    detect_pornography,
    detect_scene,
    detect_violence,
)
from inference_detect import sentiment_analysis_inference
from summary import summary_infer
from pipeline.audio_feature_extract import AudioFeatureExtractor
from pipeline.image_feature_extract import ImageFeatureExtractor
from pipeline import detect_scene, smoke_drunk_detect, politic_detect
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
VIOLECE_CHECKPOINT= "/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/ckpt/violence.pkl"
HORROR_CHECKPOINT = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/ckpt/horror.pkl"
POLITIC_CHECKPOINT = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/protest-detection-violence-estimation/model_best.pth.tar"
ROOT_API = "http://183.81.35.24:32774"
COMMAND_UPDATE_STATUS_API = f"{ROOT_API}/content_command/update_status"
CONTENT_UPDATE_STATUS_API = f"{ROOT_API}/content/update_status"

def analysis_process():
    """Process all content in wait list."""
    # get all content wait list
    content_list = requests.get(f"{ROOT_API}/content_command/get_wait")
    content_list = content_list.json()

    # processing
    for content in content_list:  
        # get content information
        content_id = content["content_id"]
        category_id = content["category_id"]
        command_id = content["id"]
        threshold = content["threshold"]
        content_info = requests.get(
            f"{ROOT_API}/content/get_by_id/{content_id}"
        ).json()
        language = content_info["language"]
        
        # check path and download if not exists.
        video_path = content_info["path"]

        #
        if not os.path.exists(video_path):
            try:
                video_url = content_info["url"]
                # title = content_info["title"].replace(" ", "_")
                sub_file_path = os.path.join(SUB_PATH, f"{content_id}.srt")
                print(f"Url: {video_url}")
                video_path = f"{VIDEO_PATH}/{content_id}.mp4"
                if not os.path.exists(video_path):
                    os.system(
                        f"wget -O {video_path} {video_url}"
                    )
            except:
                update_status(
                    type="command_status",
                    command_id=command_id,
                    status="Dowload error"
                )
        
        # check content_type
        content_type = content["command"]
        # selected_function = FUNCTION_MAP.get(content_type, classify_image)
        # selected_function()
        # try:
        if content_type == "speech":
            speech(
                command_id=command_id, 
                video_path=video_path,
                language=language,
                content_id=content_id,
                sub_file_path=sub_file_path
            )
        elif content_type == "classify_text":
            classify_text(
                content_id=content_id,
                category_id=category_id,
                command_id=command_id,
                language=language,
                threshold=threshold,
                sub_file_path=sub_file_path
            )
        elif content_type == "speech_and_classify_text":
            speech_and_classify_text(
                command_id=command_id,
                video_path=video_path,
                language=language,
                content_id=content_id,
                category_id=category_id,
                threshold=threshold,
                sub_file_path=sub_file_path
            )
            
        else:
            classify_image(
                video_path=video_path,
                command_id=command_id,
                content_id=content_id,
                category_id=category_id,
                threshold=threshold
                )
        # except Exception as e:
        #     update_progress_status(
        #         command_id=command_id,
        #         note=f"{e}",
        #         process_percent=100
        #     )
        #     update_status(
        #         type="command_status", command_id=command_id, status="Error"
        #     )
        #     update_status(
        #         type="content_status", command_id=content_id, status="Error"
        #     )
            
        
def classify_text(
    content_id,
    category_id,
    command_id,
    language,
    sub_file_path,
    threshold
):
    # update status
    update_status(
        type="command_status", command_id=command_id, status="Processing"
    )
    analysis_update_api = f"{ROOT_API}/content_category/create"
    
    # get script and process
    try:
        content_info = requests.get(
            f"{ROOT_API}/content_script/get_by_content_id/{content_id}"
        ).json()
        speech2text_result = content_info["script"]
        print(speech2text_result)

        if language != "en":
            translated_text = translation(
                speech2text_result,
                language,
                sub_file_path
            )
        else:
            write_sub_file(sub_file_path, speech2text_result)
        
        text_analysis_results = sentiment_analysis_inference(
            category_id,
            sub_file_path,
            threshold=threshold
        )
        for result in text_analysis_results:
            text_analysis_data = {
                "command_id": command_id,
                "content_id": content_id,
                "category_id": category_id,
                "timespan": result["time"],
                "content": result["text"],
                "detect_from": "text",
                "threshold": result["probability"]
            }
            requests.post(url=analysis_update_api, json=text_analysis_data)

        update_status(
            type="command_status", command_id=command_id, status="Done"
        )
        update_status(
            type="content_status", command_id=content_id, status="Done"
        )

    except Exception as e:
        update_progress_status(
            command_id=command_id,
            process_percent=100,
            note=f"{e}"
        )
        update_status(
            type="command_status", command_id=command_id, status="Error"
        )
        update_status(
            type="content_status", command_id=command_id, status="Error"
        )


def speech(
    command_id,
    video_path,
    language,
    content_id,
    sub_file_path
):
    try:
        content_script = requests.get(
            f"{ROOT_API}/Content_Script/Get_By_Content_Id/{content_id}"
        ).json()["script"]
        update_status(
            type="command_status", command_id=command_id, status="Done"
        )
        update_progress_status(
            command_id=command_id,
            process_percent=100,
            note="This content has been speech to text!"
        )
    except:
        update_status(
            type="command_status", command_id=command_id, status="Processing"
        )
        
        # convert mp4 to audio and update progress status
        audio_path = audio_extract(video_path)
        update_progress_status(
            command_id=command_id,
            note="Convert video to audio done. Convert audio to text now.",
            process_percent=10
        )
        
        # speech to text
        speech2text_result = whisper_infer(audio_path, language, sub_file_path)
        
        # update progress status
        update_progress_status(
            command_id=command_id,
            note="Convert video to audio done.",
            process_percent=100
        )
        
        # post to api
        speech2text_update_api = f"{ROOT_API}/content_script/create"
        requests.post(
            speech2text_update_api,
            json = {
                "content_id": content_id,
                "script": speech2text_result,
                "language": language,
                "user_id": 1
            }
        )
        
        # update command and content status
        update_status(type="command_status", command_id=command_id, status="done")
        update_status(type="content_status", command_id=command_id, status="done")


def speech_and_classify_text(
    command_id,
    video_path,
    language,
    content_id,
    category_id,
    sub_file_path,
    threshold
):  
    update_status(
        type="command_status", command_id=command_id, status="Processing"
    )
    
    try:
        script_api_respone = requests.get(
            f"{ROOT_API}/Content_Script/Get_By_Content_Id/{content_id}"
        ).json()
        speech2text_result = script_api_respone["script"]
        
        if not os.path.exists(sub_file_path):
            write_sub_file(sub_file_path, speech2text_result)
            
        update_progress_status(
            command_id=command_id,
            note="Get transcript done. Analysis text now.",
            process_percent=30
        )
    except:
        # Convert mp4 to audio and update progress status
        audio_path = audio_extract(video_path)
        update_progress_status(
            command_id=command_id,
            note="Convert video to audio done. Convert audio to text now.",
            process_percent=10
        )
        # speech to text
        speech2text_result = whisper_infer(
            audio_path,
            language,
            sub_file_path
        )
        # update progress status
        update_progress_status(
            command_id=command_id,
            note="Speech to text done.",
            process_percent=70
        )
    
    # post speech to text result to api
    speech2text_update_api = f"{ROOT_API}/content_script/create"
    requests.post(
        speech2text_update_api,
        json = {
            "command_id": command_id,
            "content_id": content_id,
            "script": speech2text_result,
            "language": language,
            "user_id": 1
        }
    )

    # translate if language is not vietnamese
    if language != "en":
        speech2text_result = speech2text_result.split("\n")
        speech2text_result, sub_file_path = translation(
            speech2text_result,
            language=language,
            sub_file_path=sub_file_path
        )
    
    # Do text analysis with speech2text_result           
    text_analysis_results = sentiment_analysis_inference(
        category_id,
        sub_file_path,
        threshold,
    )

    # update progress status
    update_progress_status(
        command_id=command_id,
        note="Text analysis done.",
        process_percent=100
    )
    
    # post analysis results to api
    analysis_update_api = f"{ROOT_API}/content_category/create"
    try:
        for text_analysis_result in text_analysis_results:
            text_analysis_data = {
                "command_id": command_id,
                "content_id": content_id,
                "category_id": category_id,
                "timespan": text_analysis_result["time"],
                "content": text_analysis_result["text"],
                "detect_from": "text",
                "threshold": text_analysis_result["probability"]
            }
            requests.post(url=analysis_update_api, json=text_analysis_data)
    except:
        print("Text analysis failed with None")
        
    # update command and content status
    update_status(
        type="command_status", command_id=command_id, status="Done"
    )
    update_status(
        type="content_status", command_id=command_id, status="Done"
    )


def classify_image(video_path, command_id, content_id, category_id, threshold):

    update_status(
        type="command_status", command_id=command_id, status="Processing"
    )
    # process image analysis
    category_api = f"{ROOT_API}/content_category/create" 
    
    if category_id == 1 or category_id == 5:
        fps, list_img_dir = convert_mp4_to_jpg(video_path, IMAGE_PATH)
        list_img_dir = [list_img_dir]
        audio_path = convert_mp4_to_avi(video_path, AUDIO_PATH)
        audio_path = [audio_path]
        if category_id == 1:
            pred, elapsed_seconds = detect_violence(
                list_img_dir,
                audio_path,
                fps
            )
        else:
            pred, elapsed_seconds = detect_horror(
                list_img_dir,
                audio_path, 
                fps
            )
        post_predictions(
            pred=pred, 
            command_id=command_id,
            elapsed_seconds=elapsed_seconds,
            api=category_api,
            content_id=content_id,
            category_id=category_id,
            content='Bao luc',
            threshold=threshold
        )
        
    elif category_id == 4:
        pred, elapsed_seconds = detect_pornography(video_path)
        post_predictions(
            pred=pred, 
            command_id=command_id,
            elapsed_seconds=elapsed_seconds,
            api=category_api,
            content_id=content_id,
            category_id=category_id,
            content='Khieu dam',
            threshold=threshold
        )
        
    elif category_id == 3:
        pred, elapsed_seconds = smoke_drink_detect.infer(video_path=video_path)
        print(video_path)
        post_predictions(
            pred=pred, 
            command_id=command_id,
            elapsed_seconds=elapsed_seconds,
            api=category_api,
            content_id=content_id,
            category_id=category_id,
            content='Chat kich thich, gay nghien',
            threshold=threshold
        )
    
    elif category_id == 5:
        fps, list_img_dir = convert_mp4_to_jpg(video_path, IMAGE_PATH)
        list_img_dir = [list_img_dir]
        audio_path = convert_mp4_to_avi(video_path, AUDIO_PATH)
        audio_path = [audio_path]
        pred, elapsed_seconds = detect_horror(list_img_dir, audio_path, fps)
        post_predictions(pred, command_id, elapsed_seconds, category_api, content_id, category_id=category_id, content='Kinh di')
    
    elif category_id == 3:
        pred, elapsed_seconds = smoke_drink_detect.infer(video_path)
        post_predictions(pred, command_id, elapsed_seconds, category_api, content_id, category_id=category_id, content='Chat kich thich gay nghien')     
    
    elif category_id == 10:
        fps, list_img_dir = convert_mp4_to_jpg(video_path, IMAGE_PATH)
        pred, elapsed_seconds = politic_detect.infer(list_img_dir, POLITIC_CHECKPOINT)
        post_predictions(pred, command_id, elapsed_seconds, category_api, content_id, category_id=category_id, content='Chinh tri')
    
    update_status(
        type="command_status", command_id=command_id, status="Done"
    )
    update_status(
        type="content_status", command_id=command_id, status="Done"
    )




FUNCTION_MAP = {
    "speech": speech,
    "speech_and_classify_text": speech_and_classify_text,
    "classify_image": classify_image
}

       
if __name__ == "__main__":
    while True:
        analysis_process()