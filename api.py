import os
import nltk
import requests
import time
import torch
import menovideo.menovideo as menoformer
import detect_violence
import opennsfw2 as n2

from transformers import pipeline,AutoTokenizer,AutoModelForSeq2SeqLM
from waitress import serve
from flask import Flask, request, Response

from utils import audio_extract, whisper_infer, translation
from inference_detect import *
from inference_detect import sentiment_analysis_inference
from summary import summary_infer

app = Flask(__name__)

LANGUAGES = {
    "vietnamese": "vi",
    "chinese": "zh",
    "english": "en"
}


VIDEO_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/videos"
ROOT_API = "http://183.81.35.24:32774"
COMMAND_UPDATE_STATUS_API = f"{ROOT_API}/content_commands/update_status"
CONTENT_UPDATE_STATUS_API = f"{ROOT_API}/content/update_status"


def analysis_process():
    """Process all content in wait list."""
    # get all content wait list
    content_list = requests.get(f"{ROOT_API}/content_commands/get_wait")
    content_list = [
        content for content in content_list if content["status"] == "wait"
    ]
    # processing
    for content in content_list:  
        # get content information
        content_id = content["content_id"]
        video_id = content["id"]
        language = LANGUAGES[content["language"]]
        
        content_info = requests.get(
            f"{ROOT_API}/content/get_by_id/{content_id}"
        )
        
        # check path and download if not exists.
        video_path = content_info["path"]
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
        
        if content_type == "audio":
            # process audio analysis
            update_status(
                type="command_status", video_id=video_id, status="Processing"
            )
            
            audio_path = audio_extract(video_path)
            speech2text_result = whisper_infer(audio_path, language)
            speech2text_update_api = f"{ROOT_API}/content_script/create"
            requests.post(
                speech2text_update_api,
                {
                    "content_id": content_id,
                    "script": speech2text_result
                }
            )
            
            # translate if language is not vietnamese
            if language != "vi":
                speech2text_result = speech2text_result.split("\n")
                speech2text_result = translation(
                    speech2text_result,
                    language=language
                )
            
            # Do summary and text analysis with speech2text_result
            
        else:
            # process image analysis
            pass


def update_status(type, video_id, status):
    api = (
        COMMAND_UPDATE_STATUS_API if type == "command_status"
        else CONTENT_UPDATE_STATUS_API
    )
    requests.post(api, {"id": video_id, "status": status})

# def text_analysis():
#     content_list = requests.get(f"{ROOT_API}/content_commands/get_wait")
#     for content in content_list:
#         # process speech2text for audio content type
#         content_id = content["content_id"]
#         content_command = content["command"]
#         if content_command == "audio":
#             # update status processing for content
#             requests.post(
#                 COMMAND_UPDATE_STATUS_API,
#                 {
#                     "id": content_id,
#                     "status": "Processing"
#                 }
#             )
        

# def summary():
#     # get file sub path
#     sub_file_path = request.args.get("sub_file_path")
#     if sub_file_path is not None and os.path.exists(sub_file_path):
#         # summary file process
#         summary_path = summary_infer(sub_file_path)
        
#         return {"summary_file_path": summary_path, "status": 200, "user_id": 1}
#     else:
#         message = "No file sub path"
#         return Response((message), status=400) 


# def detection_text():
#      # get file sub path
#     sub_file_path = request.args.get("sub_file_path")
#     if sub_file_path is not None and os.path.exists(sub_file_path):
#         # sentiment-analysis detection
#         detect_results = sentiment_analysis_inference(sub_file_path)
        
#         return {"detect_results": detect_results, "status": 200, "user_id": 1}
#     else:
#         message = "No file sub path"
#         return Response((message), status=400)

      
# def detect_violence():
#     model = menoformer.DeVTr()
#     detect_violence.resume_checkpoint(model, '/home/www/data/data/saigonmusic/Dev_AI/kiendn/detect_violence/checkpoint/checkpoint-epoch10.pth')
#     video_path = request.args.get("video_path")
#     videos, intervals = detect_violence.capture(video_path,timesep=40,rgb=3,h=200,w=480,frame_interval=3)
#     device =  'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)
#     predictions = []
#     for index in range(len(intervals)):
#         sample = torch.unsqueeze(videos[index],0).to(device)
#         output = model(sample)
#         predictions.append({
#             "start":intervals[index][0], 
#             "end":intervals[index][1],
#             "prediction":torch.sigmoid(output)[0].item(), 
#         })
#     return json.dumps(predictions)

# @app.route("/api/detetcion_pornography", methods=["POST","GET"])
# def detect_pornography():
#     video_path = request.args.get("video_path")
#     elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(video_path)
#     sum_nsfw = 0
#     count = 0
#     start = 0
#     end = 0
#     predictions = []
#     for i in range(len(nsfw_probabilities)):
#         if nsfw_probabilities[i] >= 0.5:
#             if count == 0:
#                 start = elapsed_seconds[i]
#             count += 1
#             sum_nsfw += nsfw_probabilities[i]
#         else:
#             if count !=0:
#                 end = elapsed_seconds[i-1]
#                 predictions.append({
#                     'start':start,
#                     'end':end,
#                     'avg_nsfw_score': sum_nsfw/count
#                 })
#                 count = 0
#                 sum_nsfw = 0
#     if count != 0:
#         end = elapsed_seconds[i]
#         predictions.append({
#             'start':start,
#             'end':end,
#             'avg_nsfw_score': sum_nsfw/count
#         })
#     return json.dumps(predictions)
    
    
if __name__ == "__main__":
    while True:
        analysis_process()
        time.sleep(30)
        