import os
import nltk
import time
import torch
import menovideo.menovideo as menoformer
import detect_violence
import opennsfw2 as n2

from transformers import pipeline,AutoTokenizer,AutoModelForSeq2SeqLM
from waitress import serve
from flask import Flask, request, Response

from utils import audio_extract, whisper_infer
from inference_detect import *
from inference_detect import sentiment_analysis_inference
from summary import summary_infer

app = Flask(__name__)

LANGUAGES = {
    "vietnamese": "vi",
    "chinese": "zh",
    "english": "en"
}

SUB_OUTPUT = "http://210.245.90.204:13000/hg_project_effect/movie_classify/subs"


@app.route("/api/speech2text", methods=["POST", "GET"])
def speech2text():
    video_path = request.args.get("video_path")

    # get language id
    language = LANGUAGES[request.args.get("language")]
    
    if video_path is not None:
        # get audio
        audio_path = audio_extract(video_path)
        
        srt_file_name = audio_path.split("/")[-1].replace("mp3", "srt")
        srt_file = os.path.join(SUB_OUTPUT, srt_file_name)
        
        # speech to text
        sub_path = whisper_infer(
            audio_path=audio_path,
            language=language,
            srt_file=srt_file_name
        )
        
        return {"sub_path": srt_file, "status": 200, "user_id": 1}
    else:
        message = "No video path"
        return Response((message), status=400)


@app.route("/api/summary", methods=["POST","GET"])
def summary():
    # get file sub path
    sub_file_path = request.args.get("sub_file_path")
    if sub_file_path is not None and os.path.exists(sub_file_path):
        # summary file process
        summary_path = summary_infer(sub_file_path)
        
        return {"summary_file_path": summary_path, "status": 200, "user_id": 1}
    else:
        message = "No file sub path"
        return Response((message), status=400) 

      
@app.route("/api/detection_text", methods=["POST", "GET"])
def detection_text():
     # get file sub path
    sub_file_path = request.args.get("sub_file_path")
    if sub_file_path is not None and os.path.exists(sub_file_path):
        # sentiment-analysis detection
        detect_results = sentiment_analysis_inference(sub_file_path)
        
        return {"detect_results": detect_results, "status": 200, "user_id": 1}
    else:
        message = "No file sub path"
        return Response((message), status=400)

      
@app.route("/api/detetcion_violence", methods=["POST","GET"])
def detect_violence():
    model = menoformer.DeVTr()
    detect_violence.resume_checkpoint(model, '/home/www/data/data/saigonmusic/Dev_AI/kiendn/detect_violence/checkpoint/checkpoint-epoch10.pth')
    video_path = request.args.get("video_path")
    videos, intervals = detect_violence.capture(video_path,timesep=40,rgb=3,h=200,w=480,frame_interval=3)
    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    predictions = []
    for index in range(len(intervals)):
        sample = torch.unsqueeze(videos[index],0).to(device)
        output = model(sample)
        predictions.append({
            "start":intervals[index][0], 
            "end":intervals[index][1],
            "prediction":torch.sigmoid(output)[0].item(), 
        })
    return json.dumps(predictions)

@app.route("/api/detetcion_pornography", methods=["POST","GET"])
def detect_pornography():
    video_path = request.args.get("video_path")
    elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(video_path)
    sum_nsfw = 0
    count = 0
    start = 0
    end = 0
    predictions = []
    for i in range(len(nsfw_probabilities)):
        if nsfw_probabilities[i] >= 0.5:
            if count == 0:
                start = elapsed_seconds[i]
            count += 1
            sum_nsfw += nsfw_probabilities[i]
        else:
            if count !=0:
                end = elapsed_seconds[i-1]
                predictions.append({
                    'start':start,
                    'end':end,
                    'avg_nsfw_score': sum_nsfw/count
                })
                count = 0
                sum_nsfw = 0
    if count != 0:
        end = elapsed_seconds[i]
        predictions.append({
            'start':start,
            'end':end,
            'avg_nsfw_score': sum_nsfw/count
        })
    return json.dumps(predictions)
    
    
serve(app, host="0.0.0.0", port=6001, threads=15)