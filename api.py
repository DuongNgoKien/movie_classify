import json
import os

from waitress import serve
from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
from utils import audio_extract, whisper_infer


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
    

serve(app, host="0.0.0.0", port=6001, threads=15)