import os
import cv2
import torch
import subprocess
import whisper
import logging
import requests

from moviepy.editor import *
from datetime import timedelta
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


VIDEO_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/videos"
AUDIO_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/audios"
IMAGE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/images"
IMG_FEATURE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/features/img_features"
AUDIO_FEATURE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/features/audio_features"
SUB_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/subs"
VIOLECE_CHECKPOINT= "/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/ckpt/violence.pkl"
HORROR_CHECKPOINT = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/ckpt/horror.pkl"
ROOT_API = "http://183.81.35.24:32774"
COMMAND_UPDATE_STATUS_API = f"{ROOT_API}/content_command/update_status"
CONTENT_UPDATE_STATUS_API = f"{ROOT_API}/content/update_status"


logger = logging.getLogger(__name__)


def audio_extract(video_file):
    """
    Convert video to audio using `ffmpeg` command with the help of subprocess
    module
    Args:
        video_file(str): path of video to extract audio
    """
    print("[SPEECH_PROCESSING]: Extract audio file.")
    assert os.path.exists(video_file), f"Video path {video_file} not exists."
    filename, ext = os.path.splitext(video_file)
    filename = filename.split("/")[-1]

    if not os.path.exists(AUDIO_PATH):
        os.mkdir(AUDIO_PATH)

    output_path = os.path.join(AUDIO_PATH, f"{filename}.wav")

    if not os.path.exists(output_path):
        command = f"ffmpeg -i {video_file} -ar 16000 -ac 1 {output_path} -y"
        subprocess.call(command, shell=True)
    
    return output_path
    

def frames_extract(video_file, output_ext="jpg", save_every_frames=5):
    """
    Extract frames image from video
    Args:
        video_file (str): path to video to extract frames.
        output_ext (str): extension of frames. Defaults to "jpg".
        save_every_frames(int): save image every save_every_frames.
        Defaults to 5
    """
    print("[IMAGE_PROCESSING]: Extract video frame.")
    assert os.path.exists(video_file), f"Video path {video_file} not exists"
    filename, ext = os.path.splitext(video_file)
    filename = filename.split("/")[-1]
    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH, exist_ok=True)
    path_save = os.path.join(IMAGE_PATH, filename)
    os.makedirs(path_save, exist_ok=True)
    
    # read video
    video = cv2.VideoCapture(video_file)
    
    count = 0
    while True:
        # reading from frame
        success, frame = video.read()
        if success:
            name = str(count) + "." + output_ext
            if count % save_every_frames == 0:
                print(f"save frame {count}")
                cv2.imwrite(os.path.join(path_save, name), frame)
            count += 1
        else:
            break
    
    return path_save


def whisper_infer(audio_path, language="vi", sub_file_path=""):
    """Speech to text inference.

    Args:
        audio_path (str): Path to audio file.
        language (str): Language of audio [en | zh | vi]. Defaults to "en".
        sub_file_path (str or PathLike): Path to save sub file. Defaults to "".
    """
    print(f"[SPEECH_PROCESSING]: Start speech to text in audio path: {audio_path}")
    # Load model and compute output
    model = whisper.load_model("/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/weights/whisper/large-v2.pt")
    
    transcribe = model.transcribe(
        audio_path,
        verbose=True,
        language=language,
        fp16=True
    )
    
    result = ""
    segments = transcribe["segments"]
    
    # del model
    del model
    
    # create timestamp
    for segment in segments:
        start_time = (
            str(0) + str(timedelta(seconds=int(segment["start"]))) + ",000"
        )
        end_time = (
            str(0) + str(timedelta(seconds=int(segment["end"]))) + ",000"
        )
        text = segment["text"]
        text = text[1:] if text[0] == " " else text
        # segment_id = segment["id"] + 1
        segment = (
            f"{start_time} --> {end_time}\n{text}\n\n"
        )
        result += segment
        
    # write sub_file
    print(f"Sub file path: {sub_file_path}")
    write_sub_file(sub_file_path, result) 
            
    return result


def translation(text, language="vietnamese", sub_file_path=""):
    """Translate text to english.
    
    Args:
        text (str|list): input text to translate (should be Vietnamese text or
        Chinese text)
        language (str): input text language [vi | zh]. Defaults to vi.
        sub_file_path (str or PathLike): path to save translate sub.
        Defaults to "".
    """
    if language == "vi":
        # define device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # define tokenizer and translate model
        tokenizer = AutoTokenizer.from_pretrained(
            "weights/vietnamese_to_english",
            src_lang="vi_VN",
            local_files_only=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "weights/vietnamese_to_english",
            local_files_only=True
        ).to(device)
        
        # translate vietnames to english text
        if isinstance(text, str):
            input_ids = tokenizer(
                text,
                padding=True,
                return_tensors="pt"
            ).to(device)
            output_ids = model.generate(
                **input_ids,
                decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
                num_return_sequences=1,
                num_beams=5,
                early_stopping=True
            )
            
            # decode ids to text
            en_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # write sub_file for text analysis
            write_sub_file(sub_file_path, en_texts)
            
            # delete tokenizer and model and
            del tokenizer
            del model
                
            return en_texts
        
        elif isinstance(text, list):
            english_texts = []
            for t in text:
                input_ids = tokenizer(
                    t,
                    padding=True,
                    return_tensors="pt"
                ).to(device)
                output_ids = model.generate(
                    **input_ids,
                    decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
                    num_return_sequences=1,
                    num_beams=5,
                    early_stopping=True
                )
                
                # decode ids to text
                en_texts = tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True
                )
                english_texts.append(en_texts[0])
            
            # delete tokenizer and model and
            del tokenizer
            del model

            results = "\n".join(english_texts)
            
            # write sub_file for text analysis
            write_sub_file(sub_file_path, results)
            
            return results
                
    elif language == "zh":
        # define the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # define tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            "weights/chinese_to_english",
            local_files_only=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "weights/chinese_to_english",
            local_files_only=True
        ).to(device)
        
        if isinstance(text, str):
            # encode input ids
            input_ids = tokenizer.prepare_seq2seq_batch(
                [text],
                return_tensors="pt"
            ).to(device)
            
            # translation
            output_ids = model.generate(**input_ids)
            
            # decode the output_ids to text
            english_texts = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True
            )
            
            # delete tokenizer and model
            del tokenizer
            del model
            
            # write sub_file for text analysis
            write_sub_file(sub_file_path, english_texts)
            
            return english_texts
        
        else:
            english_texts = []
            for t in text:
                # encode input ids
                input_ids = tokenizer.prepare_seq2seq_batch(
                    [text],
                    return_tensors="pt"
                ).to(device)
                
                # translation
                output_ids = model.generate(**input_ids)
                
                # decode the output_ids to text
                en_texts = tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True
                )
                english_texts.append(en_texts)
            
            # delete tokenizer and model
            del tokenizer
            del model
            
            results = "\n".join(english_texts)
            
            # write sub_file for text_analysis
            write_sub_file(sub_file_path, results)
            
            return results
    else:
        raise NotImplementedError(f"Language {language} not supported yet!")


def timestamp_format(milliseconds):
    """Format time from miniseconds to hours:minutes:seconds,milliseconds

    Args:
        milliseconds (int): time in milliseconds.

    Returns:
        string: time in format
    """
    milliseconds = int(milliseconds)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1000
    milliseconds -= seconds * 1000
    
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def write_sub_file(sub_file_path, text):
    """Write transcript to sub file.

    Args:
        sub_file_path (str or PathLike): Path to save transcript file.
        text (str): transcript.
    """
    with open(sub_file_path, "w", encoding="utf-8") as sub_f:
        sub_f.write(text)


def post_predictions(
    pred,
    command_id,
    elapsed_seconds,
    api,
    content_id,
    category_id,
    content,
    threshold
):
    """Post content detected from images.

    Args:
        pred (np.array): model predictions.
        command_id (int): command id.
        elapsed_seconds (str): Time of content detected.
        api (str): api to post data.
        content_id (int): content id.
        category_id (int): category id.
        content (str): content (Kinh di | Khieu dam ...)
        threshold (int): model predictions.
    """
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
                avg_prob = int(sum_prob / count * 100)
                if avg_prob >= threshold:
                    json_data = {
                        "command_id": command_id,
                        "category_id": category_id, 
                        'content_id': content_id, 
                        'timespan': start + " --> " + end,
                        'content': content, 
                        'detect_from': 'image',
                        'threshold': avg_prob
                    }
                    requests.post(api, json = json_data)
                count = 0
                sum_prob = 0
                
    if count != 0:
        end = end_seconds[i]
        end = timestamp_format(end * 1000)
        avg_prob = int(sum_prob / count * 100)
        if avg_prob >= threshold:
            json_data = {
                'command_id': command_id,
                'category_id': category_id,
                'content_id': content_id, 
                'timespan': start + " --> " + end,
                'content': content, 
                'detect_from': 'image',
                'threshold': avg_prob
            }
            requests.post(api, json = json_data)

def update_progress_status(command_id, process_percent, note):
    requests.put(
        f"{ROOT_API}/content_command/update_progress",
        params={
            "id": command_id,
            "new_note": note,
            "progress": process_percent
        }
    ) 
    


def update_status(type, command_id, status):
    api = (
        COMMAND_UPDATE_STATUS_API if type == "command_status"
        else CONTENT_UPDATE_STATUS_API
    )
    api = f"{api}?id={command_id}&status={status}"
    requests.put(api)


if __name__ == "__main__":
    print(timestamp_format(1000000))
