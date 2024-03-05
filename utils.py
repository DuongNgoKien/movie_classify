import os
import cv2
import torch
import subprocess
import whisper
import logging

from moviepy.editor import *
from datetime import timedelta
from transformers import (
    pipelines,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    BartTokenizer
)


SUB_OUTPUT_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/subs"
IMAGE_OUTPUT_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/images"
AUDIO_OUTPUT_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/audios"


logger = logging.getLogger(__name__)


def audio_extract(video_file):
    """
    Convert video to audio using `ffmpeg` command with the help of subprocess
    module
    Args:
        video_file(str): path of video to extract audio
    """
    assert os.path.exists(video_file), f"Video path {video_file} not exists."
    filename, ext = os.path.splitext(video_file)
    filename = filename.split("/")[-1]

    if not os.path.exists(AUDIO_OUTPUT_PATH):
        os.mkdir(AUDIO_OUTPUT_PATH)

    output_path = os.path.join(AUDIO_OUTPUT_PATH, f"{filename}.wav")

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
    assert os.path.exists(video_file), f"Video path {video_file} not exists"
    filename, ext = os.path.splitext(video_file)
    filename = filename.split("/")[-1]
    if not os.path.exists(IMAGE_OUTPUT_PATH):
        os.makedirs(IMAGE_OUTPUT_PATH, exist_ok=True)
    path_save = os.path.join(IMAGE_OUTPUT_PATH, filename)
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
    milliseconds = int(milliseconds)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1000
    milliseconds -= seconds * 1000
    
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def write_sub_file(sub_file_path, text):
    with open(sub_file_path, "w", encoding="utf-8") as sub_f:
        sub_f.write(text)

if __name__ == "__main__":
    print(timestamp_format(1000000))
