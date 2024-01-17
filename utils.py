import os
import cv2
import torch
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


SUB_OUTPUT_PATH = "/home/www/data/data/saigonmusic/hg_project_effect/movie_classify/subs"
IMAGE_OUTPUT_PATH = "/home/www/data/data/saigonmusic/hg_project_effect/movie_classify/images"
AUDIO_OUTPUT_PATH = "/home/www/data/data/saigonmusic/hg_project_effect/movie_classify/audios"


logger = logging.getLogger(__name__)


def audio_extract(video_file, output_ext="mp3"):
    """
    Convert video to audio using `ffmpeg` command with the help of subprocess
    module
    Args:
        video_file(str): path of video to extract audio
        output_ext(str): extension of output audio ([mp3 | wav])
    """
    assert os.path.exists(video_file), f"Video path {video_file} not exists."
    filename, ext = os.path.splitext(video_file)
    filename = filename.split("/")[-1]

    if not os.path.exists(AUDIO_OUTPUT_PATH):
        os.mkdir(AUDIO_OUTPUT_PATH)

    output_path = os.path.join(AUDIO_OUTPUT_PATH, f"{filename}.{output_ext}")

    # read video
    video = VideoFileClip(video_file)

    # convert to mp3 file
    audio = video.audio
    
    # export audio
    audio.write_audiofile(output_path)
    
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


def whisper_infer(audio_path, language="vi"):
    """Speech to text inference.

    Args:
        audio_path (str): Path to audio file.
        language (str): Language of audio [en | zh | vi]. Defaults to "en".
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
        segment_id = segment["id"] + 1
        segment = (
            f"{start_time} --> {end_time}: {text}\n"
        )
        result += segment
        
    return result


def translation(text, language="vietnamese"):
    """Translate text to english.
    
    Args:
        text (str|list): input text to translate (should be Vietnamese text or
        Chinese text)
        language (str): input text language [vi | zh]. Defaults to vi.
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
                english_texts.append(en_texts)
            
            # delete tokenizer and model and
            del tokenizer
            del model
            
            return "\n".join(english_texts)
                
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

            return "\n".join(english_texts)
    else:
        raise NotImplementedError(f"Language {language} not supported yet!")
    

if __name__ == "__main__":
    pass
