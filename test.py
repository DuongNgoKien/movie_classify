import os

from utils import *


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
    srt_file = audio_file.split("/")[-1].replace("mp3", "srt")
    srt_file = whisper_infer(audio_path=audio_file, language="en", srt_file=srt_file)

if __name__ == "__main__":
    # extract_audio_test()
    whisper_infer_test()