import torch
import os
import pandas as pd
import pysrt
import json
import random
import re
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from tqdm import tqdm
def extract_subtitle_info(text: str) -> list:
    """
    Extract the time and text from the subtitle text.
    
    Args:
    text (str): The original subtitle text
    
    Returns:
    list: A list of tuples containing the time and text for each subtitle line
    """
    subtitle_info = re.findall(r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\n(.+)', text)
    return subtitle_info


def sentiment_analysis_inference(category_id, sub_file_path, threshold):
    """Text sentiment analysis inferences.

    Args:
        sub_file_path (str or like_path): path to sub file.
    Category id :
    1. Action
    4. Sexism
    5. Horror
    6. Racist
    10. Political
    11. Religious
    12. Toxic
    """
    
    # get sub
    if category_id == 3 or category_id == 5:
        return []
    else:
        subs = pysrt.open(sub_file_path)
        # subtitle_info = extract_subtitle_info(sub_script)
        start_time_seconds = subs[0].start.hours * 3600 + subs[0].start.minutes * 60 + subs[0].start.seconds
        hours = subs[-1].end.hours
        minutes = subs[-1].end.minutes
        seconds = subs[-1].end.seconds
        total_duration_seconds = hours * 3600 + minutes * 60 + seconds

        slices_srt = []
        # Slice the text in 30-second intervals and print the start and end times
        while start_time_seconds < total_duration_seconds:
            end_time_seconds = start_time_seconds + 90
            # print(f"Start time: {start_time_seconds} seconds, End time: {end_time_seconds} seconds")
            part = subs.slice(starts_after={'seconds': start_time_seconds}, ends_before={'seconds': end_time_seconds})
            # print(part.text)
            start_time_seconds = end_time_seconds
            if (re.sub("♪♪", "",part.text).replace("\n","") != ''):
                slices_srt.append(part)

        slices_srt = [s for s in slices_srt if s]
        # initial tokenizer and model
        if category_id == 6 or category_id == 12:
            path = "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-6-12-pytorch"
            model = AutoModelForSequenceClassification.from_pretrained(path, from_tf=True).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(path)
        else:
            path_to_model = "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-{}-pytorch".format(category_id)
            tokenizer = AutoTokenizer.from_pretrained(path_to_model)
            model = AutoModelForSequenceClassification.from_pretrained(path_to_model).to("cuda")

        # get sentiment-analysis results
        results = []
        for sub in tqdm(slices_srt, desc="Predicting", colour="cyan"):
            inputs = tokenizer(
                sub.text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to("cuda")
            outputs = model(**inputs)
            probs = outputs[0].softmax(1)
            pred_label_idx = probs.argmax()
            pred_label = model.config.id2label[pred_label_idx.item()]
            probability = int(probs[0][pred_label_idx.item()].item()*100)
            result = {
                'pred_label_idx': pred_label_idx.item(),
                'pred_label': pred_label,
                'text': sub.text.replace("\n", " "),
                # 'time': "{} --> {}".format(sub[0], sub[1]),
                'time': "{}:{:02d}:{:02d},{:02d} --> {}:{:02d}:{:02d},{:02d}".format(
                    sub[0].start.hours, sub[0].start.minutes, sub[0].start.seconds, sub[0].start.milliseconds,
                    sub[-1].end.hours, sub[-1].end.minutes, sub[-1].end.seconds, sub[-1].end.milliseconds),
                'probability': probability
            }
            # if the probability is greater than threshold and the label is True, add the result to the list
            if category_id == 6 and float(probability) >= threshold and pred_label_idx.item() == 2:
                results.append(result) 
            elif (float(probability) >= threshold and pred_label_idx.item() == 1):
                results.append(result)
        # delete the model and tokenizer to free GPU
        del model
        del tokenizer

        return results


if __name__ == '__main__':
    pass
