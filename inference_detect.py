import torch
import os
import pandas as pd
import pysrt
import json
import random
import re
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer
)

# category_id = random.randint(1, 12)
threshold = 0.7
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
def sentiment_analysis_inference(category_id, sub_script):
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
    
    # subs = pysrt.open(sub_file_path)
    subtitle_info = extract_subtitle_info(sub_script)
    
    # initial tokenizer and model
    path_to_model = "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-{}-pytorch".format(category_id)
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = BertForSequenceClassification.from_pretrained(path_to_model).to("cuda")

    # get sentiment-analysis results
    results = []
    for sub in subtitle_info:
        inputs = tokenizer(
            sub[2],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to("cuda")
        outputs = model(**inputs)
        probs = outputs[0].softmax(1)
        pred_label_idx = probs.argmax()
        pred_label = model.config.id2label[pred_label_idx.item()]
        probability = "{:.2f}%".format(probs[0][pred_label_idx.item()].item()*100)
        result = {
            'pred_label_idx': pred_label_idx.item(),
            'pred_label': pred_label,
            'text': sub[2],
            # 'time': "{}:{:02d}:{:02d},{:02d} --> {}:{:02d}:{:02d},{:02d}".format(
            #     sub.start.hours, sub.start.minutes, sub.start.seconds, sub.start.milliseconds,
            #     sub.end.hours, sub.end.minutes, sub.end.seconds, sub.end.milliseconds),
            'time': "{} --> {}".format(sub[0], sub[1]),
            'probability': probability
        }
        # if the probability is greater than threshold and the label is True, add the result to the list
        if float(probability.strip('%')) >= threshold and pred_label_idx.item() == 1:
            results.append(result)
    # delete the model and tokenizer to free GPU
    del model
    del tokenizer

    return results


if __name__ == '__main__':
    pass
