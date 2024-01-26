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
    """
    # get sub
    # subs = pysrt.open(sub_file_path)
    subtitle_info = extract_subtitle_info(sub_script)
    # initial tokenizer and model
    if category_id == 1 or category_id == 2:  # Action detect model
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-action-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-action-pytorch"
        )

    elif category_id == 4:  # Sexism detect model
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-sexism-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-sexism-pytorch"
        ).to("cuda")

    elif category_id == 5:  # Horror detect model
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-horror-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-horror-pytorch"
        ).to("cuda")

    elif category_id == 6:  # Racism detect model
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-racism-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-racism-pytorch"
        ).to("cuda")

    elif category_id == 7 or category_id == 8 or category_id == 9 or category_id == 12:  # Toxic detect model
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-toxic-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-toxic-pytorch"
        ).to("cuda")

    elif category_id == 10:  # Political detect model
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-political-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-political-pytorch"
        ).to("cuda")

    elif category_id == 11:  # Religious detect model
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-religious-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-religious-pytorch"
        ).to("cuda")

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
