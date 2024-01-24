import torch
import os
import pandas as pd
import pysrt
import json
import random
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer
)


category_id = random.randint(1, 12)


def sentiment_analysis_inference(category_id, sub_file_path):

    """Text sentiment analysis inferences.

    Args:
        sub_file_path (str or like_path): path to sub file.
    """
    # get sub
    subs = pysrt.open(sub_file_path)

    # initial tokenizer and model
    if category_id == 5:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-horror-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-horror-pytorch"
        ).to("cuda")

    elif category_id == 10:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-political-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-political-pytorch"
        ).to("cuda")

    elif category_id == 11:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-religious-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-religious-pytorch"
        ).to("cuda")

    elif category_id == 9 or category_id == 8 or category_id == 7:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-toxic-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-toxic-pytorch"
        ).to("cuda")

    elif category_id == 6:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-racism-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-racism-pytorch"
        ).to("cuda")

    elif category_id == 4:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-sexism-pytorch"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-sexism-pytorch"
        ).to("cuda")

    # get sentiment-analysis results
    results = []
    for sub in subs:
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
        probability = "{:2f}".format(probs[0][pred_label_idx.item()].item())
        result = {
            'pred_label_idx': pred_label_idx.item(),
            'pred_label': pred_label,
            'text': sub.text,
            'time': "{}:{:02d}:{:02d},{:02d} --> {}:{:02d}:{:02d},{:02d}".format(
                sub.start.hours, sub.start.minutes, sub.start.seconds, sub.start.milliseconds,
                sub.end.hours, sub.end.minutes, sub.end.seconds, sub.end.milliseconds),
            'probability': probability
        }
        if float(probability) >= 0.7 and pred_label_idx.item() == 1:
            results.append(result)

    del model
    del tokenizer

    return results


if __name__ == '__main__':
    pass
