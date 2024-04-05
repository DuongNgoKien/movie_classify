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
import nltk.data

CATEGORY_MAP = {
    "1": "Action",
    "4": "Sexism",
    "5": "Horror",
    "6": "Racist",
    "10": "Political",
    "11": "Religious",
    "12": "Toxic"
}


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
        print(f"[TEXT_PROCESSING]: Skip with category_id: {category_id}")
        return []
    else:
        print(
            f"[TEXT_PROCESSING]: Start process category: {CATEGORY_MAP[str(category_id)]}")
        
        subs = pysrt.open(sub_file_path)
        length = 0
        
        if category_id == 6 or category_id == 12:
            path = "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-6-12-pytorch"
            model = AutoModelForSequenceClassification.from_pretrained(
                path, from_tf=True).to("cuda")
            tokenizer = AutoTokenizer.from_pretrained(path)
        else:
            path_to_model = "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Pytorch-model/bert-{}-pytorch".format(
                category_id)
            tokenizer = AutoTokenizer.from_pretrained(path_to_model)
            model = AutoModelForSequenceClassification.from_pretrained(
                path_to_model).to("cuda")
        
        chunk = ""
        chunks = []
        current_chunk_subs = []
        for sub in subs:
            for sentence in nltk.sent_tokenize(sub.text):
                combined_length = len(tokenizer.tokenize(sentence)) + length
                if combined_length <= tokenizer.max_len_single_sentence and len(chunk.split(".")) <= 2:
                    chunk += sentence + " "
                    length = combined_length
                    current_chunk_subs.append(sub)
                else:
                    chunks.append(
                        (chunk.strip(), current_chunk_subs[0], current_chunk_subs[-1]))
                    length = len(tokenizer.tokenize(sentence))
                    chunk = sentence + " "
                    current_chunk_subs = [sub]
        if chunk:
            chunks.append(
                (chunk.strip(), current_chunk_subs[0], current_chunk_subs[-1]))

        results = []

        for chunk, start_sub, end_sub in tqdm(chunks, desc="Predicting", colour="cyan"):
            inputs = tokenizer(
                chunk,
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
            # result = {
            #     'pred_label_idx': pred_label_idx.item(),
            #     'pred_label': pred_label,
            #     'text': sub.text.replace("\n", " "),
            #     'time': "{}:{:02d}:{:02d},{:02d} --> {}:{:02d}:{:02d},{:02d}".format(
            #         start_sub.start.hours, start_sub.start.minutes, start_sub.start.seconds, start_sub.start.milliseconds,
            #         end_sub.end.hours, end_sub.end.minutes, end_sub.end.seconds, end_sub.end.milliseconds),
            #     'probability': probability
            # }
            result = {
                'pred_label_idx': pred_label_idx.item(),
                'pred_label': pred_label,
                'text': chunk.replace("\n", " "),
                'time': f"{start_sub.start.hours:02d}:{start_sub.start.minutes:02d}:{start_sub.start.seconds:02d},{start_sub.start.milliseconds:03d} --> {end_sub.end.hours:02d}:{end_sub.end.minutes:02d}:{end_sub.end.seconds:02d},{end_sub.end.milliseconds:03d}",
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
        
        print(results)
        return results


if __name__ == '__main__':
    pass
