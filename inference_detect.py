import torch, os
import pandas as pd
import pysrt
import json
import random
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer
)

category_id = random.randint(1,12)
def sentiment_analysis_inference(sub_file_path):
    """Text sentiment analysis inferences.

    Args:
        sub_file_path (str or like_path): path to sub file.
    """
    # get sub
    subs = pysrt.open(sub_file_path)
    
    # initial tokenizer and model
    if category_id == 5:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-horror-action"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-horror-action"
        ).to("cuda")
    
    elif category_id == 10:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-political-racial"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-political-racial"
        ).to("cuda")
    
    elif category_id == 11:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-political-racial"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-political-racial"
        ).to("cuda")
    
    elif category_id == 9 or category_id == 8 or category_id == 6 or category_id == 7:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-toxic-1"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-toxic-1"
        ).to("cuda")
    
    elif category_id == 2:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-horror-action"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-horror-action"
        ).to("cuda")
    
    elif category_id == 4:
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-sexism"
        )
        model = BertForSequenceClassification.from_pretrained(
            "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/Bert-sexism"
        ).to("cuda")
        
    # get sentiment-analysis results
    results = []
    for sub in subs:
        inputs = tokenizer(
            sub.text,
            padding = True,
            truncation = True,
            max_length = 512,
            return_tensors = "pt"
        ).to("cuda")
        outputs = model(**inputs)
        probs = outputs[0].softmax(1)
        pred_label_idx = probs.argmax()
        pred_label = model.config.id2label[pred_label_idx.item()]
        result = {
            'pred_label_idx': pred_label_idx.item(),
            'pred_label': pred_label,
            'text': sub.text,
            'start': {
                'minutes': sub.start.minutes,
                'seconds': sub.start.seconds
            },
            'end': {'minutes': sub.end.minutes, 'seconds': sub.end.seconds}
        }
        
        results.append(result)
    
    del model
    del tokenizer
    
    return results

 
if __name__ == '__main__':
    pass
    
    

