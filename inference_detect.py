import torch, os
import pandas as pd
import pysrt
import json

from transformers import (
    BertForSequenceClassification,
    AutoTokenizer
)


def sentiment_analysis_inference(sub_file_path):
    """Text sentiment analysis inferences.

    Args:
        sub_file_path (str or like_path): path to sub file.
    """
    # get sub
    subs = pysrt.open(sub_file_path)
    
    # initial tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "path_to_model"
    )
    model = BertForSequenceClassification.from_pretrained(
        "path_to_model"
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
    
    return results

 
if __name__ == '__main__':
    pass
    
    

