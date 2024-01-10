import torch, os
import pandas as pd
from transformers import pipeline,BertForSequenceClassification, BertTokenizerFast,AutoTokenizer
import pysrt
import json

subs = pysrt.open("D:\Work\Insidious I (2010).srt")
# list_sub=[subs[1],subs[2],subs[3]]

tokenizer=AutoTokenizer.from_pretrained("D:\Work\MODEL\Bert-horor-action")
model=BertForSequenceClassification.from_pretrained("D:\Work\MODEL\Bert-horor-action").to("cuda")
output=pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)

def inference(subs):
    results = []
    for sub in subs:
        inputs=tokenizer(sub.text,padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda")
        outputs=model(**inputs)
        probs=outputs[0].softmax(1)
        pred_label_idx=probs.argmax()
        pred_label=model.config.id2label[pred_label_idx.item()]
        result = {
                'pred_label_idx': pred_label_idx.item(),
                'pred_label': pred_label,
                'text': sub.text,
                'start': {'minutes': sub.start.minutes, 'seconds': sub.start.seconds},
                'end': {'minutes': sub.end.minutes, 'seconds': sub.end.seconds}
            }
        
        # if pred_label=="Thriller" or pred_label=="Horror":
        results.append(result)
    
    with open('results.json', 'w') as f:
        json.dump(results, f)
        
if __name__ == '__main__':
    inference(subs)
    
    

