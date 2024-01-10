from transformers import pipeline, AutoTokenizer,AutoModelForSeq2SeqLM,PegasusForConditionalGeneration, PegasusTokenizer
import pysrt
import nltk
import time
import datetime

start_time = time.time()

# file_path=""
# subs=pysrt.open(file_path)
# for sub in subs:
#     print(sub.text)

file_path="D:\Work\INPUT\Sub_txt\\the_help.txt"
with open(file_path,"r") as f:
    a=f.read().strip("\n")
# tokenizer=AutoTokenizer.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")
# sentences=nltk.tokenize.sent_tokenize(a)

tokenizer=AutoTokenizer.from_pretrained("D:\Work\MODEL\summary-bart-large-cnn")
sentences=nltk.tokenize.sent_tokenize(a)
#Seperate the file if the file is too long compare to the model
length=0
chunk=""
chunks=[]
count=-1
for sentence in sentences:
    count+=1
    combined_length=len(tokenizer.tokenize(sentence)) +length
    
    if combined_length <= tokenizer.max_len_single_sentence:
        chunk+=sentence + ""
        length=combined_length
        if count==len(sentences) -1:
            chunks.append(chunk.strip())
    else:
        chunks.append(chunk.strip())
        
        length=0
        chunk=""
        chunk+=sentence+ " "
        length=len(tokenizer.tokenize(sentence))
    

inputs=[tokenizer(chunk,return_tensors="pt").to("cuda") for chunk in chunks]
# model=AutoModelForSeq2SeqLM.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")
model=AutoModelForSeq2SeqLM.from_pretrained("D:\Work\MODEL\summary-bart-large-cnn").to("cuda")
with open("summary.txt","w") as f:
    pass
for input in inputs:
    outputs=model.generate(**input,max_length=512)
    with open("summary.txt","a") as f:
        f.write(tokenizer.decode(*outputs,skip_special_tokens=True))

end_time = time.time()
execution_time = end_time - start_time

minutes, seconds = divmod(execution_time, 60)
time_format = "{:02d}:{:02d}".format(int(minutes), int(seconds))

del tokenizer, model

print("Execution time:", time_format)