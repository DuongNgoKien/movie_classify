import nltk
import time

from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM)

def summary_infer(file_path, path_save):
    """Summary inferences.

    Args:
        file_path (str): input sub file.
        path_save (_type_): summry file save path

    Returns:
        return summary file save path for api inferences.
    """
    start_time = time.time()

    with open(file_path, "r") as f:
        texts = f.read().strip("\n")

    tokenizer = AutoTokenizer.from_pretrained(
        "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/summary-bart-large-cnn"
    )
    sentences = nltk.tokenize.sent_tokenize(texts)
    
    # seperate the file if the file is too long compare to the model
    length = 0
    chunk = ""
    chunks = []
    count =- 1
    
    for sentence in sentences:
        count += 1
        combined_length = len(tokenizer.tokenize(sentence)) + length
        
        if combined_length <= tokenizer.max_len_single_sentence:
            chunk += sentence + ""
            length = combined_length
            if count == len(sentences) -1:
                chunks.append(chunk.strip())
        else:
            chunks.append(chunk.strip())
            
            length = 0
            chunk = ""
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))
        
    inputs = [
        tokenizer(chunk,return_tensors="pt").to("cuda") for chunk in chunks
    ]

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/summary-bart-large-cnn"
    ).to("cuda")
    with open("summary.txt", "w") as f:
        pass
    for input in inputs:
        # generate text summary
        outputs = model.generate(**input, max_length=512)
        text_summary = tokenizer.decode(*outputs, skip_special_tokens=True)
        
        # save the summary
        with open(path_save, "a") as f:
            f.write(text_summary)

    # time caculate for debug
    end_time = time.time()
    execution_time = end_time - start_time

    minutes, seconds = divmod(execution_time, 60)
    time_format = "{:02d}:{:02d}".format(int(minutes), int(seconds))
    
    print(time_format)
    
    del tokenizer, model

    return path_save