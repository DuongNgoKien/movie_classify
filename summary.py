import nltk
import time
import pysrt
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from tqdm import tqdm

nltk.download('punkt')

ROOT_API = "http://183.81.35.24:32774"
CONTENT_COMMAND_UPDATE_STATUS_API = f"{ROOT_API}/content_command/update_status"
CONTENT_CATEGORY_UPDATE_API = f"{ROOT_API}/content_category/create"
CONTENT_COMMAND_PROGRESS = f"{ROOT_API}/content_command/update_progress"


def translate_en2vi(en_texts: str,model,tokenizer) -> str:
    """Translate from english text to Vietnamese text.

    Args:
        en_texts (str): English text
        model (models): Translation model.
        tokenizer (tokenizers): Tokenizer model.

    Returns:
        str: _description_
    """
    input_ids = tokenizer(
        en_texts, padding=True, return_tensors="pt"
    ).to("cuda")
    output_ids = model.generate(
        **input_ids,
        decoder_start_token_id=tokenizer.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    return vi_texts
    
    
def summary_infer(script):
    """Summary inferences.

    Args:
        file_path (str): input sub file.
        path_save (_type_): summry file save path

    Returns:
        return summary file save path for api inferences.
    """
    outputs_list = []
    # Choose the process step based on the input is text or file
#    subs = pysrt.open(file_path)
    first_text = script
    texts = re.sub(r'\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+\n', '', first_text)
#   texts = script.strip("\n")
    path_sum = "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/summary-led-base-book"
    tokenizer = AutoTokenizer.from_pretrained(path_sum)
    sentences = nltk.tokenize.sent_tokenize(texts)

    # seperate the file if the file is too long compare to the model
    length = 0
    chunk = ""
    chunks = []
    count = - 1

    for sentence in sentences:
        count += 1
        combined_length = len(tokenizer.tokenize(sentence)) + length

        if combined_length <= tokenizer.max_len_single_sentence:
            chunk += sentence + ""
            length = combined_length
            if count == len(sentences) - 1:
                chunks.append(chunk.strip())
        else:
            chunks.append(chunk.strip())

            length = 0
            chunk = ""
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))

    inputs = [
        tokenizer(chunk, return_tensors="pt").to("cuda") for chunk in chunks
    ]

    model = AutoModelForSeq2SeqLM.from_pretrained(path_sum).to("cuda")

    for word in tqdm(inputs, desc="Summary", colour="cyan"):
        # generate text summary
        outputs = model.generate(**word, max_length=1024)
        text_summary = tokenizer.decode(*outputs, skip_special_tokens=True)
        outputs_list.append(text_summary)

    del tokenizer, model
    
    # path_trans = "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/models--vinai--vinai-translate-en2vi-v2"
    tokenizer_1 = AutoTokenizer.from_pretrained(
        "vinai/vinai-translate-en2vi-v2", 
        cache_dir="/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL"
    )
    model_1 = AutoModelForSeq2SeqLM.from_pretrained(
        "vinai/vinai-translate-en2vi-v2", 
        cache_dir="/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL"
    ).to("cuda")

    trans_list = []
    for trans in tqdm(outputs_list, desc="Translate", colour="cyan"):
        trans_list.append(
            translate_en2vi(trans, model=model_1,tokenizer=tokenizer_1)
        )
    
    joined_text = '.'.join([sub[0] for sub in trans_list])
    
    del tokenizer_1, model_1

    return joined_text


def update_status(id, status):
    api = (
        CONTENT_COMMAND_UPDATE_STATUS_API
    )
    requests.put(
        api, params={"id": id, "status": status})


def main():
    content_list = requests.get(f"{ROOT_API}/content_command/get_wait")
    content_list = content_list.json()
    content_list = [
        content for content in content_list if content["command"] == "summarize"
    ]

    for content in content_list:
        id_list = content["id"]
        content_id = content["content_id"]
        command = content["command"]
        content_script = requests.get(
            f"{ROOT_API}/content_script/get_by_content_id/{content_id}"
        ).json()["script"]

        update_status(
            id=id_list,
            status="processing"
        )

        # process
        summary_output = summary_infer(
            script=content_script
        )

        requests.put(
            CONTENT_COMMAND_PROGRESS,
            params={
                "id": id_list,
                "new_note": str(summary_output),
                "progress": 100
            }   
        )
        
        update_status(
            id=id_list,
            status="done"
        )


if __name__ == "__main__":
    main()
