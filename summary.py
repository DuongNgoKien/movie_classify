import nltk
import time
import pysrt
import requests
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM)
import re
nltk.download('punkt')
ROOT_API = "http://183.81.35.24:32774"
CONTENT_COMMAND_UPDATE_STATUS_API = f"{ROOT_API}/Content_Command/Update_Status"
CONTENT_CATEGORY_UPDATE_API = f"{ROOT_API}/Content_Category/Create"


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
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/summary-bart-large-cnn"
    )
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

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/www/data/data/saigonmusic/Dev_AI/thainh/MODEL/summary-bart-large-cnn"
    ).to("cuda")

    for input in inputs:
        # generate text summary
        outputs = model.generate(**input, max_length=512)
        text_summary = tokenizer.decode(*outputs, skip_special_tokens=True)
        outputs_list.append(text_summary)

    del tokenizer, model

    return outputs_list


def update_status(id, content_id, status):
    api = (
        CONTENT_COMMAND_UPDATE_STATUS_API
    )
    requests.post(
        api, json={"id": id, "content_id": content_id, "status": status})


def main():
    content_list = requests.get(f"{ROOT_API}/Content_Command/Get_Wait")
    content_list = content_list.json()
    content_list = [
        content for content in content_list if content["status"] == "wait"
    ]
    for content in content_list:
        id = content["id"]
        content_id = content["content_id"]
        command = content["command"]
        content_script = requests.get(
            f"{ROOT_API}/Content_Script/Get_By_Content_Id/{content_id}"
        ).json()["script"]

        update_status(
            id=id,
            status="processing"
        )

        # process
        summary_output = summary_infer(
            script=content_script
        )
        # post result
        requests.post(
            CONTENT_CATEGORY_UPDATE_API,
            json={
                "id": id,
                "content_id": content_id,
                "content": summary_output
            }
        )
        update_status(
            id=id,
            status="done"
        )


if __name__ == "__main__":
    main()
