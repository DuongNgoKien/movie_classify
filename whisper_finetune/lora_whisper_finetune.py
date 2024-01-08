import os
import torch
import evaluate
import transformers

from typing import Any, Dict, List, Union
from dataclasses import dataclass
from tqdm import tqdm
from typing import DefaultDict
from datasets import load_dataset, DatasetDict, Audio, Dataset, load_from_disk
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    PeftModel,
    LoraModel,
    get_peft_model
)
from transformers import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperForConditionalGeneration
)

# ignore warning in transformer
transformers.logging.set_verbosity_error()

# define eature extractor
FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-large-v3"
)

# define tokenizer
TOKENIZER = WhisperTokenizer.from_pretrained(
    "openai/whisper-large-v3",
    language="vi",
    task="transcribe"
)

# define metric
METRIC = evaluate.load("wer")


def make_dataset(
    data_path: str = "data/vlsp2020_train_set_02",
    phase="train",
):
    """
    Make dataset from json file path
    Args:
        json_file_path (str): path to train or valid json data file. Defaults
        to "data/vlsp2022_train_set_02".
        phase (str): train or validation phase. Defaults to train.
    """
    files = os.listdir(data_path)
    audio_files = [f for f in files if f.endswith("wav")]
    ratio_num = int(len(audio_files) * 0.9)
    audio_files = (
        audio_files[:10000] if phase == "train" else audio_files[10000:13000]
    )
    audio_paths = [os.path.join(data_path, f) for f in audio_files]

    dataset = Dataset.from_dict(
        {"audio": audio_paths}
    ).cast_column("audio", Audio(sampling_rate=16000))

    return dataset


def prepare_dataset(batch):
    """
    Load, compute log-mel input features from input audio and encode target
    text to label ids.
    Args:
        batch (_type_): batch of datasets.
    """
    # load audio
    audio = batch["audio"]
    # compute log-mel input features
    batch["input_features"] = FEATURE_EXTRACTOR(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # encode text to label ids
    text_file_path = audio["path"].replace("wav", "txt")
    with open(text_file_path, "r") as f:
        batch["labels"] = TOKENIZER(f.read()).input_ids

    return batch


@dataclass
class DataCollectorSpeechSeq2SeqWithPadding:
    """Define data collector."""
    processor: Any

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # tredat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (
            labels[:, 0] == self.processor.tokenizer.bos_token_id
        ).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad token id
    label_ids[label_ids == -100] = TOKENIZER.pad_token_id

    pred_str = TOKENIZER.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = TOKENIZER.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * METRIC.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def main():
    # initialize 8 bit model for training
    model = WhisperForConditionalGeneration.from_pretrained(
        "/home/www/data/data/saigonmusic/Dev_AI/manhvd/recommend_system/weights/whisper_large_v3",
        load_in_8bit=True
    )
    model.config.forced_decoders_ids = None
    model.config.suppress_tokens = []
    model = prepare_model_for_int8_training(model)
    
    # setup LoRA
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    


if __name__ == "__main__":
    main()
