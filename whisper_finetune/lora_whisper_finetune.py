import os
import librosa
import torch
import numpy as np
import evaluate
import transformers

from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Union
from dataclasses import dataclass
from typing import DefaultDict
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    PeftConfig,
    PeftModel,
    LoraModel,
    get_peft_model
)
from transformers import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


# ignore warning in transformer
transformers.logging.set_verbosity_error()

# define eature extractor
FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(
    "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/weights/whisper_large_v2",
    local_files_only=True
)

# define tokenizer
TOKENIZER = WhisperTokenizer.from_pretrained(
    "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/weights/whisper_large_v2",
    local_files_only=True,
    language="vi",
    task="transcribe"
)

# define metric
METRIC = evaluate.load("wer")


class Vin100hDataset(Dataset):
    def __init__(self, dataset_path, split_ratio=0.9, phase="train"):
        self.dataset_path = dataset_path
        audio_files = os.listdir(dataset_path)
        self.audio_file_lst = []
        if phase == "train":
            audio_files = audio_files[:int(len(audio_files) * split_ratio)]
            for audio_file in audio_files:
                if audio_file.split(".")[-1] == "wav":
                    self.audio_file_lst.append(audio_file)
        else:
            audio_files = audio_files[int(len(audio_files) * split_ratio):]
            for audio_file in audio_files:
                if audio_file.split(".")[-1] == "wav":
                    self.audio_file_lst.append(audio_file)
                    
    def __len__(self):
        return len(self.audio_file_lst)
    
    
    def __getitem__(self, index):
        batch = DefaultDict()
        # get audio and label text
        audio_path = os.path.join(self.dataset_path, self.audio_file_lst[index])
        label_path = audio_path.replace("wav", "txt")
        
        # extract audio features
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = FEATURE_EXTRACTOR(audio, sampling_rate=sr).input_features[0]
        batch["input_features"] = audio
        
        # decode target text to label ids
        with open(label_path, "r") as f:
            text = f.read()
            batch["labels"] = TOKENIZER(text).input_ids
            
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
    """Compute the metrics.

    Args:
        pred (dict): model predictions.

    Returns:
        dict: word error rate.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad token id
    label_ids[label_ids == -100] = TOKENIZER.pad_token_id

    pred_str = TOKENIZER.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = TOKENIZER.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * METRIC.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Save checkpoint paths.

        Args:
            args (TrainingArguments): Training arguments.
            state (TrainerState): Training state.
            control (TrainerControl): Training control.
        """
        checkpoint_folder = os.path.join(
            args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        
        pytorch_model_path = os.path.join(
            checkpoint_folder,
            "pytorch_model.bin"
        )
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        
        return control


def evaluate():
    # initial necessary model
    peft_model_id = "weights_to_peft_model"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        load_in_8bit=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    
    # processor initial
    processor = WhisperProcessor.from_pretrained(
        "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/weights/whisper_large_v2",
        language="vi",
        task="transcribe"
    )
    
    # data_collator
    data_collator = DataCollectorSpeechSeq2SeqWithPadding(processor=processor)
    
    eval_dataset = Vin100hDataset("data/vlsp2020_train_set_02", phase="test")
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=16,
        collate_fn=data_collator
    )
    
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                        max_new_tokens=255
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels=  np.where(
                    labels != -100,
                    labels,
                    TOKENIZER.pad_token_id
                )
                decoded_preds = TOKENIZER.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True
                )
                decoded_labels = TOKENIZER.batch_decode(
                    labels,
                    skip_special_tokens=True
                )
                METRIC.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels
                )
        del generated_tokens, labels, batch
    
    wer = 100 * METRIC.compute()
    
    return wer

def main():
    # initialize 8 bit model for training
    model = WhisperForConditionalGeneration.from_pretrained(
        "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/weights/whisper_large_v2",
        local_files_only=True,
        load_in_8bit=True
    )
    model.config.forced_decoders_ids = None
    model.config.suppress_tokens = []
    model = prepare_model_for_int8_training(model)
    
    # create train/test dataset
    dataset_path = "data/vlsp2020_train_set_02"
    train_dataset = Vin100hDataset(dataset_path, phase="train")
    test_dataset = Vin100hDataset(dataset_path, phase="test")
    
    # processor initial
    processor = WhisperProcessor.from_pretrained(
        "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/weights/whisper_large_v2",
        language="vi",
        task="transcribe"
    )
    
    # data_collator
    data_collator = DataCollectorSpeechSeq2SeqWithPadding(processor=processor)
    
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
    
    # training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="temp",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=50,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        fp16=True,
        per_device_eval_batch_size=8,
        generation_max_length=128,
        logging_steps=25,
        remove_unused_columns=False,
        label_names=["labels"]
    )
    
    # trainer initialize
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback]
    )
    model.config.use_cache = False
    
    # train
    trainer.train()
    
    
if __name__ == "__main__":
    main()
