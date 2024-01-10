import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor
)
from peft import PeftModel, PeftConfig


def lora_whisper_infer(audio_path):
    """Inferences with lora finetuning whisper large v2.

    Args:
        audio_path (str): audio path for inferences.
    """
    peft_model_id = "temp/checkpoint-500"
    language = "vi"
    task = "transcribe"
    
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        load_in_8bit=True,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(model, peft_model_id)
    
    tokenizer = WhisperTokenizer.from_pretrained(
        peft_config.base_model_name_or_path,
        language=language,
        task=task
    )
    processor = WhisperProcessor.from_pretrained(
        peft_config.base_model_name_or_path,
        language=language,
        task=task
    )
    feature_extractor = processor.feature_extractor
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task=task
    )
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor
    )
    
    with torch.cuda.amp.autocast():
        text = pipe(
            audio_path,
            generate_kwargs={"forced_decoder_ids": forced_decoder_ids},
            max_new_tokens=255
        )
        print(text)


if __name__ == "__main__":
    pass